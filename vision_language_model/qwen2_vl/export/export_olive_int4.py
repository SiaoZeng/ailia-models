"""
Export Qwen2-VL-2B int4 quantized ONNX model using Microsoft Olive.

Requirements:
    pip install olive-ai[auto-opt] onnxruntime

Usage:
    python export_olive_int4.py

This script quantizes the Qwen2-VL-2B LLM decoder to int4 using Olive.
The vision encoder is kept at fp16 precision.
"""

import os
import sys
import argparse
import subprocess
import shutil


def download_model(model_name, remote_path):
    """Download model file from remote storage if not present."""
    if os.path.exists(model_name):
        print(f"  {model_name} already exists, skipping download.")
        return
    url = remote_path + model_name
    print(f"  Downloading {model_name} ...")
    subprocess.check_call(["wget", "-q", url, "-O", model_name])


def quantize_with_olive(input_model, output_dir):
    """Quantize ONNX model to int4 using Olive CLI."""
    print(f"  Quantizing {input_model} to int4 with Olive ...")
    cmd = [
        "olive", "quantize",
        "-m", input_model,
        "--precision", "int4",
        "-o", output_dir,
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def find_quantized_model(output_dir):
    """Find the quantized ONNX model in Olive output directory."""
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".onnx"):
                return os.path.join(root, f)
    return None


def generate_prototxt(onnx_path):
    """Generate prototxt from ONNX model using onnx2prototxt.py."""
    prototxt_path = onnx_path + ".prototxt"
    script_url = "https://raw.githubusercontent.com/ailia-ai/export-to-onnx/master/onnx2prototxt.py"
    script_path = "onnx2prototxt.py"

    if not os.path.exists(script_path):
        print(f"  Downloading onnx2prototxt.py ...")
        subprocess.check_call(["wget", "-q", script_url, "-O", script_path])

    print(f"  Generating prototxt for {onnx_path} ...")
    subprocess.check_call([
        sys.executable, script_path, onnx_path, prototxt_path
    ])
    return prototxt_path


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen2-VL-2B int4 quantized model using Olive"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for quantized model files",
    )
    args = parser.parse_args()

    remote_path = "https://storage.googleapis.com/ailia-models/qwen2_vl/"
    work_dir = os.path.abspath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Step 1: Download original fp16 ONNX model (smaller base for quantization)
    print("[1/4] Downloading original model ...")
    original_model = "Qwen2-VL-2B_fp16.onnx"
    original_pb = "Qwen2-VL-2B_weights_fp16.pb"
    os.chdir(work_dir)
    download_model(original_model, remote_path)
    download_model(original_pb, remote_path)

    # Step 2: Quantize LLM decoder to int4 with Olive
    print("[2/4] Quantizing LLM decoder to int4 ...")
    olive_output_dir = os.path.join(work_dir, "olive_output")
    quantize_with_olive(original_model, olive_output_dir)

    # Step 3: Copy quantized model to output
    print("[3/4] Copying quantized model ...")
    quantized_src = find_quantized_model(olive_output_dir)
    if quantized_src is None:
        print("ERROR: Could not find quantized ONNX model in Olive output.")
        sys.exit(1)

    quantized_dst = os.path.join(work_dir, "Qwen2-VL-2B_int4.onnx")
    shutil.copy2(quantized_src, quantized_dst)
    print(f"  Quantized model saved to: {quantized_dst}")

    # Also copy any external data files
    quantized_data = quantized_src + ".data"
    if os.path.exists(quantized_data):
        shutil.copy2(quantized_data, quantized_dst + ".data")

    # Step 4: Generate prototxt
    print("[4/4] Generating prototxt ...")
    prototxt_path = generate_prototxt(quantized_dst)
    print(f"  Prototxt saved to: {prototxt_path}")

    # Cleanup
    if os.path.exists(olive_output_dir):
        shutil.rmtree(olive_output_dir)

    print("\nDone! Generated files:")
    print(f"  - {quantized_dst}")
    print(f"  - {prototxt_path}")
    print("\nNote: The vision encoder uses fp16 model (Qwen2-VL-2B_vis_fp16.onnx).")
    print("Upload the generated files to: https://console.cloud.google.com/storage/browser/ailia-models")


if __name__ == "__main__":
    main()
