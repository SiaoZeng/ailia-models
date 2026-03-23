"""
Export Qwen2-VL int4 quantized ONNX model using onnxruntime quantization.

Tested versions:
    onnxruntime 1.24.4
    onnx 1.20.1

Requirements:
    pip install onnxruntime onnx numpy transformers torch

Usage:
    python export_olive_int4.py --model 2B-int4
    python export_olive_int4.py --model 7B-int4
    python export_olive_int4.py --model 2B-int4 --output_dir /path/to/output

This script exports the Qwen2-VL LLM decoder to ONNX and quantizes it to int4
(4-bit weight-only quantization). The vision encoder is kept at fp32 precision.

The quantization uses onnxruntime's MatMulNBitsQuantizer which is the same
engine used by Microsoft Olive for int4 weight quantization.
The quantized model uses the com.microsoft:MatMulNBits operator.

Steps:
  1. Download/export the ONNX model
     - 2B: Download pre-exported fp16 ONNX from ailia-models storage
     - 7B: Export from HuggingFace PyTorch model via torch.onnx.export
  2. Convert fp16 -> fp32 for quantization compatibility
  3. Quantize all MatMul weights to int4 (block_size=128, symmetric)
  4. Fix any remaining fp16 type references
  5. Save the quantized model as a single ONNX file
  6. Generate prototxt
"""

import os
import sys
import gc
import argparse
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.quantization.quant_utils import QuantFormat

MODEL_CONFIGS = {
    "2B-int4": {
        "hf_name": "Qwen/Qwen2-VL-2B-Instruct",
        "fp16_onnx": "Qwen2-VL-2B_fp16.onnx",
        "fp16_pb": "Qwen2-VL-2B_weights_fp16.pb",
        "output_onnx": "Qwen2-VL-2B_int4.onnx",
        "has_remote_fp16": True,
        "num_hidden_layers": 28,
        "hidden_size": 1536,
    },
    "7B-int4": {
        "hf_name": "Qwen/Qwen2-VL-7B-Instruct",
        "hf_onnx_repo": "pdufour/Qwen2-VL-7B-Instruct-onnx",
        "hf_onnx_file": "onnx/llm.onnx",
        "fp16_onnx": "Qwen2-VL-7B_fp32.onnx",
        "fp16_pb": "Qwen2-VL-7B_fp32_weights.pb",
        "output_onnx": "Qwen2-VL-7B_int4.onnx",
        "has_remote_fp16": False,
        "num_hidden_layers": 28,
        "hidden_size": 3584,
    },
}


def download_model(model_name, remote_path):
    """Download model file from remote storage if not present."""
    if os.path.exists(model_name):
        print(f"  {model_name} already exists, skipping download.")
        return
    url = remote_path + model_name
    print(f"  Downloading {model_name} ...")
    subprocess.check_call(["wget", "-q", url, "-O", model_name])


def download_from_huggingface_onnx(config, output_dir):
    """Download LLM decoder ONNX from HuggingFace ONNX repo.

    Requires ~30GB disk and ~30GB RAM for the 7B model.
    """
    from huggingface_hub import hf_hub_download

    repo_id = config["hf_onnx_repo"]
    onnx_file = config["hf_onnx_file"]
    fp_onnx = os.path.join(output_dir, config["fp16_onnx"])

    if os.path.exists(fp_onnx):
        print(f"  {fp_onnx} already exists, skipping download.")
        return

    print(f"  Downloading {onnx_file} from {repo_id} ...")
    # Download the ONNX model file
    local_onnx = hf_hub_download(repo_id=repo_id, filename=onnx_file)
    # Download the external data file
    data_file = onnx_file + ".data"
    local_data = hf_hub_download(repo_id=repo_id, filename=data_file)

    # Copy to output directory
    import shutil
    shutil.copy2(local_onnx, fp_onnx)
    fp_pb = os.path.join(output_dir, config["fp16_pb"])
    shutil.copy2(local_data, fp_pb)
    print(f"  Downloaded to {fp_onnx}")


def convert_fp16_to_fp32(model):
    """Convert all fp16 elements in the model to fp32."""
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new_t = numpy_helper.from_array(arr, init.name)
            init.CopyFrom(new_t)

    for vi in (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    ):
        if (
            vi.type.HasField("tensor_type")
            and vi.type.tensor_type.elem_type == TensorProto.FLOAT16
        ):
            vi.type.tensor_type.elem_type = TensorProto.FLOAT

    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    new_t = numpy_helper.from_array(arr)
                    attr.t.CopyFrom(new_t)
        if node.op_type == "ConstantOfShape":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    new_t = numpy_helper.from_array(arr)
                    attr.t.CopyFrom(new_t)


def quantize_int4(model, output_model_path):
    """Quantize ONNX model to int4 using MatMulNBitsQuantizer."""
    print("  Quantizing to int4 (block_size=128, symmetric) ...")
    quant = MatMulNBitsQuantizer(
        model=model,
        bits=4,
        block_size=128,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
    )
    quant.process()

    print(f"  Saving quantized model: {output_model_path}")
    pb_name = os.path.basename(output_model_path).replace(".onnx", "_weights.pb")
    onnx.save_model(
        quant.model.model,
        output_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=pb_name,
        size_threshold=1024,
    )


def generate_prototxt(onnx_path):
    """Generate prototxt from ONNX model using onnx2prototxt.py."""
    prototxt_path = onnx_path + ".prototxt"
    script_url = "https://raw.githubusercontent.com/ailia-ai/export-to-onnx/master/onnx2prototxt.py"
    script_path = "onnx2prototxt.py"

    if not os.path.exists(script_path):
        print("  Downloading onnx2prototxt.py ...")
        subprocess.check_call(["wget", "-q", script_url, "-O", script_path])

    print(f"  Generating prototxt for {onnx_path} ...")
    subprocess.check_call([sys.executable, script_path, onnx_path, prototxt_path])
    return prototxt_path


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen2-VL int4 quantized model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to export: 2B-int4 or 7B-int4",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="..",
        help="Output directory for quantized model files (default: parent dir)",
    )
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    remote_path = "https://storage.googleapis.com/ailia-models/qwen2_vl/"
    work_dir = os.path.abspath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    print(f"=== Exporting {args.model} ===\n")

    # Step 1: Get ONNX model
    if config["has_remote_fp16"]:
        print("[1/4] Downloading original fp16 model from ailia-models ...")
        download_model(config["fp16_onnx"], remote_path)
        download_model(config["fp16_pb"], remote_path)
    else:
        print("[1/4] Downloading ONNX model from HuggingFace ...")
        download_from_huggingface_onnx(config, work_dir)

    # Step 2: Load and convert fp16 -> fp32
    print("[2/4] Loading and converting fp16 -> fp32 ...")
    fp16_path = os.path.join(work_dir, config["fp16_onnx"])
    fp16_pb_path = os.path.join(work_dir, config["fp16_pb"])
    model = onnx.load(fp16_path, load_external_data=True)
    # Remove source files to free disk space (data is now in memory)
    if os.path.exists(fp16_path):
        os.remove(fp16_path)
    if os.path.exists(fp16_pb_path):
        os.remove(fp16_pb_path)
    convert_fp16_to_fp32(model)

    # Step 3: Quantize LLM decoder to int4
    print("[3/4] Quantizing LLM decoder to int4 ...")
    quantized_model = os.path.join(work_dir, config["output_onnx"])
    quantize_int4(model, quantized_model)
    del model
    gc.collect()

    # Step 4: Generate prototxt
    print("[4/4] Generating prototxt ...")
    prototxt_path = generate_prototxt(quantized_model)

    print(f"\nDone! Generated files:")
    print(f"  - {quantized_model}")
    print(f"  - {prototxt_path}")
    print(
        "\nUpload the generated files to:"
        " https://console.cloud.google.com/storage/browser/ailia-models"
    )


if __name__ == "__main__":
    main()
