"""
Export Qwen2-VL-2B int4 quantized ONNX model using onnxruntime quantization.

Tested versions:
    onnxruntime 1.24.4
    onnx 1.20.1

Requirements:
    pip install onnxruntime onnx numpy

Usage:
    python export_olive_int4.py
    python export_olive_int4.py --output_dir /path/to/output

This script quantizes the Qwen2-VL-2B LLM decoder to int4 (4-bit weight-only
quantization). The vision encoder is kept at fp32 precision.

The quantization uses onnxruntime's MatMulNBitsQuantizer which is the same
engine used by Microsoft Olive for int4 weight quantization.
The quantized model uses the com.microsoft:MatMulNBits operator.

Steps:
  1. Download the fp16 ONNX model from ailia-models storage
  2. Convert fp16 -> fp32 for quantization compatibility
  3. Quantize all MatMul weights to int4 (block_size=128, symmetric)
  4. Fix any remaining fp16 type references
  5. Save the quantized model as a single ONNX file
  6. Generate prototxt
"""

import os
import sys
import argparse
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.quantization.quant_utils import QuantFormat


def download_model(model_name, remote_path):
    """Download model file from remote storage if not present."""
    if os.path.exists(model_name):
        print(f"  {model_name} already exists, skipping download.")
        return
    url = remote_path + model_name
    print(f"  Downloading {model_name} ...")
    subprocess.check_call(["wget", "-q", url, "-O", model_name])


def convert_fp16_to_fp32(model):
    """Convert all fp16 elements in the model to fp32."""
    # Convert initializers
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new_t = numpy_helper.from_array(arr, init.name)
            init.CopyFrom(new_t)

    # Convert graph inputs, outputs, and value_info
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

    # Convert Cast and Constant nodes
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
    onnx.save(quant.model.model, output_model_path)


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
        description="Export Qwen2-VL-2B int4 quantized model using Olive/onnxruntime"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="..",
        help="Output directory for quantized model files (default: parent dir)",
    )
    args = parser.parse_args()

    remote_path = "https://storage.googleapis.com/ailia-models/qwen2_vl/"
    work_dir = os.path.abspath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    # Step 1: Download original fp16 ONNX model
    print("[1/4] Downloading original fp16 model ...")
    original_model = "Qwen2-VL-2B.opt.onnx"
    original_pb = "Qwen2-VL-2B_weights.pb"
    download_model(original_model, remote_path)
    download_model(original_pb, remote_path)

    # Step 2: Load and convert fp16 -> fp32
    print("[2/4] Loading and converting fp16 -> fp32 ...")
    model = onnx.load(original_model, load_external_data=True)
    convert_fp16_to_fp32(model)

    # Step 3: Quantize LLM decoder to int4
    print("[3/4] Quantizing LLM decoder to int4 ...")
    quantized_model = os.path.join(work_dir, "Qwen2-VL-2B_int4.onnx")
    quantize_int4(model, quantized_model)
    del model

    # Step 4: Generate prototxt
    print("[4/4] Generating prototxt ...")
    prototxt_path = generate_prototxt(quantized_model)

    print("\nDone! Generated files:")
    print(f"  - {quantized_model}")
    print(f"  - {prototxt_path}")
    print("\nNote: The vision encoder uses fp32 model (Qwen2-VL-2B_vis.onnx).")
    print(
        "Upload the generated files to:"
        " https://console.cloud.google.com/storage/browser/ailia-models"
    )


if __name__ == "__main__":
    main()
