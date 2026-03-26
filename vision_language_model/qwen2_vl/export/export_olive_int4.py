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

This script quantizes both the Qwen2-VL-2B LLM decoder and vision encoder
to int4 (4-bit weight-only quantization).

The quantization uses onnxruntime's MatMulNBitsQuantizer which is the same
engine used by Microsoft Olive for int4 weight quantization.
The quantized model uses the com.microsoft:MatMulNBits operator.

For the vision encoder, Gemm nodes are first converted to MatMul+Add with
weight tensors transposed in-place, since MatMulNBitsQuantizer only supports
MatMul with constant weights.

Steps:
  1. Download the fp32 ONNX models from ailia-models storage
  2. Quantize all MatMul weights to int4 (block_size=128, symmetric)
  3. Save the quantized models
  4. Generate prototxt
"""

import os
import sys
import argparse
import subprocess

import onnx
from onnx import numpy_helper
from onnx.external_data_helper import convert_model_to_external_data
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


def quantize_int4(input_model_path, output_model_path):
    """Quantize ONNX model to int4 using MatMulNBitsQuantizer."""
    print("  Quantizing to int4 (block_size=128, symmetric) ...")
    quant = MatMulNBitsQuantizer(
        model=input_model_path,
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


def convert_gemm_to_matmul(model):
    """Convert Gemm nodes to MatMul+Add with in-place weight transpose.

    MatMulNBitsQuantizer does not support Gemm, so we convert Gemm nodes to
    MatMul (with transposed weight initializers) + Add (for bias).
    """
    graph = model.graph

    # Clear external data references so weights are accessible as inline data
    for tensor in graph.initializer:
        while len(tensor.external_data) > 0:
            tensor.external_data.pop()
        tensor.ClearField("data_location")

    init_map = {t.name: i for i, t in enumerate(graph.initializer)}
    nodes_to_remove = []
    nodes_to_add = []

    for node in graph.node:
        if node.op_type != "Gemm":
            continue
        transB = 0
        for attr in node.attribute:
            if attr.name == "transB":
                transB = attr.i

        A = node.input[0]
        B = node.input[1]

        if transB and B in init_map:
            # Transpose weight data in the initializer
            idx = init_map[B]
            w = numpy_helper.to_array(graph.initializer[idx])
            w_t = w.T.copy()
            new_name = B + "_transposed"
            graph.initializer.append(numpy_helper.from_array(w_t, name=new_name))
            B = new_name
        elif transB:
            transpose_out = f"{node.name}_transB_out"
            nodes_to_add.append(
                onnx.helper.make_node(
                    "Transpose",
                    inputs=[B],
                    outputs=[transpose_out],
                    name=f"{node.name}_transpose",
                    perm=[1, 0],
                )
            )
            B = transpose_out

        matmul_out = f"{node.name}_matmul_out"
        if len(node.input) > 2 and node.input[2]:
            nodes_to_add.append(
                onnx.helper.make_node(
                    "MatMul",
                    inputs=[A, B],
                    outputs=[matmul_out],
                    name=f"{node.name}_matmul",
                )
            )
            nodes_to_add.append(
                onnx.helper.make_node(
                    "Add",
                    inputs=[matmul_out, node.input[2]],
                    outputs=node.output,
                    name=f"{node.name}_add",
                )
            )
        else:
            nodes_to_add.append(
                onnx.helper.make_node(
                    "MatMul",
                    inputs=[A, B],
                    outputs=node.output,
                    name=f"{node.name}_matmul",
                )
            )
        nodes_to_remove.append(node)

    for node in nodes_to_remove:
        graph.node.remove(node)
    for node in nodes_to_add:
        graph.node.append(node)

    print(f"  Converted {len(nodes_to_remove)} Gemm nodes to MatMul+Add")
    return model


def quantize_vis_int4(input_model_path, output_model_path, output_pb_path):
    """Quantize vision encoder ONNX model to int4.

    The vision encoder uses Gemm nodes which are not supported by
    MatMulNBitsQuantizer directly. This function first converts Gemm to
    MatMul+Add, then quantizes.
    """
    print("  Loading vision encoder model...")
    model = onnx.load(input_model_path)

    print("  Converting Gemm to MatMul+Add...")
    model = convert_gemm_to_matmul(model)

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

    result = quant.model.model
    nbits_count = sum(1 for n in result.graph.node if n.op_type == "MatMulNBits")
    print(f"  MatMulNBits nodes created: {nbits_count}")

    print(f"  Saving quantized model: {output_model_path}")
    convert_model_to_external_data(
        result,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_pb_path),
        size_threshold=0,
    )
    onnx.save(result, output_model_path)


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
        description="Export Qwen2-VL-2B int4 quantized model using onnxruntime"
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

    # Step 1: Download original ONNX models
    print("[1/4] Downloading original models ...")
    original_model = "Qwen2-VL-2B.onnx"
    original_pb = "Qwen2-VL-2B_weights.pb"
    original_vis_model = "Qwen2-VL-2B_vis.opt.onnx"
    original_vis_pb = "Qwen2-VL-2B_vis_weights.pb"
    download_model(original_model, remote_path)
    download_model(original_pb, remote_path)
    download_model(original_vis_model, remote_path)
    download_model(original_vis_pb, remote_path)

    # Step 2: Quantize LLM decoder to int4
    print("[2/4] Quantizing LLM decoder to int4 ...")
    quantized_model = os.path.join(work_dir, "Qwen2-VL-2B_int4.onnx")
    quantize_int4(original_model, quantized_model)

    # Step 3: Quantize vision encoder to int4
    print("[3/4] Quantizing vision encoder to int4 ...")
    quantized_vis_model = os.path.join(work_dir, "Qwen2-VL-2B_vis_int4.onnx")
    quantized_vis_pb = os.path.join(work_dir, "Qwen2-VL-2B_vis_int4_weights.pb")
    quantize_vis_int4(original_vis_model, quantized_vis_model, quantized_vis_pb)

    # Step 4: Generate prototxt
    print("[4/4] Generating prototxt ...")
    prototxt_path = generate_prototxt(quantized_model)
    vis_prototxt_path = generate_prototxt(quantized_vis_model)

    print("\nDone! Generated files:")
    print(f"  - {quantized_model}")
    print(f"  - {prototxt_path}")
    print(f"  - {quantized_vis_model}")
    print(f"  - {vis_prototxt_path}")
    print(
        "\nUpload the generated files to:"
        " https://console.cloud.google.com/storage/browser/ailia-models"
    )


if __name__ == "__main__":
    main()
