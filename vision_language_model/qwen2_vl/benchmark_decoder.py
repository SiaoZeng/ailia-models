"""
Benchmark Qwen2-VL decoder (single token decode step) for FP32 vs INT4 comparison.
Usage:
    python3 benchmark_decoder.py --model_type fp32
    python3 benchmark_decoder.py --model_type int4
    python3 benchmark_decoder.py --model_type fp32 --onnxruntime
    python3 benchmark_decoder.py --model_type int4 --onnxruntime
"""
import sys
import time
import argparse

import numpy as np

sys.path.append("../../util")
from model_utils import check_and_download_models  # noqa


REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen2_vl/"
NUM_KV = 28
HIDDEN_SIZE = 1536
NUM_HEADS = 2
HEAD_DIM = 128
WARMUP = 3
ITERATIONS = 20


def print_results(label, times, seq_len):
    avg = sum(times) / len(times)
    median = sorted(times)[len(times) // 2]
    min_t = min(times)
    max_t = max(times)

    print(f"\n=== Qwen2-VL Decoder Benchmark ({label}) ===")
    print(f"Seq length: {seq_len}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Average: {avg:.2f} ms")
    print(f"Median:  {median:.2f} ms")
    print(f"Min:     {min_t:.2f} ms")
    print(f"Max:     {max_t:.2f} ms")
    print(f"All times: {[f'{t:.2f}' for t in times]}")


def benchmark_ailia(args, weight_path, model_path, input_ids, inputs_embeds,
                    position_ids, attention_mask, past_key_values, seq_len):
    import ailia

    memory_mode = ailia.get_memory_mode(
        reduce_constant=True,
        ignore_input_with_initializer=True,
        reduce_interstage=False,
        reuse_interstage=True,
    )
    net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=memory_mode)

    # First run to initialize
    output = net.predict([input_ids, inputs_embeds, position_ids, attention_mask, *past_key_values])

    def run_decode_step():
        """Run a single decode step using copy_blob_data for KV cache."""
        key_shapes = []
        value_shapes = []
        for j in range(NUM_KV):
            key_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"key_cache_out{j}")))
            value_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"value_cache_out{j}")))
        net.set_input_blob_data(input_ids, net.find_blob_index_by_name("input_ids"))
        net.set_input_blob_data(inputs_embeds, net.find_blob_index_by_name("inputs_embeds"))
        net.set_input_blob_data(position_ids, net.find_blob_index_by_name("position_ids"))
        cur_kv_len = key_shapes[0][2]
        cur_mask = np.ones((1, cur_kv_len + 1), dtype=np.int64)
        net.set_input_blob_data(cur_mask, net.find_blob_index_by_name("attention_mask"))
        for j in range(NUM_KV):
            net.set_input_blob_shape(key_shapes[j], net.find_blob_index_by_name(f"key_cache{j}"))
            net.set_input_blob_shape(value_shapes[j], net.find_blob_index_by_name(f"value_cache{j}"))
            net.copy_blob_data(f"key_cache{j}", f"key_cache_out{j}")
            net.copy_blob_data(f"value_cache{j}", f"value_cache_out{j}")
        net.update()

    # Warmup
    for i in range(WARMUP):
        run_decode_step()

    # Benchmark
    times = []
    for i in range(ITERATIONS):
        key_shapes = []
        value_shapes = []
        for j in range(NUM_KV):
            key_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"key_cache_out{j}")))
            value_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"value_cache_out{j}")))
        net.set_input_blob_data(input_ids, net.find_blob_index_by_name("input_ids"))
        net.set_input_blob_data(inputs_embeds, net.find_blob_index_by_name("inputs_embeds"))
        net.set_input_blob_data(position_ids, net.find_blob_index_by_name("position_ids"))
        cur_kv_len = key_shapes[0][2]
        cur_mask = np.ones((1, cur_kv_len + 1), dtype=np.int64)
        net.set_input_blob_data(cur_mask, net.find_blob_index_by_name("attention_mask"))
        for j in range(NUM_KV):
            net.set_input_blob_shape(key_shapes[j], net.find_blob_index_by_name(f"key_cache{j}"))
            net.set_input_blob_shape(value_shapes[j], net.find_blob_index_by_name(f"value_cache{j}"))
            net.copy_blob_data(f"key_cache{j}", f"key_cache_out{j}")
            net.copy_blob_data(f"value_cache{j}", f"value_cache_out{j}")

        start = time.perf_counter()
        net.update()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    print_results(f"ailia {args.model_type.upper()}", times, seq_len)


def benchmark_onnxruntime(args, weight_path, input_ids, inputs_embeds,
                          position_ids, attention_mask, past_key_values, seq_len):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(weight_path, sess_options)

    # Build feed dict
    def build_feed(kv_cache):
        feed = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        for j in range(NUM_KV):
            feed[f"key_cache{j}"] = kv_cache[j * 2]
            feed[f"value_cache{j}"] = kv_cache[j * 2 + 1]
        return feed

    output_names = [o.name for o in session.get_outputs()]

    # First run to initialize
    feed = build_feed(past_key_values)
    outputs = session.run(output_names, feed)

    def extract_kv_from_outputs(outputs):
        """Extract KV cache from outputs for next iteration."""
        kv = []
        for j in range(NUM_KV):
            kv.append(outputs[1 + j * 2])      # key_cache_out{j}
            kv.append(outputs[1 + j * 2 + 1])  # value_cache_out{j}
        return kv

    # Use output KV cache for subsequent runs
    current_kv = extract_kv_from_outputs(outputs)

    # Warmup
    for i in range(WARMUP):
        cur_kv_len = current_kv[0].shape[2]
        cur_mask = np.ones((1, cur_kv_len + 1), dtype=np.int64)
        feed = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": cur_mask,
        }
        for j in range(NUM_KV):
            feed[f"key_cache{j}"] = current_kv[j * 2]
            feed[f"value_cache{j}"] = current_kv[j * 2 + 1]
        outputs = session.run(output_names, feed)
        current_kv = extract_kv_from_outputs(outputs)

    # Benchmark
    times = []
    for i in range(ITERATIONS):
        # Prepare inputs (not timed)
        cur_kv_len = current_kv[0].shape[2]
        cur_mask = np.ones((1, cur_kv_len + 1), dtype=np.int64)
        feed = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": cur_mask,
        }
        for j in range(NUM_KV):
            feed[f"key_cache{j}"] = current_kv[j * 2]
            feed[f"value_cache{j}"] = current_kv[j * 2 + 1]

        start = time.perf_counter()
        outputs = session.run(output_names, feed)
        end = time.perf_counter()
        times.append((end - start) * 1000)

        current_kv = extract_kv_from_outputs(outputs)

    print_results(f"ONNX Runtime {args.model_type.upper()}", times, seq_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="fp32", choices=["fp32", "int4"])
    parser.add_argument("--env_id", type=int, default=-1)
    parser.add_argument("--seq_len", type=int, default=100,
                        help="Simulated past sequence length for KV cache")
    parser.add_argument("--onnxruntime", action="store_true",
                        help="Use ONNX Runtime instead of ailia")
    args = parser.parse_args()

    if args.model_type == "int4":
        weight_path = "Qwen2-VL-2B_int4.onnx"
        model_path = "Qwen2-VL-2B_int4.onnx.prototxt"
    else:
        weight_path = "Qwen2-VL-2B.onnx"
        model_path = "Qwen2-VL-2B.onnx.prototxt"

    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    seq_len = args.seq_len

    # Prepare dummy inputs for single-token decode step
    input_ids = np.array([[1]], dtype=np.int64)  # single token
    inputs_embeds = np.zeros((0, 1, HIDDEN_SIZE), dtype=np.float32)  # empty (cache mode)
    position_ids = np.array([[[seq_len]], [[seq_len]], [[seq_len]]], dtype=np.int64)  # (3,1,1)
    attention_mask = np.ones((1, seq_len + 1), dtype=np.int64)

    # Create dummy KV cache with seq_len entries
    past_key_values = [
        np.random.randn(1, NUM_HEADS, seq_len, HEAD_DIM).astype(np.float32)
        for _ in range(NUM_KV * 2)
    ]

    if args.onnxruntime:
        benchmark_onnxruntime(args, weight_path, input_ids, inputs_embeds,
                              position_ids, attention_mask, past_key_values, seq_len)
    else:
        benchmark_ailia(args, weight_path, model_path, input_ids, inputs_embeds,
                        position_ids, attention_mask, past_key_values, seq_len)


if __name__ == "__main__":
    main()
