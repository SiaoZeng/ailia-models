"""Export Moirai (uni2ts) to ONNX.

Wraps `uni2ts.model.moirai.MoiraiModule` so that its forward returns the raw
mixture-distribution parameter tensors (along with the scaler's loc / scale).
The actual sampling and post-processing is performed in the inference script.

Supported sizes: small, base, large (Moirai 1.1-R).
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from uni2ts.model.moirai import MoiraiModule


def _manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, scale=None, is_causal=False
):
    """ONNX-export-friendly replacement for F.scaled_dot_product_attention.

    PyTorch's ONNX exporter (torchscript path, opset 17) has a bug when the
    `scale` argument is a Python float (TypeError on z_(...)). We avoid the
    builtin op entirely.
    """
    scale_factor = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros_like(attn_weight)
            attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask
        attn_weight = attn_weight + attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return torch.matmul(attn_weight, value)


REPO_TEMPLATE = "Salesforce/moirai-{version}-R-{size}"


class MoiraiExportWrapper(nn.Module):
    """Wraps MoiraiModule to return distribution parameters as a flat tuple."""

    def __init__(self, module: MoiraiModule):
        super().__init__()
        self.module = module

    @staticmethod
    def _packed_attention_mask(sample_id: torch.Tensor) -> torch.Tensor:
        # Replacement for uni2ts.common.torch_util.packed_attention_mask which
        # uses Tensor.mT (matrix transpose); aten::mT is not supported by the
        # ONNX exporter.
        s = sample_id.unsqueeze(-1)
        return s.eq(s.transpose(-2, -1))

    def forward(
        self,
        target,
        observed_mask,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
        patch_size,
    ):
        m = self.module

        loc, scale = m.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        # The einops `reduce(..., "... seq1 seq2 -> ... seq1 1", "sum")` inside
        # PackedStdScaler emits an ONNX subgraph whose output dim_param matches
        # `seq_len` instead of being a literal 1, causing broadcast errors at
        # runtime. Slice to force the trailing axis to be statically 1.
        loc = loc[..., :1]
        scale = scale[..., :1]
        scaled_target = (target - loc) / scale
        reprs = m.in_proj(scaled_target, patch_size)
        from uni2ts.common.torch_util import mask_fill

        masked_reprs = mask_fill(reprs, prediction_mask, m.mask_encoding.weight)
        reprs = m.encoder(
            masked_reprs,
            self._packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=variate_id,
        )
        distr_param = m.param_proj(reprs, patch_size)

        # Flatten the mixture parameter pytree into a fixed list of tensors.
        # Order matches uni2ts MixtureOutput components for Moirai-1.x-R:
        #   [StudentT, NormalFixedScale, NegativeBinomial, LogNormal]
        weights_logits = distr_param["weights_logits"]
        comps = distr_param["components"]
        student_t = comps[0]
        normal_fs = comps[1]
        neg_bin = comps[2]
        log_normal = comps[3]

        return (
            weights_logits,
            student_t["df"],
            student_t["loc"],
            student_t["scale"],
            normal_fs["loc"],
            neg_bin["total_count"],
            neg_bin["logits"],
            log_normal["loc"],
            log_normal["scale"],
            loc,
            scale,
        )


def export(size: str, version: str, opset: int, output_dir: str, seq_len: int):
    # Patch SDPA before loading the module to avoid the ONNX exporter bug.
    F.scaled_dot_product_attention = _manual_scaled_dot_product_attention

    repo = REPO_TEMPLATE.format(version=version, size=size)
    print(f"Loading {repo} ...", flush=True)
    module = MoiraiModule.from_pretrained(repo)
    module.eval()

    wrapper = MoiraiExportWrapper(module).eval()

    max_patch = max(module.patch_sizes)
    patch_value = max_patch  # any valid patch size for tracing

    B, S, P = 1, seq_len, max_patch
    target = torch.zeros(B, S, P, dtype=torch.float32)
    observed_mask = torch.ones(B, S, P, dtype=torch.bool)
    sample_id = torch.zeros(B, S, dtype=torch.long)
    time_id = torch.arange(S, dtype=torch.long).unsqueeze(0).repeat(B, 1)
    variate_id = torch.zeros(B, S, dtype=torch.long)
    prediction_mask = torch.zeros(B, S, dtype=torch.bool)
    prediction_mask[:, -1:] = True
    patch_size = torch.full((B, S), patch_value, dtype=torch.long)

    with torch.no_grad():
        wrapper(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            patch_size,
        )

    onnx_name = f"moirai-{version}-R-{size}.onnx"
    onnx_path = os.path.join(output_dir, onnx_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting to {onnx_path} (opset={opset}) ...", flush=True)

    input_names = [
        "target",
        "observed_mask",
        "sample_id",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
    ]
    output_names = [
        "weights_logits",
        "student_t_df",
        "student_t_loc",
        "student_t_scale",
        "normal_loc",
        "nb_total_count",
        "nb_logits",
        "lognormal_loc",
        "lognormal_scale",
        "loc",
        "scale",
    ]

    dynamic_axes = {n: {0: "batch", 1: "seq_len"} for n in input_names}
    for n in output_names:
        dynamic_axes[n] = {0: "batch", 1: "seq_len"}

    torch.onnx.export(
        wrapper,
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            patch_size,
        ),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    print(f"Saved: {onnx_path}", flush=True)
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export Moirai to ONNX")
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "base", "large"],
        help="Model size",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.1",
        choices=["1.0", "1.1"],
        help="Model version (Moirai-?-R)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Where to write the .onnx file",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=64,
        help="Trace sequence length (output is dynamic)",
    )
    args = parser.parse_args()

    export(
        size=args.size,
        version=args.version,
        opset=args.opset,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
