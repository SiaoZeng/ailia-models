import os
import sys
import warnings
from logging import getLogger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

DATA_PATH = "input.csv"
SAVE_IMAGE_PATH = "output.png"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/moirai/"

SIZE_TO_FILES = {
    "small": ("moirai-1.1-R-small.onnx", "moirai-1.1-R-small.onnx.prototxt"),
    "base": ("moirai-1.1-R-base.onnx", "moirai-1.1-R-base.onnx.prototxt"),
    "large": ("moirai-1.1-R-large.onnx", "moirai-1.1-R-large.onnx.prototxt"),
}

# Static configuration of Moirai-1.1-R (all sizes).
PATCH_SIZES = (8, 16, 32, 64, 128)
MAX_PATCH = max(PATCH_SIZES)
NUM_MIXTURE_COMPONENTS = 4  # StudentT, NormalFixedScale, NegativeBinomial, LogNormal

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("Moirai", DATA_PATH, SAVE_IMAGE_PATH)
parser.add_argument("-i", "--input", type=str, default=DATA_PATH)
parser.add_argument(
    "--size",
    type=str,
    default="small",
    choices=list(SIZE_TO_FILES.keys()),
    help="Moirai-1.1-R model size",
)
parser.add_argument(
    "--target",
    type=str,
    default=None,
    help="Target column name (defaults to first non-date column).",
)
parser.add_argument(
    "--feat",
    type=str,
    default=None,
    help=(
        "Comma-separated covariate column names whose future values are known "
        "(feat_dynamic_real)."
    ),
)
parser.add_argument(
    "--context_len",
    type=int,
    default=200,
    help="Context length (history)",
)
parser.add_argument(
    "--prediction_len",
    type=int,
    default=20,
    help="Prediction horizon length",
)
parser.add_argument(
    "--patch_size",
    type=str,
    default="auto",
    help="Patch size: 'auto' or one of {8, 16, 32, 64, 128}",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=100,
    help="Number of samples drawn from the predictive distribution",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed used for sampling",
)
parser.add_argument(
    "--onnx",
    action="store_true",
    help="Use ONNX Runtime instead of ailia SDK",
)
args = update_parser(parser)


# ======================
# ONNX-backed MoiraiModule
# ======================


def _make_onnx_module(net, base_module, use_onnx_runtime: bool):
    """Build a torch.nn.Module subclass of MoiraiModule whose forward
    delegates to the ONNX session and returns the same Distribution object as
    the PyTorch original."""
    import torch
    from uni2ts.model.moirai import MoiraiModule

    class OnnxMoiraiModule(MoiraiModule):
        def __init__(self):
            # Do NOT call super().__init__: the parent allocates the full
            # transformer. We re-use the configuration of the loaded module
            # but skip parameter creation.
            torch.nn.Module.__init__(self)
            self.distr_output = base_module.distr_output
            self.patch_sizes = base_module.patch_sizes
            self.d_model = base_module.d_model
            self.num_layers = base_module.num_layers
            self.max_seq_len = base_module.max_seq_len
            self.scaling = base_module.scaling

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
            inputs = {
                "target": target.detach().cpu().numpy().astype(np.float32),
                "observed_mask": observed_mask.detach().cpu().numpy().astype(bool),
                "sample_id": sample_id.detach().cpu().numpy().astype(np.int64),
                "time_id": time_id.detach().cpu().numpy().astype(np.int64),
                "variate_id": variate_id.detach().cpu().numpy().astype(np.int64),
                "prediction_mask": prediction_mask.detach().cpu().numpy().astype(bool),
                "patch_size": patch_size.detach().cpu().numpy().astype(np.int64),
            }

            if use_onnx_runtime:
                outputs = net.run(None, inputs)
            else:
                # ailia.Net expects positional inputs.
                ailia_inputs = [
                    inputs["target"],
                    inputs["observed_mask"],
                    inputs["sample_id"],
                    inputs["time_id"],
                    inputs["variate_id"],
                    inputs["prediction_mask"],
                    inputs["patch_size"],
                ]
                outputs = net.run(ailia_inputs)

            (
                weights_logits,
                st_df,
                st_loc,
                st_scale,
                normal_loc,
                nb_total_count,
                nb_logits,
                ln_loc,
                ln_scale,
                loc,
                scale,
            ) = outputs

            def _t(x):
                return torch.from_numpy(np.ascontiguousarray(x))

            distr_param = {
                "weights_logits": _t(weights_logits),
                "components": [
                    {
                        "df": _t(st_df),
                        "loc": _t(st_loc),
                        "scale": _t(st_scale),
                    },
                    {"loc": _t(normal_loc)},
                    {
                        "total_count": _t(nb_total_count),
                        "logits": _t(nb_logits),
                    },
                    {
                        "loc": _t(ln_loc),
                        "scale": _t(ln_scale),
                    },
                ],
            }
            distr = self.distr_output.distribution(
                distr_param, loc=_t(loc), scale=_t(scale)
            )
            return distr

    return OnnxMoiraiModule()


# ======================
# Plotting
# ======================


def draw_result(
    history, trues, preds_quantiles, save_path, target_name, covariates=None
):
    has_cov = covariates is not None and len(covariates) > 0
    holiday_band = None
    if has_cov:
        for name, values in covariates.items():
            # Heuristic: a binary covariate (0/1) is treated as a holiday-style
            # indicator and highlighted as orange bands on the forecast plot.
            uniq = np.unique(values)
            if set(uniq.tolist()).issubset({0, 1}):
                holiday_band = (name, np.asarray(values))
                break

    n_rows = 2 + (len(covariates) if has_cov else 0)
    fig = plt.figure(figsize=(12, 2.5 * n_rows))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.35)

    n_hist = len(history)
    n_pred = preds_quantiles["median"].shape[-1]
    x_hist = np.arange(n_hist)
    x_pred = np.arange(n_hist, n_hist + n_pred)
    median = preds_quantiles["median"]
    q10 = preds_quantiles["q10"]
    q90 = preds_quantiles["q90"]

    # (1) Full overview ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x_hist, history, label=f"History ({n_hist} steps)", color="darkblue")
    if 0 < len(trues):
        ax.plot(
            x_pred[: len(trues)],
            trues,
            label=f"Ground Truth ({len(trues)} steps)",
            color="darkblue",
            linestyle="--",
            alpha=0.5,
        )
    ax.plot(x_pred, median, label="Forecast (median)", color="red", linestyle="--")
    ax.fill_between(x_pred, q10, q90, color="red", alpha=0.2, label="80% interval")
    ax.set_ylabel(target_name)
    ax.set_title("Overview")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # (2) Zoom on last 3x prediction_len of history + the prediction window.
    ax = fig.add_subplot(gs[1, 0])
    zoom_hist = min(3 * n_pred, n_hist)
    xh = np.arange(-zoom_hist, 0)
    xp = np.arange(0, n_pred)
    ax.plot(xh, history[-zoom_hist:], color="darkblue", label="History")
    if 0 < len(trues):
        ax.plot(
            xp[: len(trues)],
            trues,
            "--",
            color="darkblue",
            alpha=0.5,
            label="Ground Truth",
        )
    ax.plot(xp, median, "--", color="red", label="Forecast (median)")
    ax.fill_between(xp, q10, q90, color="red", alpha=0.2, label="80% interval")
    if holiday_band is not None:
        name, vals = holiday_band
        # Highlight the binary indicator on the prediction window.
        future_vals = vals[-n_pred:]
        for i, h in enumerate(future_vals):
            if h:
                ax.axvspan(i - 0.4, i + 0.4, color="orange", alpha=0.18)
        ax.axvline(0, color="gray", linestyle=":")
        ax.set_title(f"Zoomed forecast (orange bands = {name}=1)")
    else:
        ax.axvline(0, color="gray", linestyle=":")
        ax.set_title("Zoomed forecast")
    ax.set_ylabel(target_name)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # (3+) Covariate panels (full series).
    if has_cov:
        for ax_idx, (name, values) in enumerate(covariates.items()):
            ax_i = fig.add_subplot(gs[2 + ax_idx, 0])
            ax_i.plot(np.arange(len(values)), values, color="green")
            ax_i.set_ylabel(name)
            ax_i.grid(alpha=0.3)

    fig.axes[-1].set_xlabel("Time")
    plt.savefig(save_path)


# ======================
# Forecasting
# ======================


def time_series_forecasting(net):
    import torch
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    data_path = args.input if isinstance(args.input, str) else args.input[0]
    df = pd.read_csv(data_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    target = args.target
    if target is None:
        target = df.columns[0]
    logger.info(f"target column: {target}")

    feat_cols = []
    if args.feat:
        feat_cols = [c.strip() for c in args.feat.split(",") if c.strip()]
        logger.info(f"feat_dynamic_real: {feat_cols}")

    df_input = df[[target] + feat_cols].copy()

    context_len = args.context_len
    prediction_len = args.prediction_len

    if len(df_input) < context_len + prediction_len:
        logger.warning(
            "Input length %d is shorter than context_len + prediction_len=%d",
            len(df_input),
            context_len + prediction_len,
        )

    history_df = df_input.iloc[-(context_len + prediction_len) : -prediction_len]
    truth_df = df_input.iloc[-prediction_len:]

    # Build a GluonTS PandasDataset that includes both the past context and
    # the future window. The future window is needed so that `feat_dynamic_real`
    # values for the forecast horizon are available to the model.
    ds_df = pd.concat([history_df, truth_df])
    ds = PandasDataset(
        ds_df,
        target=target,
        feat_dynamic_real=feat_cols if feat_cols else None,
    )
    # Split to a `test_template` that exposes future values of dynamic
    # covariates inside `test_data.input`.
    _, test_template = split(ds, offset=-prediction_len)
    test_data = test_template.generate_instances(
        prediction_length=prediction_len, windows=1, distance=prediction_len
    )

    # Resolve patch_size argument.
    if args.patch_size == "auto":
        patch_size_arg = "auto"
    else:
        try:
            patch_size_arg = int(args.patch_size)
        except ValueError:
            raise ValueError(f"Unsupported patch_size: {args.patch_size}")

    # Load the original MoiraiModule once so we can re-use its DistributionOutput
    # configuration (patch_sizes, mixture components, etc.). The actual forward
    # pass is replaced with the ONNX-backed OnnxMoiraiModule below.
    repo = f"Salesforce/moirai-1.1-R-{args.size}"
    logger.info(f"Loading Moirai config from HuggingFace: {repo}")
    base_module = MoiraiModule.from_pretrained(repo)
    base_module.eval()

    onnx_module = _make_onnx_module(net, base_module, use_onnx_runtime=args.onnx)

    forecast_model = MoiraiForecast(
        module=onnx_module,
        prediction_length=prediction_len,
        context_length=context_len,
        patch_size=patch_size_arg,
        num_samples=args.num_samples,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
    forecast_model.eval()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    predictor = forecast_model.create_predictor(batch_size=1)
    forecasts = list(predictor.predict(test_data.input))

    if not forecasts:
        raise RuntimeError("No forecast was generated.")
    fc = forecasts[0]
    samples = np.asarray(fc.samples)  # [num_samples, prediction_length]

    median = np.quantile(samples, 0.5, axis=0)
    q10 = np.quantile(samples, 0.1, axis=0)
    q90 = np.quantile(samples, 0.9, axis=0)

    history_vals = history_df[target].values
    truth_vals = truth_df[target].values

    covariates = None
    if feat_cols:
        covariates = {
            c: pd.concat([history_df[c], truth_df[c]]).values for c in feat_cols
        }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        savepath = get_savepath(args.savepath, data_path, ext=".png")
        logger.info(f"saved at : {savepath}")
        draw_result(
            history_vals,
            truth_vals,
            {"median": median, "q10": q10, "q90": q90},
            savepath,
            target_name=target,
            covariates=covariates,
        )

    if getattr(args, "write_json", False):
        import json

        out = {
            "median": median.tolist(),
            "q10": q10.tolist(),
            "q90": q90.tolist(),
        }
        json_path = os.path.splitext(savepath)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"saved json at : {json_path}")

    logger.info("Script finished successfully.")


def main():
    weight_path, model_path = SIZE_TO_FILES[args.size]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=args.env_id)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(weight_path)

    time_series_forecasting(net)


if __name__ == "__main__":
    main()
