# Moirai version comparison

This folder runs a side-by-side comparison of three publicly released
Moirai families on the same konbini sample data
(`../input.csv`), using PyTorch (uni2ts) — the ONNX export under `../`
is **not** involved here. The goal is to see how the architecture has
evolved between versions, especially in terms of how well each model
exploits known-future covariates (`feat_dynamic_real`).

## Models compared

| Family | Size | License of weights | Patch size used |
|---|---|---|---|
| Moirai-1.0-R | large | **Apache-2.0** (revision `bc5caba194...`) | 32 |
| Moirai-1.1-R | large | CC-BY-NC-4.0 | 16 |
| Moirai-2.0-R | small (the only public size) | CC-BY-NC-4.0 | 16 (fixed) |

Patch sizes were picked from earlier sweeps as the value that maximises
the holiday-vs-non-holiday median gap for each family. Moirai-2.0-R has
a single fixed patch size of 16 baked into the architecture.

Apache-2.0 1.0-R weights are the same ones used by `../moirai.py` (the
shipped ONNX file). The 1.1-R and 2.0-R weights are pulled here for
non-commercial side-by-side analysis only — do not redistribute them
under Apache-2.0.

## Output

![comparison](output_compare.png)

Each panel shows the last 60 history days plus the 20-day forecast
window. Orange vertical bands mark `is_holiday=1` days inside the
forecast horizon. The dashed red line is the median forecast and the
shaded area is the 80% prediction interval.

## Metrics on the 20-day forecast horizon

| Model | MAE | RMSE | PI80 coverage | Median gap (holiday − non-holiday) |
|---|--:|--:|--:|--:|
| Moirai-1.0-R-large (Apache-2.0, p=32) | 10.03 | 12.58 | 65% | +5.26 |
| Moirai-1.1-R-large (CC-BY-NC-4.0, p=16) | 8.28 | 12.06 | 70% | +10.52 |
| **Moirai-2.0-R-small (CC-BY-NC-4.0, p=16)** | **5.70** | **10.51** | **80%** | **+16.26** |

Reference: observed `sales` gap between holiday and non-holiday days in
the past 200 days of context is **+22.34**.

## Takeaways

- **Moirai-2.0-R-small beats both 1.x families** on every metric, despite
  being the smallest model (~45 MB safetensors). It captures roughly 73%
  of the true holiday effect (+16.26 / +22.34) versus 47% for 1.1-R and
  24% for 1.0-R.
- **Architecture matters more than scale** in this task. Moirai-2.0-R
  switched from sample-based mixture forecasting to direct quantile
  regression with a deeper FFN (`d_ff=1024`) and a tighter causal
  attention scheme; the small variant outperforms the large of older
  families.
- **For the Apache-2.0 constraint**, Moirai-1.0-R is the only choice and
  remains usable. Expect ~24% covariate uptake on this benchmark, so
  augment with a hand-engineered holiday baseline if higher accuracy is
  needed.
- **For CC-BY-NC-4.0 use cases** (research, internal evaluation, etc.)
  Moirai-2.0-R-small is clearly the best public Moirai for covariate
  forecasting today.

## Reproducing

```bash
$ pip install uni2ts gluonts matplotlib
$ python3 compare_moirai_versions.py
```

Useful flags:

- `--size_v1 {small,base,large}` — which Moirai-1.x size to load
  (default `large`, matches the patch-size table in `../README.md`)
- `--patch_v10 / --patch_v11` — override patch sizes for 1.0/1.1-R
- `--context_len`, `--prediction_len`, `--num_samples`, `--seed`
- `--data PATH` — supply a different CSV; must contain `date`, the
  target column, and the listed `--feat` covariate columns
- `--save PATH.png` — output plot location
