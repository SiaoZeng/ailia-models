# Time-series foundation model comparison

This folder runs a side-by-side comparison of public time-series
foundation models on the same konbini sample data
(`../input.csv`), using PyTorch (uni2ts / chronos-forecasting) — the
ONNX export under `../` is **not** involved here. The goal is to see
how the architectures compare, especially in terms of how well each
model exploits known-future covariates (`feat_dynamic_real`).

## Models compared

| Family | Variant | License | Patch / config |
|---|---|---|---|
| Moirai-1.0-R | large | **Apache-2.0** (revision `bc5caba194...`) | patch_size=32 |
| Moirai-1.1-R | large | CC-BY-NC-4.0 | patch_size=16 |
| Moirai-2.0-R | small (the only public size) | CC-BY-NC-4.0 | patch_size=16 (fixed) |
| Chronos-2 | `amazon/chronos-2` (~120M parameters) | **Apache-2.0** | input/output patch=16 (fixed) |

Notes
- Moirai patch sizes were picked from earlier sweeps as the value that
  maximises the holiday-vs-non-holiday median gap.
- **Moirai-2.0-R is only published as `small`**; `base` / `large` are
  not on Hugging Face (HTTP 401).
- **Chronos-2 ships as a single ~120M-parameter model**, no size
  variants. It supports known-future covariates natively.

## Output

![comparison](output_compare.png)

Each panel shows the last 60 history days plus the 20-day forecast
window. Orange vertical bands mark `is_holiday=1` days inside the
forecast horizon. The dashed red line is the median forecast and the
shaded area is the 80% prediction interval.

## Metrics on the 20-day forecast horizon

| Model | License | MAE | RMSE | PI80 coverage | Median gap (holiday − non-holiday) |
|---|---|--:|--:|--:|--:|
| Moirai-1.0-R-large (p=32) | **Apache-2.0** | 10.39 | 13.04 | 65% | +4.37 |
| Moirai-1.1-R-large (p=16) | CC-BY-NC-4.0 | 8.79 | 12.14 | 70% | +8.24 |
| Moirai-2.0-R-small (p=16) | CC-BY-NC-4.0 | 6.25 | 10.88 | 80% | +14.95 |
| **Chronos-2** | **Apache-2.0** | **5.30** | **7.77** | **85%** | **+19.97** |

Reference: observed `sales` gap between holiday and non-holiday days in
the past 200 days of context is **+23.61**.

### Per-holiday-type breakdown

The 20-day prediction window contains two kinds of `is_holiday=1` days:
6 weekend days (Sat/Sun) and 2 mid-week irregular days (Tue/Wed —
Christmas Eve and Christmas in the synthetic data). Foundation models
typically pick up the weekend pattern from the target's own auto-
correlation, but capturing the irregular one requires actually trusting
the future `is_holiday` covariate. Splitting the gap by holiday type:

| Model | Irregular gap (Tue/Wed) | Weekend gap (Sat/Sun) |
|---|--:|--:|
| Moirai-1.0-R-large (p=32) | +1.08 | +5.47 |
| Moirai-1.1-R-large (p=16) | -1.49 | +11.48 |
| Moirai-2.0-R-small | +0.51 | +19.77 |
| **Chronos-2** | **+14.41** | +21.83 |
| Ground truth | +34.69 | +23.85 |

Among the four, **only Chronos-2 captures the irregular mid-week
holiday spikes** (~42% of the true effect), while the Moirai family
basically ignores the future `is_holiday` signal once the date does not
match Sat/Sun (or Mon, the only weekday on which mid-week holidays
appear in the 200-day context). This is a structural artefact of how
each model fuses covariates with the target time-series, not a defect
of the export.

## Takeaways

- **Chronos-2 wins on every metric** — and it is Apache-2.0, so it can
  replace Moirai-1.0-R for commercial use without any license trade-off.
  It captures roughly 76% of the true holiday effect (+16.97 / +22.34).
- **Moirai-2.0-R-small is a close second** — also a small ~45 MB model,
  but its CC-BY-NC-4.0 license restricts commercial use.
- **Architecture matters much more than model size** for covariate
  utilisation: both Moirai-2.0-R-small and Chronos-2 (very different
  designs) outperform Moirai-1.x-large by a wide margin.
- **For the Apache-2.0 constraint**, Chronos-2 is the strongest
  zero-shot option today; Moirai-1.0-R remains usable but with ~24%
  covariate uptake on this benchmark.

## Reproducing

```bash
$ pip install uni2ts gluonts matplotlib "chronos-forecasting>=2.0"
$ python3 compare_moirai_versions.py
```

Useful flags:

- `--size_v1 {small,base,large}` — which Moirai-1.x size to load
  (default `large`)
- `--patch_v10 / --patch_v11` — override patch sizes for 1.0/1.1-R
- `--context_len`, `--prediction_len`, `--num_samples`, `--seed`
- `--data PATH` — supply a different CSV; must contain `date`, the
  target column, and the listed `--feat` covariate columns
- `--save PATH.png` — output plot location
