# Euro area HICP inflation: pseudo out-of-sample forecast comparison

A small personal project comparing simple time-series and machine-learning forecasts of euro area headline inflation against a random-walk benchmark, using data from the **ECB Data Portal** (SDMX REST API).

**Short version of the result:** no model beats the random walk in a statistically significant way. AR and Ridge have slightly lower RMSE over the full sample, but this advantage comes mostly from the 2021-2023 inflation shock and is not significant. A random forest trained on the level of inflation is clearly worse than the random walk; predicting the *change* instead removes most of that gap.

## Data

- Source: ECB Data Portal, dataset ICP, monthly, euro area (`U2`, changing composition), annual rate of change.
- Headline HICP: `ICP.M.U2.N.000000.4.ANR`
- Core HICP (excl. energy, food, alcohol, tobacco): `ICP.M.U2.N.XEF000.4.ANR`
- Sample retrieved: **1997-01 to 2025-12, n = 348**. Run date: `<fill in>`.
- Latest data vintage only (no real-time vintages), so this is a *pseudo* out-of-sample exercise.

## Design

- Target: headline inflation 1, 3, 6 and 12 months ahead, direct forecasts.
- Models:
  - **RW**: last observed value (benchmark)
  - **AR**: OLS on 6 lags of headline inflation
  - **Ridge**: standardised inputs, alpha = 10, on 6 lags of headline and core
  - **RF**: random forest (300 trees, min leaf size 3) on 6 lags of headline and core
- Evaluation: expanding window, first forecast origin 2015-01, models refit every 12 months, only targets already observed at the forecast origin are used for training. Number of forecast origins: 131 / 129 / 126 / 120 for h = 1 / 3 / 6 / 12.
- No hyperparameter tuning; all settings were fixed in advance.
- Two specifications of the model target:
  - **Spec 1 ("level")**: models predict `y_{t+h}` directly.
  - **Spec 2 ("change")**: models predict `y_{t+h} - y_t`, and the forecast is `y_t` plus the predicted change.
- Metrics: RMSE, RMSE relative to RW, and Diebold-Mariano tests vs RW (squared-error loss, HAC variance with lag h-1, Harvey-Leybourne-Newbold small-sample correction, t-distribution).
- Subsample check: RMSE recomputed excluding forecasts whose target month falls in 2021-2023.

**Order of work (transparency):** Spec 1, the DM tests and the 2021-2023 subsample check were not all planned at the start. Spec 2, the DM tests and the subsample were added *after* seeing the Spec 1 full-sample table. Both specifications are reported in full below; nothing was dropped.

## Results

RMSE relative to RW (values below 1 mean lower RMSE than the random walk). RW RMSE is in percentage points.

### Full sample (2015-01 onward)

| h | RW RMSE | AR | Ridge (level) | RF (level) | Ridge (change) | RF (change) |
|---|---------|------|------|-------|------|-------|
| 1  | 0.425 | 0.941 | 1.057 | 3.946 | 0.953 | 1.100 |
| 3  | 0.937 | 0.896 | 0.998 | 2.450 | 0.941 | 1.187 |
| 6  | 1.627 | 0.884 | 0.951 | 1.885 | 0.919 | 1.225 |
| 12 | 2.856 | 0.922 | 0.954 | 1.156 | 0.939 | 0.995 |

AR is identical under both specifications, as expected: the current inflation value is one of its regressors, so subtracting it from the target does not change OLS predictions.

### Excluding targets in 2021-2023

| h | RW RMSE | AR | Ridge (level) | RF (level) | Ridge (change) | RF (change) |
|---|---------|------|------|-------|------|-------|
| 1  | 0.259 | 1.017 | 1.064 | 1.117 | 0.984 | 1.072 |
| 3  | 0.474 | 1.125 | 1.117 | 1.084 | 1.102 | 1.146 |
| 6  | 0.751 | 1.098 | 1.081 | 1.732 | 1.068 | 1.324 |
| 12 | 1.583 | 0.887 | 0.931 | 1.028 | 0.928 | 0.912 |

### Diebold-Mariano tests vs RW (full sample)

Statistic (p-value). Positive statistic means the model has lower squared error than RW.

| h | model | Spec 1 (level) | Spec 2 (change) |
|---|-------|----------------|-----------------|
| 1  | AR    | 1.32 (0.188)  | 1.32 (0.188)  |
| 1  | Ridge | -1.25 (0.212) | 1.08 (0.281)  |
| 1  | RF    | -4.17 (<0.001) | -1.89 (0.061) |
| 3  | AR    | 1.02 (0.311)  | 1.02 (0.311)  |
| 3  | Ridge | 0.02 (0.984)  | 0.54 (0.587)  |
| 3  | RF    | -2.07 (0.040) | -1.42 (0.159) |
| 6  | AR    | 0.63 (0.529)  | 0.63 (0.529)  |
| 6  | Ridge | 0.25 (0.801)  | 0.43 (0.665)  |
| 6  | RF    | -1.49 (0.138) | -1.27 (0.208) |
| 12 | AR    | 0.55 (0.586)  | 0.55 (0.586)  |
| 12 | Ridge | 0.29 (0.776)  | 0.39 (0.701)  |
| 12 | RF    | -0.82 (0.412) | 0.04 (0.969)  |

### What the results show

1. **No model significantly beats the random walk.** AR and Ridge have RMSE ratios of about 0.88-0.96 in most full-sample cells, but every DM p-value for these models is above 0.18.
2. **The full-sample advantage of AR/Ridge is driven by the 2021-2023 shock.** Excluding it, AR and Ridge are roughly on par with or worse than RW at h = 1, 3 and 6 (ratios 0.98-1.13). Only at h = 12 do they keep a lower RMSE (AR 0.887, Ridge 0.93). DM tests were not run on the subsample, so this 12-month advantage is not shown to be significant.
3. **RF on the level of inflation is worse than RW at every horizon** (ratios 1.16-3.95), significantly so at h = 1 (p < 0.001) and h = 3 (p = 0.040). Predicting the change instead improves RF considerably (e.g. h = 1: 3.946 to 1.100), but it stays at or above RW at every horizon in the full sample (0.995 at h = 12 is the closest).
4. **Possible reason for RF's behaviour (not tested):** tree models cannot predict outside the range of their training targets, which is a disadvantage when inflation moves to levels not seen before. A diagnostic of maximum h = 1 forecasts (RF level: 9.31, RF change: 10.14, maximum actual: 10.6) is consistent with this, but I did not isolate the mechanism.

## Limitations

- Pseudo out-of-sample: latest data vintage, not real-time data.
- Euro area composition changes over the sample (`U2`).
- Short evaluation window dominated by a single large shock (2021-2023); forecast errors at longer horizons overlap, so tests have limited power.
- Only headline and core inflation are used as predictors; models are deliberately simple and untuned.
- The subsample comparison has no significance test.
- Results are from a single run on the data vintage available at the run date.

## Reproduce

```
pip install -r requirements.txt
python hicp_forecast.py --spec level     # Spec 1
python hicp_forecast.py --spec change    # Spec 2
```

`python hicp_forecast.py --synthetic` runs a smoke test on made-up data only; its output is not a result.

Outputs (suffix `_level` or `_change`): `forecasts_*.csv`, `rmse_table_*.csv`, `rmse_table_ex2021_2023_*.csv`, `dm_tests_*.csv`, `h12_forecasts_*.png`.
