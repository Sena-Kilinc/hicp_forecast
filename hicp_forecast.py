"""
Euro area HICP inflation: pseudo out-of-sample forecast comparison.

Data: ECB Data Portal (SDMX REST API), monthly, euro area (changing composition)
  - headline HICP, annual rate of change : ICP.M.U2.N.000000.4.ANR
  - core HICP (excl. energy, food, alcohol, tobacco), annual rate of change : ICP.M.U2.N.XEF000.4.ANR

Models (direct multi-step forecasts, expanding window, refit every 12 months):
  - RW    : random walk (last observed value)            <- benchmark
  - AR    : OLS on lags of headline inflation
  - Ridge : regularised regression on lags of headline + core
  - RF    : random forest on lags of headline + core

Specifications (target of the ML/AR models):
  - level  (Spec 1): y_{t+h}            -> the original specification
  - change (Spec 2): y_{t+h} - y_t      -> forecast = y_t + predicted change
  Spec 2 was added AFTER seeing Spec 1 results (tree models cannot extrapolate
  beyond the training range). Report both, say so.

Extra outputs:
  - RMSE on the full sample and excluding targets in 2021-2023
  - Diebold-Mariano tests vs RW (squared-error loss, HAC lag h-1, HLN small-sample correction)

Run:   python hicp_forecast.py --spec level
       python hicp_forecast.py --spec change
       python hicp_forecast.py --synthetic   (offline smoke test; NOT for reporting)

NOTE: uses the latest data vintage, not real-time vintages, so this is a
pseudo out-of-sample exercise.
"""
import argparse, io
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = "https://data-api.ecb.europa.eu/service/data"
KEYS = {"headline": "ICP/M.U2.N.000000.4.ANR", "core": "ICP/M.U2.N.XEF000.4.ANR"}
HORIZONS = [1, 3, 6, 12]
N_LAGS = 6
EVAL_START = "2015-01"      # first forecast origin
REFIT_EVERY = 12
SHOCK = (2021, 2023)        # target years excluded in the subsample check


def fetch(key: str) -> pd.Series:
    import requests
    r = requests.get(f"{BASE}/{key}", params={"format": "csvdata", "startPeriod": "1997-01"}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = df.set_index("TIME_PERIOD")["OBS_VALUE"].astype(float)
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s.sort_index()


def synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    h = np.zeros(n); c = np.zeros(n)
    for t in range(1, n):
        c[t] = 0.02 + 0.97 * c[t-1] + rng.normal(0, 0.08)
        h[t] = 0.05 + 0.85 * h[t-1] + 0.15 * c[t-1] + rng.normal(0, 0.25)
    idx = pd.period_range("2000-01", periods=n, freq="M")
    return pd.DataFrame({"headline": h, "core": c}, index=idx)


def make_features(df: pd.DataFrame, cols) -> pd.DataFrame:
    X = {}
    for col in cols:
        for l in range(N_LAGS):
            X[f"{col}_l{l}"] = df[col].shift(l)
    return pd.DataFrame(X)


MODELS = {
    "AR":    (["headline"],         lambda: LinearRegression()),
    "Ridge": (["headline", "core"], lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
    "RF":    (["headline", "core"], lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1)),
}


def run(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    df = df.dropna()
    start = df.index.get_loc(pd.Period(EVAL_START, "M"))
    feats = {m: make_features(df, cols) for m, (cols, _) in MODELS.items()}
    rows = []
    for h in HORIZONS:
        y_level = df["headline"].shift(-h)                 # value h months ahead of origin t
        y = y_level - df["headline"] if spec == "change" else y_level
        fitted = {}
        for i, T in enumerate(range(start, len(df) - h)):
            actual = y_level.iloc[T]
            last = df["headline"].iloc[T]
            base = last if spec == "change" else 0.0
            preds = {"RW": last}
            for m in MODELS:
                if i % REFIT_EVERY == 0:
                    X = feats[m]
                    ok = X.notna().all(axis=1) & y.notna()
                    ok &= (np.arange(len(df)) + h) <= T    # only targets already observed at origin T
                    fitted[m] = MODELS[m][1]().fit(X[ok].values, y[ok].values)
                preds[m] = base + float(fitted[m].predict(feats[m].iloc[[T]].values)[0])
            for m, p in preds.items():
                rows.append({"h": h, "origin": df.index[T], "target": df.index[T + h],
                             "model": m, "forecast": p, "actual": actual})
    return pd.DataFrame(rows)


def rmse_table(res: pd.DataFrame) -> pd.DataFrame:
    res = res.assign(err2=(res.forecast - res.actual) ** 2)
    rmse = res.groupby(["h", "model"]).err2.mean().pow(0.5).unstack("model")
    rel = rmse.div(rmse["RW"], axis=0)
    return pd.concat({"RMSE": rmse, "RMSE / RW": rel}, axis=1).round(3)


def dm_pvalue(e_rw2: np.ndarray, e_m2: np.ndarray, h: int) -> tuple:
    """Two-sided DM test, H0: equal MSE. Returns (stat, p). stat > 0 -> model better than RW."""
    d = e_rw2 - e_m2
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    var = (dc @ dc) / n
    for k in range(1, h):                                  # rectangular kernel, lag h-1
        var += 2 * (dc[k:] @ dc[:-k]) / n
    var /= n
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # Harvey-Leybourne-Newbold
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))


def dm_table(res: pd.DataFrame) -> pd.DataFrame:
    res = res.assign(err2=(res.forecast - res.actual) ** 2)
    out = []
    for h in HORIZONS:
        e = res[res.h == h].pivot(index="origin", columns="model", values="err2").sort_index()
        for m in [c for c in e.columns if c != "RW"]:
            stat, p = dm_pvalue(e["RW"].values, e[m].values, h)
            out.append({"h": h, "model": m, "DM_stat(+ = better than RW)": round(stat, 2), "p_value": round(p, 3), "n": len(e)})
    return pd.DataFrame(out).set_index(["h", "model"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--spec", choices=["level", "change"], default="level")
    a = ap.parse_args()
    if a.synthetic:
        data = synthetic(); print("SYNTHETIC DATA - smoke test only")
    else:
        data = pd.DataFrame({k: fetch(v) for k, v in KEYS.items()})
        print("Data:", data.index.min(), "->", data.index.max(), "| n =", len(data))
    print("Spec:", a.spec)

    res = run(data, a.spec)
    tag = a.spec
    print("\n=== Full sample ===")
    full = rmse_table(res); print(full.to_string())

    in_shock = res["target"].dt.year.between(*SHOCK)
    print(f"\n=== Excluding targets in {SHOCK[0]}-{SHOCK[1]} ===")
    sub = rmse_table(res[~in_shock]); print(sub.to_string())

    print("\n=== Diebold-Mariano vs RW (full sample) ===")
    dm = dm_table(res); print(dm.to_string())

    print("\n=== Diagnostic: max forecast vs max actual (h=1) ===")
    h1 = res[res.h == 1]
    print(h1.groupby("model").forecast.max().round(2).to_string())
    print("max actual:", round(h1.actual.max(), 2))

    res.to_csv(f"forecasts_{tag}.csv", index=False)
    full.to_csv(f"rmse_table_{tag}.csv")
    sub.to_csv(f"rmse_table_ex{SHOCK[0]}_{SHOCK[1]}_{tag}.csv")
    dm.to_csv(f"dm_tests_{tag}.csv")

    p = res[res.h == 12].pivot_table(index="origin", columns="model", values="forecast")
    act = res[(res.h == 12) & (res.model == "RW")].set_index("origin")["actual"]
    fig, ax = plt.subplots(figsize=(9, 4))
    act.index = act.index.to_timestamp(); p.index = p.index.to_timestamp()
    ax.plot(act.index, act.values, "k", lw=2, label="Actual (t+12)")
    for m in p.columns: ax.plot(p.index, p[m], lw=1, label=m)
    ax.set_title(f"12-month-ahead forecasts of euro area HICP inflation ({tag} spec, by forecast origin)")
    ax.set_ylabel("% y/y"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"h12_forecasts_{tag}.png", dpi=150)
    print(f"\nSaved files with suffix _{tag}")


if __name__ == "__main__":
    main()