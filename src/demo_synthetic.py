"""Sanity check of the calibration pipeline on synthetic logits (no training needed).

Labels are drawn from softmax(z), so z is calibrated by construction; the "model" reports z * T_true
(overconfident for T_true > 1). Same split sizes as CIFAR-100: 5,000 validation / 10,000 test, K = 100.
Writes results/tables/synthetic_check.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import HistogramBinning, IsotonicCalibration
from calibration_metrics import all_metrics, softmax
from temperature_scaling import TemperatureScaler

ROOT = Path(__file__).resolve().parents[1]


def main(t_true=2.5, k=100, n_val=5000, n_test=10000, seeds=(0, 1, 2)):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        z = rng.normal(0, 2.0, size=(n_val + n_test, k))
        y = (softmax(z).cumsum(1) > rng.random((n_val + n_test, 1))).argmax(1)
        z = z * t_true
        zv, yv, zt, yt = z[:n_val], y[:n_val], z[n_val:], y[n_val:]
        ts = TemperatureScaler().fit(zv, yv)
        for name, p in [("uncalibrated", softmax(zt)), ("temp_scaling", ts.predict_proba(zt)),
                        ("hist_binning", HistogramBinning(15).fit(zv, yv).predict_proba(zt)),
                        ("isotonic", IsotonicCalibration().fit(zv, yv).predict_proba(zt))]:
            rows.append({"seed": seed, "method": name, "fitted_T": ts.t_, **all_metrics(p, yt)})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "synthetic_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    s = df.groupby("method", sort=False).agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.drop(columns=["seed_mean", "seed_std"]).to_csv(out, float_format="%.5f")
    print(f"true T = {t_true}, fitted T = {df.fitted_T.mean():.3f} +/- {df.fitted_T.std():.3f} (3 seeds)")
    for m, r in s.iterrows():
        print(f"{m:13s} error {100 * r.error_mean:.2f}%  ECE {100 * r.ece_mean:.2f} +/- {100 * r.ece_std:.2f}%  "
              f"NLL {r.nll_mean:.3f}")


if __name__ == "__main__":
    main()
