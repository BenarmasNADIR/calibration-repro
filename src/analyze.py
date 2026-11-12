"""Analyze saved logits: calibration before/after TS, histogram binning and isotonic regression.

Usage: uv run python src/analyze.py [--logits-dir results/logits] [--out-dir results]

Every calibrator is fit on the 5,000 validation logits only and evaluated on the 10,000 test logits.
Writes results/tables/*.csv, results/tables/*.md and results/figures/*.png.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import HistogramBinning, IsotonicCalibration
from calibration_metrics import all_metrics, softmax
from reliability_diagram import save_reliability_grid
from temperature_scaling import TemperatureScaler

ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"resnet(\d+)_wd([0-9.e+-]+)_seed(\d+)\.npz$")
METHODS = ["uncalibrated", "temp_scaling", "hist_binning", "isotonic"]
# Guo et al. (2017), CIFAR-100 ResNet-110: Table 1 (ECE), S1 (MCE), S2 (error), S3 (NLL).
PAPER = {
    "uncalibrated": {"ece": 16.53, "mce": 35.5, "error": 27.83, "nll": 1.4978},
    "hist_binning": {"ece": 2.66, "mce": 7.03, "error": 34.78, "nll": 1.4379},
    "isotonic": {"ece": 4.99, "mce": 10.36, "error": 28.41, "nll": 1.207},
    "temp_scaling": {"ece": 1.26, "mce": 4.74, "error": 27.83, "nll": 1.0442},
}


def evaluate_run(path, n_bins=15):
    d = np.load(path)
    zv, yv, zt, yt = d["val_logits"], d["val_labels"], d["test_logits"], d["test_labels"]
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    ts = TemperatureScaler().fit(zv, yv)
    probs = {
        "uncalibrated": softmax(zt),
        "temp_scaling": ts.predict_proba(zt),
        "hist_binning": HistogramBinning(n_bins).fit(zv, yv).predict_proba(zt),
        "isotonic": IsotonicCalibration().fit(zv, yv).predict_proba(zt),
    }
    rows = []
    for m in METHODS:
        r = all_metrics(probs[m], yt, n_bins)
        rows.append({"method": m, **r, "temperature": ts.t_ if m == "temp_scaling" else np.nan})
    return rows, probs, yt, meta


def mean_std(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), (x.std(ddof=1) if len(x) > 1 else np.nan)


def fmt(m, s, pct=True, nd=2):
    k = 100 if pct else 1
    return f"{k * m:.{nd}f}" + ("" if np.isnan(s) else f" ± {k * s:.{nd}f}")


def summarize(df, keys):
    """Mean and sample std (ddof=1) over seeds for every metric, grouped by `keys`."""
    out = []
    for g, sub in df.groupby(keys, sort=True):
        row = dict(zip(keys, g if isinstance(g, tuple) else (g,)))
        row["n_seeds"] = sub["seed"].nunique()
        for c in ["error", "nll", "ece", "mce", "temperature"]:
            row[f"{c}_mean"], row[f"{c}_std"] = mean_std(sub[c].dropna()) if sub[c].notna().any() else (np.nan, np.nan)
        out.append(row)
    return pd.DataFrame(out)


def ablation_plot(summ, xkey, path, title, logx=False):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for method, style, label in [("uncalibrated", "o-", "ECE (uncalibrated)"),
                                 ("temp_scaling", "s--", "ECE (temp. scaling)")]:
        s = summ[summ.method == method].sort_values(xkey)
        ax.errorbar(s[xkey], 100 * s.ece_mean, yerr=100 * s.ece_std.fillna(0), fmt=style,
                    capsize=3, label=label)
    s = summ[summ.method == "uncalibrated"].sort_values(xkey)
    ax.errorbar(s[xkey], 100 * s.error_mean, yerr=100 * s.error_std.fillna(0), fmt="^:",
                capsize=3, color="gray", label="Test error")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xkey.replace("_", " ").capitalize())
    ax.set_ylabel("%")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits-dir", default=str(ROOT / "results/logits"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--n-bins", type=int, default=15)
    args = ap.parse_args()
    tables, figs = Path(args.out_dir) / "tables", Path(args.out_dir) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in Path(args.logits_dir).glob("*.npz") if NAME_RE.search(p.name))
    if not files:
        print(f"No logits found in {args.logits_dir}. Run notebooks/run_kaggle.ipynb first.")
        return

    rows, times, cache = [], [], {}
    for p in files:
        depth, wd, seed = NAME_RE.search(p.name).groups()
        depth, wd, seed = int(depth), float(wd), int(seed)
        run_rows, probs, yt, meta = evaluate_run(p, args.n_bins)
        for r in run_rows:
            rows.append({"run": p.stem, "depth": depth, "weight_decay": wd, "seed": seed, **r})
        times.append({"run": p.stem, "gpu": meta.get("gpu", ""), "amp": meta.get("amp", ""),
                      "train_time_min": meta.get("train_time_s", np.nan) / 60})
        cache[(depth, wd, seed)] = (probs, yt)
        print(f"{p.stem}: ECE {100 * run_rows[0]['ece']:.2f}% -> TS {100 * run_rows[1]['ece']:.2f}% "
              f"(T={run_rows[1]['temperature']:.3f})")
    df = pd.DataFrame(rows)
    df.to_csv(tables / "per_run.csv", index=False, float_format="%.6f")
    pd.DataFrame(times).to_csv(tables / "train_times.csv", index=False, float_format="%.1f")

    # Main result: ResNet-110, wd 1e-4.
    main_df = df[(df.depth == 110) & np.isclose(df.weight_decay, 1e-4)]
    if len(main_df):
        s = summarize(main_df, ["method"]).set_index("method").loc[METHODS].reset_index()
        for c in ["ece", "mce", "error", "nll"]:
            s[f"paper_{c}"] = [PAPER[m][c] for m in s.method]
        s.to_csv(tables / "main_resnet110.csv", index=False, float_format="%.6f")
        lines = ["| Method | Error (%) | NLL | ECE (%) | MCE (%) | Paper ECE (%) |",
                 "|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            lines.append(f"| {r.method} | {fmt(r.error_mean, r.error_std)} | "
                         f"{fmt(r.nll_mean, r.nll_std, pct=False, nd=3)} | {fmt(r.ece_mean, r.ece_std)} | "
                         f"{fmt(r.mce_mean, r.mce_std)} | {PAPER[r.method]['ece']:.2f} |")
        t_m, t_s = s.loc[s.method == "temp_scaling", ["temperature_mean", "temperature_std"]].iloc[0]
        lines.append(f"\nResNet-110, n_seeds = {int(s.n_seeds.max())}; fitted T = {fmt(t_m, t_s, pct=False, nd=3)}.")
        (tables / "main_resnet110.md").write_text("\n".join(lines) + "\n", encoding="utf8")
        key = min((k for k in cache if k[0] == 110 and np.isclose(k[1], 1e-4)), key=lambda k: k[2])
        seed0 = key[2]
        probs, yt = cache[key]
        save_reliability_grid([("Uncalibrated", probs["uncalibrated"]),
                               ("Temp. scaling", probs["temp_scaling"]),
                               ("Hist. binning", probs["hist_binning"]),
                               ("Isotonic", probs["isotonic"])], yt,
                              figs / "reliability_resnet110.png",
                              suptitle=f"ResNet-110, CIFAR-100 test set (seed {seed0})", n_bins=args.n_bins)

    # Weight-decay ablation: ResNet-56.
    wd_df = df[df.depth == 56]
    if wd_df.weight_decay.nunique() > 1:
        s = summarize(wd_df, ["weight_decay", "method"])
        s.to_csv(tables / "wd_ablation.csv", index=False, float_format="%.6f")
        ablation_plot(s, "weight_decay", figs / "ece_vs_weight_decay.png",
                      "ResNet-56, CIFAR-100: weight decay", logx=True)

    # Depth ablation: wd 1e-4.
    dp_df = df[np.isclose(df.weight_decay, 1e-4)]
    if dp_df.depth.nunique() > 1:
        s = summarize(dp_df, ["depth", "method"])
        s.to_csv(tables / "depth_ablation.csv", index=False, float_format="%.6f")
        ablation_plot(s, "depth", figs / "ece_vs_depth.png", "CIFAR-100, wd 1e-4: depth")

    print(f"Wrote tables to {tables} and figures to {figs}")


if __name__ == "__main__":
    main()
