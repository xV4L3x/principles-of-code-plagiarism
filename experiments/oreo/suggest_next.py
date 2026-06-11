#!/usr/bin/env python3
"""
suggest_next.py  —  Bayesian Optimisation advisor for Oreo threshold search.

Fits a Gaussian Process surrogate on observed threshold → target_metric,
then uses Expected Improvement (EI) to recommend the next threshold to try.

Note: Oreo similarity scores are discrete {0.0, 0.5, 1.0}, so the useful
threshold range is narrow and the GP will converge quickly.

Usage:
  ../results-analyzer/.venv/bin/python suggest_next.py
  ../results-analyzer/.venv/bin/python suggest_next.py --metric mcc
  ../results-analyzer/.venv/bin/python suggest_next.py --metric auc --top 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
RUNS_CSV = HERE / "out" / "oreo_runs.csv"

THRESHOLD_GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)


# ---------------------------------------------------------------------------
# GP fitting
# ---------------------------------------------------------------------------

def fit_gp(X: np.ndarray, y: np.ndarray) -> tuple[GaussianProcessRegressor, StandardScaler]:
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(0.05, 20.0))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 0.1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=25,
        normalize_y=True,
        random_state=42,
    ).fit(X_s, y)
    return gp, scaler


# ---------------------------------------------------------------------------
# Expected Improvement
# ---------------------------------------------------------------------------

def expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float
) -> np.ndarray:
    imp = mu - f_best - xi
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(sigma > 1e-9, imp / sigma, 0.0)
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 1e-9] = 0.0
    return np.maximum(ei, 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest next Oreo threshold via Bayesian Optimisation."
    )
    parser.add_argument(
        "--metric", default="f1", choices=["f1", "auc", "accuracy", "mcc"],
        help="Metric to maximise (default: f1)",
    )
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--xi", type=float, default=0.01)
    args = parser.parse_args()

    if not RUNS_CSV.exists():
        sys.exit(f"ERROR: {RUNS_CSV} not found — run oreo_runner.py first.")

    df = pd.read_csv(RUNS_CSV)

    if len(df) < 2:
        sys.exit("Need at least 2 observed runs to fit a GP.")

    if args.metric not in df.columns:
        sys.exit(f"ERROR: column '{args.metric}' not found in {RUNS_CSV}.")

    y = df[args.metric].to_numpy(dtype=float)
    X = df["threshold"].to_numpy(dtype=float).reshape(-1, 1)

    if "tn" in df.columns and "fn" in df.columns:
        degen = df[(df["tn"] == 0) | (df["fn"] == 0)]
        if not degen.empty:
            print(f"\nWARNING: {len(degen)} degenerate run(s) (TN=0 or FN=0):")
            for _, r in degen.iterrows():
                print(f"  {r['run_name']}  {args.metric.upper()}={r[args.metric]:.4f}  ← inflated")

    print(f"Fitting GP on {len(df)} observed runs (target: {args.metric.upper()})...")
    gp, scaler = fit_gp(X, y)

    y_pred = gp.predict(scaler.transform(X))
    residuals = np.abs(y - y_pred)
    print(f"  GP fit residuals — mean: {residuals.mean():.4f}, max: {residuals.max():.4f}")
    print(f"  Learned kernel: {gp.kernel_}")

    best_idx = int(np.argmax(y))
    best_row = df.iloc[best_idx]
    f_best   = float(y[best_idx])

    print(f"\n{'='*55}")
    print(f"Current best  {args.metric.upper()} = {f_best:.4f}")
    print(f"  {best_row['run_name']}")
    print(f"  threshold={best_row['threshold']:.2f}")
    print(f"{'='*55}")

    tried: set[float] = set(df["threshold"].round(2))

    candidates = [float(t) for t in THRESHOLD_GRID if round(float(t), 2) not in tried]

    if not candidates:
        print("\nAll grid points already tried.")
        return

    X_cand = np.array(candidates).reshape(-1, 1)
    mu_arr, sigma = gp.predict(scaler.transform(X_cand), return_std=True)
    ei = expected_improvement(mu_arr, sigma, f_best, args.xi)

    sorted_by_ei = np.argsort(ei)[::-1]
    top_n = min(args.top, len(candidates))

    print(f"\nTop {top_n} suggestions  "
          f"(xi={args.xi}, {len(candidates)} untried thresholds)")
    print(f"  {'#':<3} {'thresh':>7}  {'EI':>9}  {'pred':>8}  {'±std':>7}  note")
    print("  " + "-" * 50)
    for rank, i in enumerate(sorted_by_ei[:top_n], 1):
        t = candidates[i]
        note = "exploit" if mu_arr[i] > f_best else (
            "explore" if sigma[i] > float(np.percentile(sigma, 75)) else ""
        )
        print(f"  {rank:<3} {t:>7.2f}  {ei[i]:>9.5f}  {mu_arr[i]:>8.4f}  "
              f"±{sigma[i]:.4f}  {note}")

    print(f"\nTo run top suggestion:")
    if sorted_by_ei.size > 0:
        t = candidates[sorted_by_ei[0]]
        print(f"  python oreo_runner.py --threshold {t:.2f}")

    # Gradient at current best (finite differences on GP mean)
    t_best = float(best_row["threshold"])
    t_hi = min(round(t_best + 0.05, 2), 0.95)
    t_lo = max(round(t_best - 0.05, 2), 0.05)

    def mu_at(t: float) -> float:
        return float(gp.predict(scaler.transform([[t]]))[0])

    if t_hi != t_lo:
        grad = (mu_at(t_hi) - mu_at(t_lo)) / (t_hi - t_lo)
        direction = "↑ increase" if grad > 0 else "↓ decrease"
        print(f"\nGP gradient at t={t_best:.2f}:  "
              f"∂{args.metric}/∂threshold = {grad:+.4f}  → {direction}")

    print(f"\nObserved landscape ({args.metric.upper()}):")
    summary = df[["threshold", args.metric]].sort_values("threshold")
    for _, r in summary.iterrows():
        bar = "█" * int(r[args.metric] * 30)
        print(f"  t={r['threshold']:.2f}  {r[args.metric]:.4f}  {bar}")


if __name__ == "__main__":
    main()
