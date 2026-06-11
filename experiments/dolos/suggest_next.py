#!/usr/bin/env python3
"""
suggest_next.py  —  Bayesian Optimisation advisor for Dolos hyperparameter search.

Fits a Gaussian Process surrogate on observed (kgram, threshold, metric) →
target_metric, then uses the Expected Improvement (EI) acquisition function to
recommend the most promising next configuration to try.

EI balances:
  exploitation — points the GP predicts will exceed the current best
  exploration  — points where the GP is uncertain (high variance)

Also prints a finite-difference gradient of the GP mean at the current best,
so you can see which direction each parameter should move.

Note: window is not a search dimension — use the default (17) or override with
--window.  The printed command includes the window value for reproducibility.

Usage:
  ../results-analyzer/.venv/bin/python suggest_next.py
  ../results-analyzer/.venv/bin/python suggest_next.py --metric f1
  ../results-analyzer/.venv/bin/python suggest_next.py --metric auc
  ../results-analyzer/.venv/bin/python suggest_next.py --top 5
  ../results-analyzer/.venv/bin/python suggest_next.py --xi 0.05
  ../results-analyzer/.venv/bin/python suggest_next.py --kgram-range 5 30
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
RUNS_CSV = HERE / "out" / "dolos_runs.csv"

# Ordinal encoding spread over [0, 1] for the metric categorical variable.
# The GP will learn which metric regions perform best.
METRIC_ENCODE: dict[str, float] = {
    "COMBINED":    0.0,
    "MAX":         0.25,
    "AVG":         0.5,
    "SUB_IN_ORIG": 0.75,
    "ORIG_IN_SUB": 1.0,
}
DECODE_METRIC = {v: k for k, v in METRIC_ENCODE.items()}
DOLOS_METRICS = list(METRIC_ENCODE.keys())

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

DEFAULT_WINDOW = 17


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode(kgram: int, threshold: float, metric: str) -> list[float]:
    return [float(kgram), float(threshold), METRIC_ENCODE[metric]]


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.array([
        encode(int(r.kgram), float(r.threshold), str(r.metric))
        for _, r in df.iterrows()
    ])


# ---------------------------------------------------------------------------
# GP fitting
# ---------------------------------------------------------------------------

def fit_gp(
    X: np.ndarray, y: np.ndarray
) -> tuple[GaussianProcessRegressor, StandardScaler]:
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(
            length_scale=np.ones(X.shape[1]),
            length_scale_bounds=[(0.1, 50.0)] * X.shape[1],
        )
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
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float,
) -> np.ndarray:
    """EI = E[max(f(x) - f_best - xi, 0)] under the GP posterior."""
    imp = mu - f_best - xi
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(sigma > 1e-9, imp / sigma, 0.0)
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 1e-9] = 0.0
    return np.maximum(ei, 0.0)


# ---------------------------------------------------------------------------
# Gradient of GP mean (finite differences)
# ---------------------------------------------------------------------------

def gp_gradient(
    gp: GaussianProcessRegressor,
    scaler: StandardScaler,
    point: list[float],
    kgram_max: int,
) -> dict[str, float]:
    """
    Estimate ∂(GP mean)/∂param at `point` via finite differences.
    For the categorical metric dimension, reports the effect of each metric
    vs the current one instead.
    """
    def mu(x: list[float]) -> float:
        return float(gp.predict(scaler.transform([x]))[0])

    kg, t, m = point
    grad: dict[str, float] = {}

    # ∂/∂threshold  (step = ±0.05)
    t_hi = min(round(t + 0.05, 2), 0.95)
    t_lo = max(round(t - 0.05, 2), 0.05)
    if t_hi != t_lo:
        grad["threshold"] = (mu([kg, t_hi, m]) - mu([kg, t_lo, m])) / (t_hi - t_lo)
    else:
        grad["threshold"] = 0.0

    # ∂/∂kgram  (step = ±1)
    kg_hi = min(kg + 1, float(kgram_max))
    kg_lo = max(kg - 1, 1.0)
    if kg_hi != kg_lo:
        grad["kgram"] = (mu([kg_hi, t, m]) - mu([kg_lo, t, m])) / (kg_hi - kg_lo)
    else:
        grad["kgram"] = 0.0

    # Categorical metric: show effect of switching to each alternative
    current_label = DECODE_METRIC.get(m, str(m))
    for other_label, other_enc in METRIC_ENCODE.items():
        if other_enc == m:
            continue
        grad[f"metric → {other_label}"] = mu([kg, t, other_enc]) - mu([kg, t, m])

    return grad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest next Dolos hyperparameter configuration via Bayesian Optimisation."
    )
    parser.add_argument(
        "--metric", default="f1", choices=["f1", "auc", "accuracy", "mcc"],
        help="Metric to maximise (default: f1)",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of suggestions to show (default: 5)",
    )
    parser.add_argument(
        "--xi", type=float, default=0.01,
        help="EI exploration bonus ξ. 0 = pure exploitation, 0.1 = strong exploration (default: 0.01)",
    )
    parser.add_argument(
        "--kgram-range", type=int, nargs=2, default=[5, 30], metavar=("MIN", "MAX"),
        help="kgram search range (default: 5 30)",
    )
    parser.add_argument(
        "--diversity", type=float, default=0.4,
        help="Minimum normalised distance between suggestions (0 = no filter, default: 0.4).",
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help=f"Window size to include in suggested command (default: {DEFAULT_WINDOW})",
    )
    args = parser.parse_args()

    if not RUNS_CSV.exists():
        sys.exit(f"ERROR: {RUNS_CSV} not found — run dolos_runner.py first.")

    df = pd.read_csv(RUNS_CSV)
    if len(df) < 3:
        sys.exit("Need at least 3 observed runs to fit a GP.")

    if args.metric not in df.columns:
        sys.exit(f"ERROR: column '{args.metric}' not found in {RUNS_CSV}.")

    y = df[args.metric].to_numpy(dtype=float)
    X = feature_matrix(df)

    # Warn about degenerate runs (TN=0 or FN=0 means all-positive predictions)
    if "tn" in df.columns and "fn" in df.columns:
        degen = df[(df["tn"] == 0) | (df["fn"] == 0)]
        if not degen.empty:
            print(f"\nWARNING: {len(degen)} degenerate run(s) detected "
                  f"(TN=0 or FN=0 — threshold too low, predicting all-positive):")
            for _, r in degen.iterrows():
                print(f"  {r['run_name']}  "
                      f"TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}  "
                      f"F1={r[args.metric]:.4f}  ← inflated, not trustworthy")

    print(f"Fitting GP on {len(df)} observed runs (target: {args.metric.upper()})...")
    gp, scaler = fit_gp(X, y)

    y_pred = gp.predict(scaler.transform(X))
    residuals = np.abs(y - y_pred)
    print(f"  GP fit residuals — mean: {residuals.mean():.4f}, max: {residuals.max():.4f}")
    print(f"  Learned kernel: {gp.kernel_}")

    best_idx = int(np.argmax(y))
    best_row = df.iloc[best_idx]
    f_best   = float(y[best_idx])

    print(f"\n{'='*60}")
    print(f"Current best  {args.metric.upper()} = {f_best:.4f}")
    print(f"  {best_row['run_name']}")
    print(f"  kgram={best_row['kgram']}  threshold={best_row['threshold']:.2f}  metric={best_row['metric']}")
    print(f"{'='*60}")

    # Build search grid of untried configurations
    tried: set[tuple] = set(
        zip(
            df["kgram"].astype(int),
            df["threshold"].round(2),
            df["metric"],
        )
    )
    kg_min, kg_max = args.kgram_range
    candidates: list[tuple[int, float, str]] = []
    for kg in range(kg_min, kg_max + 1):
        for t in THRESHOLD_GRID:
            for m in DOLOS_METRICS:
                if (int(kg), round(float(t), 2), m) not in tried:
                    candidates.append((kg, float(t), m))

    if not candidates:
        print("\nAll grid points in range already tried.")
        print("Expand the range with --kgram-range or reduce threshold step size.")
        return

    X_cand = np.array([encode(kg, t, m) for kg, t, m in candidates])
    mu_arr, sigma = gp.predict(scaler.transform(X_cand), return_std=True)
    ei = expected_improvement(mu_arr, sigma, f_best, args.xi)

    # Top suggestions with diversity filter
    X_cand_scaled = scaler.transform(X_cand)
    sorted_by_ei  = np.argsort(ei)[::-1]

    selected_indices: list[int] = []
    selected_scaled:  list[np.ndarray] = []

    for i in sorted_by_ei:
        if args.diversity > 0 and selected_scaled:
            dists = [np.linalg.norm(X_cand_scaled[i] - s) for s in selected_scaled]
            if min(dists) < args.diversity:
                continue
        selected_indices.append(i)
        selected_scaled.append(X_cand_scaled[i])
        if len(selected_indices) == args.top:
            break

    print(f"\nTop {args.top} suggestions  "
          f"(xi={args.xi}, diversity≥{args.diversity}, {len(candidates)} untried points)")
    print(f"  {'#':<3} {'kgram':>6} {'thresh':>7} {'metric':<12}  "
          f"{'EI':>9}  {'pred':>8}  {'±std':>7}  note")
    print("  " + "-" * 76)
    for rank, i in enumerate(selected_indices, 1):
        kg, t, m = candidates[i]
        note = ""
        if mu_arr[i] > f_best:
            note = "exploit (predicted improvement)"
        elif sigma[i] > np.percentile(sigma, 75):
            note = "explore (high uncertainty)"
        print(f"  {rank:<3} {kg:>6} {t:>7.2f} {m:<12}  "
              f"{ei[i]:>9.5f}  {mu_arr[i]:>8.4f}  ±{sigma[i]:.4f}  {note}")

    print(f"\nTo run top suggestion:")
    if selected_indices:
        kg, t, m = candidates[selected_indices[0]]
        print(f"  python dolos_runner.py --kgram {kg} --threshold {t:.2f} --metric {m}")

    # GP gradient at current best
    best_point = encode(
        int(best_row["kgram"]),
        float(best_row["threshold"]),
        str(best_row["metric"]),
    )
    grad = gp_gradient(gp, scaler, best_point, kg_max)

    print(f"\nGP gradient at current best  (how {args.metric.upper()} changes per unit of each param):")
    for param, g in grad.items():
        if "metric" in param:
            arrow = "better" if g > 0 else "worse"
            print(f"  {param:<32}  {g:+.4f}  ({arrow} than current metric)")
        else:
            direction = "↑ increase" if g > 0 else "↓ decrease"
            print(f"  ∂{args.metric}/∂{param:<26}  {g:+.4f}  → {direction}")

    # Summary of observed landscape
    print(f"\nObserved landscape summary ({args.metric.upper()}):")
    summary = (
        df.groupby("metric")[args.metric]
        .agg(["count", "max", "mean"])
        .rename(columns={"count": "runs", "max": "best", "mean": "avg"})
    )
    print(summary.to_string())

    best_per_kgram = (
        df.sort_values(args.metric, ascending=False)
        .groupby("kgram")
        .first()[["threshold", "metric", args.metric]]
        .rename(columns={args.metric: "best_" + args.metric})
    )
    print(f"\nBest {args.metric.upper()} seen per kgram:")
    print(best_per_kgram.to_string())


if __name__ == "__main__":
    main()
