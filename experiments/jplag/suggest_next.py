#!/usr/bin/env python3
"""
suggest_next.py  —  Bayesian Optimisation advisor for JPlag hyperparameter search.

Fits a Gaussian Process surrogate on observed (min_tokens, threshold,
similarity_metric) → target_metric, then uses the Expected Improvement (EI)
acquisition function to recommend the most promising next configuration.

EI balances:
  exploitation — points the GP predicts will exceed the current best
  exploration  — points where the GP is uncertain (high variance)

Also prints a finite-difference gradient of the GP mean at the current best,
so you can see which direction each parameter should move.

Usage:
  python suggest_next.py
  python suggest_next.py --metric f1          # optimise F1 (default)
  python suggest_next.py --metric auc
  python suggest_next.py --metric accuracy
  python suggest_next.py --top 5
  python suggest_next.py --xi 0.05            # higher xi = more exploration
  python suggest_next.py --mt-range 1 8       # min_tokens search range
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
RUNS_CSV = HERE / "out" / "jplag_runs.csv"

METRIC_ENCODE = {"MAX": 0.0, "AVG": 1.0}
SIM_METRICS   = ["MAX", "AVG"]
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode(min_tokens: int, threshold: float, sim_metric: str) -> list[float]:
    return [float(min_tokens), float(threshold), METRIC_ENCODE[sim_metric]]


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.array([
        encode(r.min_tokens, r.threshold, r.similarity_metric)
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
    # Separate length scale per dimension so the GP can learn that
    # min_tokens and threshold have different natural scales.
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(
            length_scale=np.ones(X.shape[1]),
            # min_tokens (int, range 1-10) and metric (0/1) are both smooth
            # on their natural scale — allow larger length scales for them.
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
) -> dict[str, float]:
    """
    Estimate ∂(GP mean)/∂param at `point` via finite differences.
    For the categorical metric dimension, reports AVG − MAX effect instead.
    """
    def mu(x: list[float]) -> float:
        return float(gp.predict(scaler.transform([x]))[0])

    mt, t, m = point
    grad: dict[str, float] = {}

    # ∂/∂threshold  (step = ±0.05, clipped to grid bounds)
    t_hi = min(round(t + 0.05, 2), 0.95)
    t_lo = max(round(t - 0.05, 2), 0.05)
    if t_hi != t_lo:
        grad["threshold"] = (mu([mt, t_hi, m]) - mu([mt, t_lo, m])) / (t_hi - t_lo)
    else:
        grad["threshold"] = 0.0

    # ∂/∂min_tokens  (step = ±1)
    mt_hi = min(mt + 1, 10.0)
    mt_lo = max(mt - 1, 1.0)
    if mt_hi != mt_lo:
        grad["min_tokens"] = (mu([mt_hi, t, m]) - mu([mt_lo, t, m])) / (mt_hi - mt_lo)
    else:
        grad["min_tokens"] = 0.0

    # categorical metric: effect of switching to the other metric
    other_m = METRIC_ENCODE["AVG"] if m == METRIC_ENCODE["MAX"] else METRIC_ENCODE["MAX"]
    other_label = "AVG" if m == METRIC_ENCODE["MAX"] else "MAX"
    grad[f"metric → {other_label}"] = mu([mt, t, other_m]) - mu([mt, t, m])

    return grad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest next JPlag hyperparameter configuration via Bayesian Optimisation."
    )
    parser.add_argument(
        "--metric", default="f1", choices=["f1", "auc", "accuracy"],
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
        "--mt-range", type=int, nargs=2, default=[1, 10], metavar=("MIN", "MAX"),
        help="min_tokens search range (default: 1 10)",
    )
    parser.add_argument(
        "--diversity", type=float, default=0.4,
        help="Minimum normalised distance between suggestions (0 = no filter, default: 0.4). "
             "Higher values force more spread-out suggestions.",
    )
    args = parser.parse_args()

    if not RUNS_CSV.exists():
        sys.exit(f"ERROR: {RUNS_CSV} not found — run jplag_runner.py first.")

    df = pd.read_csv(RUNS_CSV)
    if len(df) < 3:
        sys.exit("Need at least 3 observed runs to fit a GP.")

    if args.metric not in df.columns:
        sys.exit(f"ERROR: column '{args.metric}' not found in {RUNS_CSV}.")

    y = df[args.metric].to_numpy(dtype=float)
    X = feature_matrix(df)

    # ---- warn about degenerate runs (TN=0 or FN=0 for binary predictions) ----
    if "tn" in df.columns and "fn" in df.columns:
        degen = df[(df["tn"] == 0) | (df["fn"] == 0)]
        if not degen.empty:
            print(f"\nWARNING: {len(degen)} degenerate run(s) detected "
                  f"(TN=0 or FN=0 — threshold too low, predicting all-positive):")
            for _, r in degen.iterrows():
                print(f"  {r['run_name']}  "
                      f"TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}  "
                      f"F1={r[args.metric]:.4f}  ← inflated, not trustworthy")

    # ---- fit GP ----
    print(f"Fitting GP on {len(df)} observed runs (target: {args.metric.upper()})...")
    gp, scaler = fit_gp(X, y)

    y_pred = gp.predict(scaler.transform(X))
    residuals = np.abs(y - y_pred)
    print(f"  GP fit residuals — mean: {residuals.mean():.4f}, max: {residuals.max():.4f}")
    print(f"  Learned kernel: {gp.kernel_}")

    # ---- current best ----
    best_idx = int(np.argmax(y))
    best_row = df.iloc[best_idx]
    f_best   = float(y[best_idx])

    print(f"\n{'='*60}")
    print(f"Current best  {args.metric.upper()} = {f_best:.4f}")
    print(f"  {best_row['run_name']}")
    print(f"  min_tokens={best_row['min_tokens']}  "
          f"threshold={best_row['threshold']:.2f}  "
          f"metric={best_row['similarity_metric']}")
    print(f"{'='*60}")

    # ---- build search grid of untried configurations ----
    tried: set[tuple] = set(
        zip(
            df["min_tokens"].astype(int),
            df["threshold"].round(2),
            df["similarity_metric"],
        )
    )
    mt_min, mt_max = args.mt_range
    candidates: list[tuple[int, float, str]] = []
    for mt in range(mt_min, mt_max + 1):
        for t in THRESHOLD_GRID:
            for m in SIM_METRICS:
                if (int(mt), round(float(t), 2), m) not in tried:
                    candidates.append((mt, float(t), m))

    if not candidates:
        print("\nAll grid points in range already tried.")
        print("Expand the range with --mt-range or reduce threshold step size.")
        return

    X_cand = np.array([encode(mt, t, m) for mt, t, m in candidates])
    mu, sigma = gp.predict(scaler.transform(X_cand), return_std=True)
    ei = expected_improvement(mu, sigma, f_best, args.xi)

    # ---- top suggestions with diversity filter ----
    # Walk candidates in descending EI order; skip any that are within
    # `diversity` normalised distance of an already-selected suggestion.
    X_cand_scaled = scaler.transform(X_cand)
    sorted_by_ei  = np.argsort(ei)[::-1]

    selected_indices: list[int] = []
    selected_scaled: list[np.ndarray] = []

    for i in sorted_by_ei:
        if args.diversity > 0 and selected_scaled:
            dists = [np.linalg.norm(X_cand_scaled[i] - s) for s in selected_scaled]
            if min(dists) < args.diversity:
                continue  # too close to a suggestion we already picked
        selected_indices.append(i)
        selected_scaled.append(X_cand_scaled[i])
        if len(selected_indices) == args.top:
            break

    print(f"\nTop {args.top} suggestions  "
          f"(xi={args.xi}, diversity≥{args.diversity}, {len(candidates)} untried points)")
    print(f"  {'#':<3} {'mt':>4} {'thresh':>7} {'metric':>6}  "
          f"{'EI':>9}  {'pred':>8}  {'±std':>7}  note")
    print("  " + "-" * 68)
    for rank, i in enumerate(selected_indices, 1):
        mt, t, m = candidates[i]
        note = ""
        if mu[i] > f_best:
            note = "exploit (predicted improvement)"
        elif sigma[i] > np.percentile(sigma, 75):
            note = "explore (high uncertainty)"
        print(f"  {rank:<3} {mt:>4} {t:>7.2f} {m:>6}  "
              f"{ei[i]:>9.5f}  {mu[i]:>8.4f}  ±{sigma[i]:.4f}  {note}")

    # ---- gradient at current best ----
    best_point = encode(
        int(best_row["min_tokens"]),
        float(best_row["threshold"]),
        str(best_row["similarity_metric"]),
    )
    grad = gp_gradient(gp, scaler, best_point)

    print(f"\nGP gradient at current best  (how {args.metric.upper()} changes per unit of each param):")
    for param, g in grad.items():
        if "metric" in param:
            arrow = "better" if g > 0 else "worse"
            print(f"  {param:<28}  {g:+.4f}  ({arrow} than current metric)")
        else:
            direction = "↑ increase" if g > 0 else "↓ decrease"
            print(f"  ∂{args.metric}/∂{param:<22}  {g:+.4f}  → {direction}")

    # ---- summary of observed landscape ----
    print(f"\nObserved landscape summary ({args.metric.upper()}):")
    summary = (
        df.groupby("similarity_metric")[args.metric]
        .agg(["count", "max", "mean"])
        .rename(columns={"count": "runs", "max": "best", "mean": "avg"})
    )
    print(summary.to_string())

    best_per_mt = (
        df.sort_values(args.metric, ascending=False)
        .groupby("min_tokens")
        .first()[["threshold", "similarity_metric", args.metric]]
        .rename(columns={args.metric: "best_" + args.metric})
    )
    print(f"\nBest {args.metric.upper()} seen per min_tokens:")
    print(best_per_mt.to_string())


if __name__ == "__main__":
    main()
