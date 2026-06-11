#!/usr/bin/env python3
"""
suggest_next.py  —  Bayesian Optimisation advisor for CodeBERT hyperparameter search.

Fits a Gaussian Process surrogate on observed (threshold, pooling, model) →
target_metric, then uses Expected Improvement (EI) to recommend the next
configuration to try.

The search space is intentionally small:
  - threshold: 0.05–0.99 (continuous)
  - pooling:   cls (0.0) / mean (1.0)
  - model:     codebert-base (0.0) / graphcodebert-base (1.0)

Usage:
  ../results-analyzer/.venv/bin/python suggest_next.py
  ../results-analyzer/.venv/bin/python suggest_next.py --metric auc
  ../results-analyzer/.venv/bin/python suggest_next.py --metric mcc --top 5
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
RUNS_CSV = HERE / "out" / "codebert_runs.csv"

POOLING_ENCODE: dict[str, float] = {"cls": 0.0, "mean": 1.0}
MODEL_ENCODE: dict[str, float] = {
    "codebert-base":      0.0,
    "graphcodebert-base": 1.0,
}
KNOWN_MODELS = list(MODEL_ENCODE.keys())
KNOWN_POOLINGS = list(POOLING_ENCODE.keys())

THRESHOLD_GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def _model_enc(model_name: str) -> float:
    short = Path(model_name).name
    return MODEL_ENCODE.get(short, 0.5)


def encode(threshold: float, pooling: str, model_name: str) -> list[float]:
    return [float(threshold), POOLING_ENCODE[pooling], _model_enc(model_name)]


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.array([
        encode(float(r.threshold), str(r.pooling), str(r.model))
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
    mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float
) -> np.ndarray:
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
    def mu(x: list[float]) -> float:
        return float(gp.predict(scaler.transform([x]))[0])

    t, pool, model_e = point
    grad: dict[str, float] = {}

    t_hi = min(round(t + 0.05, 2), 0.95)
    t_lo = max(round(t - 0.05, 2), 0.05)
    if t_hi != t_lo:
        grad["threshold"] = (mu([t_hi, pool, model_e]) - mu([t_lo, pool, model_e])) / (t_hi - t_lo)
    else:
        grad["threshold"] = 0.0

    for other_pool_label, other_pool_enc in POOLING_ENCODE.items():
        if other_pool_enc == pool:
            continue
        grad[f"pooling → {other_pool_label}"] = mu([t, other_pool_enc, model_e]) - mu([t, pool, model_e])

    for other_model_label, other_model_enc in MODEL_ENCODE.items():
        if other_model_enc == model_e:
            continue
        grad[f"model → {other_model_label}"] = mu([t, pool, other_model_enc]) - mu([t, pool, model_e])

    return grad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest next CodeBERT hyperparameter configuration via Bayesian Optimisation."
    )
    parser.add_argument(
        "--metric", default="f1", choices=["f1", "auc", "accuracy", "mcc"],
        help="Metric to maximise (default: f1)",
    )
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--xi", type=float, default=0.01)
    parser.add_argument("--diversity", type=float, default=0.4)
    args = parser.parse_args()

    if not RUNS_CSV.exists():
        sys.exit(f"ERROR: {RUNS_CSV} not found — run codebert_runner.py first.")

    df = pd.read_csv(RUNS_CSV)
    if len(df) < 3:
        sys.exit("Need at least 3 observed runs to fit a GP.")

    if args.metric not in df.columns:
        sys.exit(f"ERROR: column '{args.metric}' not found in {RUNS_CSV}.")

    y = df[args.metric].to_numpy(dtype=float)
    X = feature_matrix(df)

    if "tn" in df.columns and "fn" in df.columns:
        degen = df[(df["tn"] == 0) | (df["fn"] == 0)]
        if not degen.empty:
            print(f"\nWARNING: {len(degen)} degenerate run(s) (TN=0 or FN=0):")
            for _, r in degen.iterrows():
                print(f"  {r['run_name']}  F1={r[args.metric]:.4f}  ← inflated")

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
    print(f"  threshold={best_row['threshold']:.2f}  pooling={best_row['pooling']}  model={best_row['model']}")
    print(f"{'='*60}")

    tried: set[tuple] = set(
        zip(df["threshold"].round(2), df["pooling"], df["model"].apply(lambda m: Path(m).name))
    )

    candidates: list[tuple[float, str, str]] = []
    for t in THRESHOLD_GRID:
        for pool in KNOWN_POOLINGS:
            for m_label in KNOWN_MODELS:
                if (round(float(t), 2), pool, m_label) not in tried:
                    candidates.append((float(t), pool, m_label))

    if not candidates:
        print("\nAll grid points already tried.")
        return

    X_cand = np.array([encode(t, pool, m) for t, pool, m in candidates])
    mu_arr, sigma = gp.predict(scaler.transform(X_cand), return_std=True)
    ei = expected_improvement(mu_arr, sigma, f_best, args.xi)

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
    print(f"  {'#':<3} {'thresh':>7} {'pooling':<6} {'model':<22}  "
          f"{'EI':>9}  {'pred':>8}  {'±std':>7}  note")
    print("  " + "-" * 80)
    for rank, i in enumerate(selected_indices, 1):
        t, pool, m = candidates[i]
        note = ""
        if mu_arr[i] > f_best:
            note = "exploit"
        elif sigma[i] > np.percentile(sigma, 75):
            note = "explore"
        print(f"  {rank:<3} {t:>7.2f} {pool:<6} {m:<22}  "
              f"{ei[i]:>9.5f}  {mu_arr[i]:>8.4f}  ±{sigma[i]:.4f}  {note}")

    print(f"\nTo run top suggestion:")
    if selected_indices:
        t, pool, m = candidates[selected_indices[0]]
        m_full = f"microsoft/{m}"
        print(f"  python codebert_runner.py --threshold {t:.2f} --pooling {pool} --model {m_full}")

    best_point = encode(
        float(best_row["threshold"]),
        str(best_row["pooling"]),
        str(best_row["model"]),
    )
    grad = gp_gradient(gp, scaler, best_point)

    print(f"\nGP gradient at current best  (how {args.metric.upper()} changes per unit):")
    for param, g in grad.items():
        if "→" in param:
            arrow = "better" if g > 0 else "worse"
            print(f"  {param:<36}  {g:+.4f}  ({arrow})")
        else:
            direction = "↑ increase" if g > 0 else "↓ decrease"
            print(f"  ∂{args.metric}/∂{param:<30}  {g:+.4f}  → {direction}")

    print(f"\nObserved landscape summary ({args.metric.upper()}):")
    for col in ["pooling", "model"]:
        summary = (
            df.groupby(col)[args.metric]
            .agg(["count", "max", "mean"])
            .rename(columns={"count": "runs", "max": "best", "mean": "avg"})
        )
        print(f"\nBy {col}:")
        print(summary.to_string())


if __name__ == "__main__":
    main()
