#!/usr/bin/env python3
"""
suggest_next.py  —  Bayesian Optimisation advisor for CodeLlama hyperparameter search.

Fits a Gaussian Process surrogate on observed (threshold, model, max_context, quantization) →
target_metric, then uses Expected Improvement (EI) to recommend the next
configuration to try.

Search space:
  - threshold:    0.05–0.99 (continuous)
  - model:        codellama-7b  (0.0) / codellama-13b (0.5) / codellama-34b (1.0)
  - max_context:  1024 (0.0) / 2048 (0.33) / 4096 (0.67) / 8192 (1.0)
  - quantization: fp16 (0.0) / int4 (1.0)

Usage:
  python suggest_next.py
  python suggest_next.py --metric auc
  python suggest_next.py --metric mcc --top 5
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
RUNS_CSV = HERE / "out" / "codellama_runs.csv"

MODEL_ENCODE: dict[str, float] = {
    "mlx_codellama_4bit":              0.0,
    "CodeLlama-7b-Instruct-hf":        0.25,
    "CodeLlama-13b-Instruct-hf":       0.5,
    "CodeLlama-34b-Instruct-hf":       0.75,
    "CodeLlama-70b-Instruct-hf":       1.0,
}
MAX_CONTEXT_ENCODE: dict[int, float] = {
    1024: 0.0,
    2048: 0.33,
    4096: 0.67,
    8192: 1.0,
}
QUANT_ENCODE: dict[str, float] = {
    "fp16": 0.0,
    "int4": 1.0,
    "fp32": 0.5,
}

KNOWN_MODELS       = list(MODEL_ENCODE.keys())
KNOWN_MAX_CONTEXTS = list(MAX_CONTEXT_ENCODE.keys())
KNOWN_QUANTS       = list(QUANT_ENCODE.keys())

# Full HuggingFace id for models that need an org prefix in the run command
MODEL_HF_ID: dict[str, str] = {
    "mlx_codellama_4bit":         "./mlx_codellama_4bit",
    "CodeLlama-7b-Instruct-hf":   "codellama/CodeLlama-7b-Instruct-hf",
    "CodeLlama-13b-Instruct-hf":  "codellama/CodeLlama-13b-Instruct-hf",
    "CodeLlama-34b-Instruct-hf":  "codellama/CodeLlama-34b-Instruct-hf",
    "CodeLlama-70b-Instruct-hf":  "codellama/CodeLlama-70b-Instruct-hf",
}

THRESHOLD_GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def _model_enc(model_name: str) -> float:
    short = Path(model_name).name
    return MODEL_ENCODE.get(short, 0.5)


def _ctx_enc(max_context: int) -> float:
    return MAX_CONTEXT_ENCODE.get(int(max_context), 0.5)


def _quant_enc(quant: str) -> float:
    return QUANT_ENCODE.get(str(quant), 0.0)


def encode(threshold: float, model: str, max_context: int, quant: str) -> list[float]:
    return [float(threshold), _model_enc(model), _ctx_enc(max_context), _quant_enc(quant)]


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.array([
        encode(float(r.threshold), str(r.model), int(r.max_context), str(r.quantization))
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
# GP gradient (finite differences)
# ---------------------------------------------------------------------------

def gp_gradient(
    gp: GaussianProcessRegressor,
    scaler: StandardScaler,
    point: list[float],
    best_model: str,
    best_ctx: int,
    best_quant: str,
) -> dict[str, float]:
    def mu(x: list[float]) -> float:
        return float(gp.predict(scaler.transform([x]))[0])

    t, m_enc, ctx_enc, q_enc = point
    grad: dict[str, float] = {}

    t_hi = min(round(t + 0.05, 2), 0.95)
    t_lo = max(round(t - 0.05, 2), 0.05)
    if t_hi != t_lo:
        grad["threshold"] = (mu([t_hi, m_enc, ctx_enc, q_enc]) - mu([t_lo, m_enc, ctx_enc, q_enc])) / (t_hi - t_lo)
    else:
        grad["threshold"] = 0.0

    for m_label, m_e in MODEL_ENCODE.items():
        if m_label == Path(best_model).name:
            continue
        grad[f"model → {m_label}"] = mu([t, m_e, ctx_enc, q_enc]) - mu([t, m_enc, ctx_enc, q_enc])

    for ctx_val, ctx_e in MAX_CONTEXT_ENCODE.items():
        if ctx_val == best_ctx:
            continue
        grad[f"max_context → {ctx_val}"] = mu([t, m_enc, ctx_e, q_enc]) - mu([t, m_enc, ctx_enc, q_enc])

    for q_label, q_e in QUANT_ENCODE.items():
        if q_label == best_quant:
            continue
        grad[f"quantization → {q_label}"] = mu([t, m_enc, ctx_enc, q_e]) - mu([t, m_enc, ctx_enc, q_enc])

    return grad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest next CodeLlama hyperparameter configuration via Bayesian Optimisation."
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
        sys.exit(f"ERROR: {RUNS_CSV} not found — run codellama_runner.py first.")

    df = pd.read_csv(RUNS_CSV)

    if len(df) < 3:
        sys.exit("Need at least 3 observed runs to fit a GP.")

    if args.metric not in df.columns:
        sys.exit(f"ERROR: column '{args.metric}' not found in {RUNS_CSV}.")

    if "tn" in df.columns and "fn" in df.columns:
        degen = df[(df["tn"] == 0) | (df["fn"] == 0)]
        if not degen.empty:
            print(f"\nWARNING: {len(degen)} degenerate run(s) (TN=0 or FN=0):")
            for _, r in degen.iterrows():
                print(f"  {r['run_name']}  {args.metric.upper()}={r[args.metric]:.4f}  ← inflated")

    y = df[args.metric].to_numpy(dtype=float)
    X = feature_matrix(df)

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
    print(f"  model={best_row['model']}  ctx={best_row['max_context']}  "
          f"quant={best_row['quantization']}  threshold={best_row['threshold']:.2f}")
    print(f"{'='*60}")

    tried: set[tuple] = set(
        zip(
            df["threshold"].round(2),
            df["model"].apply(lambda m: Path(m).name),
            df["max_context"].astype(int),
            df["quantization"],
        )
    )

    candidates: list[tuple[float, str, int, str]] = []
    for t in THRESHOLD_GRID:
        for m_label in KNOWN_MODELS:
            for ctx in KNOWN_MAX_CONTEXTS:
                for q in KNOWN_QUANTS:
                    if (round(float(t), 2), m_label, ctx, q) not in tried:
                        candidates.append((float(t), m_label, ctx, q))

    if not candidates:
        print("\nAll grid points already tried.")
        return

    X_cand = np.array([encode(t, m, ctx, q) for t, m, ctx, q in candidates])
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
    print(f"  {'#':<3} {'thresh':>7} {'quant':<6} {'ctx':>5}  {'model':<34}  "
          f"{'EI':>9}  {'pred':>8}  {'±std':>7}  note")
    print("  " + "-" * 95)
    for rank, i in enumerate(selected_indices, 1):
        t, m, ctx, q = candidates[i]
        note = ""
        if mu_arr[i] > f_best:
            note = "exploit"
        elif sigma[i] > np.percentile(sigma, 75):
            note = "explore"
        print(f"  {rank:<3} {t:>7.2f} {q:<6} {ctx:>5}  {m:<34}  "
              f"{ei[i]:>9.5f}  {mu_arr[i]:>8.4f}  ±{sigma[i]:.4f}  {note}")

    if selected_indices:
        t, m, ctx, q = candidates[selected_indices[0]]
        m_full = MODEL_HF_ID.get(m, m)
        mlx_flag = "--mlx " if m == "mlx_codellama_4bit" else ""
        quant_arg = "" if m == "mlx_codellama_4bit" else f"--quantization {q} "
        print(f"\nTo run top suggestion:")
        print(f"  python codellama_runner.py {mlx_flag}--model {m_full} "
              f"--max-context {ctx} {quant_arg}--threshold {t:.2f}")

    best_point = encode(
        float(best_row["threshold"]),
        str(best_row["model"]),
        int(best_row["max_context"]),
        str(best_row["quantization"]),
    )
    grad = gp_gradient(
        gp, scaler, best_point,
        str(best_row["model"]), int(best_row["max_context"]), str(best_row["quantization"]),
    )

    print(f"\nGP gradient at current best  (how {args.metric.upper()} changes per unit):")
    for param, g in grad.items():
        if "→" in param:
            arrow = "better" if g > 0 else "worse"
            print(f"  {param:<50}  {g:+.4f}  ({arrow})")
        else:
            direction = "↑ increase" if g > 0 else "↓ decrease"
            print(f"  ∂{args.metric}/∂{param:<44}  {g:+.4f}  → {direction}")

    print(f"\nObserved landscape summary ({args.metric.upper()}):")
    for col in ["model", "max_context", "quantization"]:
        if col in df.columns:
            summary = (
                df.groupby(col)[args.metric]
                .agg(["count", "max", "mean"])
                .rename(columns={"count": "runs", "max": "best", "mean": "avg"})
            )
            print(f"\nBy {col}:")
            print(summary.to_string())


if __name__ == "__main__":
    main()
