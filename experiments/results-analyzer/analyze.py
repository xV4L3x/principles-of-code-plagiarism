#!/usr/bin/env python3
"""
analyze.py — Multi-tool SCPD results analyzer.

Reads each tool's runs CSV (produced by its runner) and produces:

  out/tables/
    global_metrics.csv      — precision, recall, F1, accuracy, AUC per run
    f1_by_level.csv         — F1 per plagiarism level × run
    f1_by_case.csv          — F1 per exercise case × run

  out/figures/
    01_similarity_distributions.png  — box plots per run, grouped by level
    02_roc_curves.png                — ROC curves for all runs
    03_metrics_bar.png               — precision / recall / F1 / accuracy bars
    04_f1_by_level_heatmap.png       — heatmap: runs × levels
    05_f1_by_case_heatmap.png        — heatmap: runs × cases
    06_similarity_plag_vs_nonplag.png— violin: plag vs non-plag per run

Each tool's runner appends one row per parameter configuration to its own
runs CSV.  The analyzer expands every row into a named "run" and uses the
threshold stored in the runs CSV — it never re-optimises thresholds itself.

Usage:
  .venv/bin/python analyze.py
  .venv/bin/python analyze.py --tools JPlag
  .venv/bin/python analyze.py --tools "JPlag-Threshold-0.50-MinTokens-5"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent

# Mapping from base tool name to its runs CSV.
# Each row in the runs CSV represents one parameter configuration.
TOOL_RUNS_CSVS: dict[str, Path] = {
    "JPlag":         HERE / "../jplag/out/jplag_runs.csv",
    "Dolos":         HERE / "../dolos/out/dolos_runs.csv",
    "SIM":           HERE / "../sim/out/sim_runs.csv",
    "Plaggie":       HERE / "../plaggie/out/plaggie_runs.csv",
    "Oreo":          HERE / "../oreo/out/oreo_runs.csv",
    "CodeBERT":      HERE / "../codebert/out/codebert_runs.csv",
    "CodeLlama":     HERE / "../codellama/out/codellama_runs.csv",
}

# Color palette: keyed by base tool name prefix so all runs of the same tool
# share the same colour family.  Unknown prefixes fall back to grey.
TOOL_BASE_COLORS: dict[str, str] = {
    "JPlag":         "#4C72B0",
    "Dolos":         "#DD8452",
    "SIM":           "#55A868",
    "Plaggie":       "#C44E52",
    "Oreo":          "#C45EA2",
    "CodeBERT":      "#8172B2",
    "GraphCodeBERT": "#937860",
    "CodeLlama":     "#26C6DA",
}

LEVEL_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "non-plag"]
CASE_ORDER  = [f"case-{i:02d}" for i in range(1, 8)]
OUT_DIR     = HERE / "out"
DPI         = 150


def _run_color(run_name: str) -> str:
    for base, color in TOOL_BASE_COLORS.items():
        if run_name.startswith(base):
            return color
    return "#888888"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    tool_runs_csvs: dict[str, Path],
    requested: list[str] | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """
    Load all runs from every tool's runs CSV.

    Returns:
      data             — {run_name: predictions_df}
      pre_thresholds   — {run_name: threshold}  (stored by the runner)

    If `requested` is given, only runs whose run_name OR base tool name
    appears in the list are included.
    """
    data: dict[str, pd.DataFrame] = {}
    pre_thresholds: dict[str, float] = {}

    for base_name, runs_path in tool_runs_csvs.items():
        if not runs_path.exists():
            print(f"  INFO: {runs_path} not found — skipping {base_name}", file=sys.stderr)
            continue

        runs = pd.read_csv(runs_path)
        if runs.empty:
            continue

        for _, run_row in runs.iterrows():
            run_name = str(run_row["run_name"])

            # Filter: accept if run_name or base_name matches any requested token
            if requested:
                if run_name not in requested and base_name not in requested:
                    continue

            pred_filename = str(run_row["predictions_csv"])
            pred_path = runs_path.parent / pred_filename
            if not pred_path.exists():
                print(f"  WARNING: {pred_path} not found — skipping run '{run_name}'",
                      file=sys.stderr)
                continue

            df = pd.read_csv(pred_path)
            df["is_plagiarized"] = df["is_plagiarized"].astype(str).str.lower() == "true"
            df["predicted_plag"] = df["predicted_plag"].astype(str).str.lower() == "true"
            df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)
            data[run_name] = df
            pre_thresholds[run_name] = float(run_row["threshold"])

    return data, pre_thresholds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def metrics_at(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    pred = y_score >= threshold
    tp = float(np.sum(pred & y_true))
    fp = float(np.sum(pred & ~y_true))
    tn = float(np.sum(~pred & ~y_true))
    fn = float(np.sum(~pred & y_true))
    p   = _safe_div(tp, tp + fp)
    r   = _safe_div(tp, tp + fn)
    f1  = _safe_div(2 * p * r, p + r)
    acc = _safe_div(tp + tn, tp + fp + tn + fn)
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(tp * tn - fp * fn, mcc_denom)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    return dict(threshold=threshold, tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                precision=p, recall=r, f1=f1, accuracy=acc, auc=auc, mcc=mcc)


def compute_global_metrics(
    data: dict[str, pd.DataFrame],
    pre_thresholds: dict[str, float],
) -> dict[str, dict]:
    """Compute metrics for every run using the threshold stored by the runner."""
    out: dict[str, dict] = {}
    for run_name, df in data.items():
        y_true  = df["is_plagiarized"].to_numpy()
        y_score = df["similarity"].to_numpy()
        t = pre_thresholds[run_name]
        out[run_name] = metrics_at(y_true, y_score, t)
    return out


def compute_f1_by_group(
    data: dict[str, pd.DataFrame],
    global_metrics: dict[str, dict],
    group_col: str,
    group_order: list[str],
) -> pd.DataFrame:
    """Return DataFrame[group × run] of F1 scores."""
    rows: dict[str, dict[str, float]] = {g: {} for g in group_order}
    for run_name, df in data.items():
        t = global_metrics[run_name]["threshold"]
        for g in group_order:
            sub = df[df[group_col] == g]
            if sub.empty:
                rows[g][run_name] = float("nan")
                continue
            y_true  = sub["is_plagiarized"].to_numpy()
            y_score = sub["similarity"].to_numpy()
            rows[g][run_name] = metrics_at(y_true, y_score, t)["f1"]
    return pd.DataFrame(rows, index=list(data.keys())).T   # shape: groups × runs


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str, fig_dir: Path) -> None:
    path = fig_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.relative_to(HERE)}")


# ---------------------------------------------------------------------------
# Figure 1 — similarity distributions by level (2-column grid, one panel per run)
# ---------------------------------------------------------------------------

def fig_similarity_distributions(
    data: dict[str, pd.DataFrame],
    global_metrics: dict[str, dict],
    fig_dir: Path,
) -> None:
    runs = list(data.keys())
    n = len(runs)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), sharey=False)
    axes_flat = np.array(axes).flatten()

    plag_palette = {True: "#E07B7B", False: "#7BA8E0"}

    for ax, run_name in zip(axes_flat, runs):
        df = data[run_name]
        t  = global_metrics[run_name]["threshold"]
        f1 = global_metrics[run_name]["f1"]
        sns.boxplot(
            data=df,
            x="level", y="similarity",
            hue="is_plagiarized",
            order=LEVEL_ORDER,
            hue_order=[True, False],
            palette=plag_palette,
            linewidth=0.8,
            fliersize=2,
            ax=ax,
        )
        ax.axhline(t, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                   label=f"threshold = {t:.2f}")
        ax.set_title(f"{run_name}  (F1 = {f1:.3f} @ t={t:.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Plagiarism level")
        ax.set_ylabel("Similarity score")
        ax.set_ylim(-0.05, 1.05)
        handles, labels = ax.get_legend_handles_labels()
        label_map = {"True": "Plagiarised", "False": "Non-plag", "is_plagiarized": ""}
        new_labels = [label_map.get(l, l) for l in labels]
        ax.legend(handles, new_labels, title="", fontsize=8, loc="upper right")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Similarity score distributions by plagiarism level", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "01_similarity_distributions.png", fig_dir)


# ---------------------------------------------------------------------------
# Figure 2 — ROC curves
# ---------------------------------------------------------------------------

def fig_roc_curves(
    data: dict[str, pd.DataFrame],
    global_metrics: dict[str, dict],
    fig_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="Random (AUC = 0.50)")

    for run_name, df in data.items():
        y_true  = df["is_plagiarized"].to_numpy()
        y_score = df["similarity"].to_numpy()
        auc = global_metrics[run_name]["auc"]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, color=_run_color(run_name), linewidth=2,
                label=f"{run_name}  (AUC = {auc:.3f})")

        t      = global_metrics[run_name]["threshold"]
        op_fpr = (global_metrics[run_name]["fp"] /
                  (global_metrics[run_name]["fp"] + global_metrics[run_name]["tn"])
                  if (global_metrics[run_name]["fp"] + global_metrics[run_name]["tn"]) else 0)
        op_tpr = global_metrics[run_name]["recall"]
        ax.scatter([op_fpr], [op_tpr], color=_run_color(run_name), s=60, zorder=5)

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "02_roc_curves.png", fig_dir)


# ---------------------------------------------------------------------------
# Figure 3 — Precision / Recall / F1 / Accuracy bar chart
# ---------------------------------------------------------------------------

def fig_metrics_bar(
    global_metrics: dict[str, dict],
    fig_dir: Path,
) -> None:
    runs  = list(global_metrics.keys())
    metric_keys   = ["precision", "recall", "f1", "accuracy", "mcc"]
    metric_labels = ["Precision", "Recall", "F1", "Accuracy", "MCC"]

    x      = np.arange(len(runs))
    n_met  = len(metric_keys)
    width  = 0.15
    offsets = np.linspace(-(n_met - 1) / 2 * width, (n_met - 1) / 2 * width, n_met)

    fig, ax = plt.subplots(figsize=(max(10, len(runs) * 1.8), 5))

    metric_colors = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F", "#8172B2"]
    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, metric_colors)):
        vals = [global_metrics[r][key] for r in runs]
        bars = ax.bar(x + offsets[i], vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Global performance metrics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    _save(fig, "03_metrics_bar.png", fig_dir)


# ---------------------------------------------------------------------------
# Figure 4 — F1 by plagiarism level (heatmap)
# ---------------------------------------------------------------------------

def fig_f1_by_level_heatmap(
    f1_level: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, max(3.5, len(f1_level.columns) * 0.5)))
    sns.heatmap(
        f1_level.T,
        annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "F1 score"},
    )
    ax.set_xlabel("Plagiarism level", fontsize=11)
    ax.set_ylabel("Run", fontsize=11)
    ax.set_title("F1 score by plagiarism level", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, "04_f1_by_level_heatmap.png", fig_dir)


# ---------------------------------------------------------------------------
# Figure 5 — F1 by case (heatmap)
# ---------------------------------------------------------------------------

def fig_f1_by_case_heatmap(
    f1_case: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, max(3.5, len(f1_case.columns) * 0.5)))
    sns.heatmap(
        f1_case.T,
        annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "F1 score"},
    )
    ax.set_xlabel("Exercise case", fontsize=11)
    ax.set_ylabel("Run", fontsize=11)
    ax.set_title("F1 score by exercise case", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, "05_f1_by_case_heatmap.png", fig_dir)


# ---------------------------------------------------------------------------
# Figure 6 — Violin: plagiarised vs non-plagiarised similarity distributions
# ---------------------------------------------------------------------------

def fig_plag_vs_nonplag(
    data: dict[str, pd.DataFrame],
    global_metrics: dict[str, dict],
    fig_dir: Path,
) -> None:
    runs = list(data.keys())
    frames = []
    for run_name, df in data.items():
        tmp = df[["similarity", "is_plagiarized"]].copy()
        tmp["run"] = run_name
        frames.append(tmp)
    combined = pd.concat(frames, ignore_index=True)
    combined["Class"] = combined["is_plagiarized"].map({True: "Plagiarised", False: "Non-plagiarised"})

    fig, ax = plt.subplots(figsize=(max(11, len(runs) * 1.5), 5))
    sns.violinplot(
        data=combined,
        x="run", y="similarity", hue="Class",
        split=True,
        inner="quartile",
        palette={"Plagiarised": "#E07B7B", "Non-plagiarised": "#7BA8E0"},
        order=runs,
        cut=0,
        ax=ax,
    )
    for xi, run_name in enumerate(runs):
        t = global_metrics[run_name]["threshold"]
        ax.hlines(t, xi - 0.45, xi + 0.45, colors="black",
                  linestyles="dashed", linewidths=1.0, alpha=0.8)

    ax.set_xlabel("Run", fontsize=11)
    ax.set_ylabel("Similarity score", fontsize=11)
    ax.set_title("Similarity score distributions: plagiarised vs non-plagiarised",
                 fontsize=13, fontweight="bold")
    ax.legend(title="", fontsize=9, loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
    fig.tight_layout()
    _save(fig, "06_similarity_plag_vs_nonplag.png", fig_dir)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def save_tables(
    global_metrics: dict[str, dict],
    f1_level: pd.DataFrame,
    f1_case: pd.DataFrame,
    tbl_dir: Path,
) -> None:
    rows = []
    for run_name, m in global_metrics.items():
        rows.append({
            "run":       run_name,
            "TP": m["tp"], "FP": m["fp"], "TN": m["tn"], "FN": m["fn"],
            "precision": round(m["precision"], 4),
            "recall":    round(m["recall"],    4),
            "f1":        round(m["f1"],        4),
            "accuracy":  round(m["accuracy"],  4),
            "auc":       round(m["auc"],       4),
            "mcc":       round(m["mcc"],       4),
        })
    gm_df = pd.DataFrame(rows).set_index("run")
    gm_df.to_csv(tbl_dir / "global_metrics.csv")
    print(f"  Saved {(tbl_dir / 'global_metrics.csv').relative_to(HERE)}")

    f1_level.round(4).to_csv(tbl_dir / "f1_by_level.csv")
    print(f"  Saved {(tbl_dir / 'f1_by_level.csv').relative_to(HERE)}")

    f1_case.round(4).to_csv(tbl_dir / "f1_by_case.csv")
    print(f"  Saved {(tbl_dir / 'f1_by_case.csv').relative_to(HERE)}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(global_metrics: dict[str, dict],
                  f1_level: pd.DataFrame,
                  f1_case: pd.DataFrame) -> None:
    col_w = max(len(r) for r in global_metrics) + 2
    line = "=" * (col_w + 7 * 8)

    print("\n" + line)
    print("GLOBAL METRICS")
    print(line)
    header = f"{'Run':<{col_w}} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Acc':>7} {'AUC':>7} {'MCC':>7}"
    print(header)
    print("-" * (col_w + 7 * 7))
    for run_name, m in global_metrics.items():
        print(f"{run_name:<{col_w}} {m['precision']:>7.3f} "
              f"{m['recall']:>7.3f} {m['f1']:>7.3f} {m['accuracy']:>7.3f} "
              f"{m['auc']:>7.3f} {m['mcc']:>7.3f}")

    print("\n" + line)
    print("F1 BY PLAGIARISM LEVEL")
    print(line)
    runs = list(f1_level.columns)
    print(f"{'Level':<10}" + "".join(f"{r:>{col_w}}" for r in runs))
    print("-" * (10 + col_w * len(runs)))
    for level, row in f1_level.iterrows():
        vals = "".join(f"{v:>{col_w}.3f}" for v in row)
        print(f"{level:<10}{vals}")

    print("\n" + line)
    print("F1 BY CASE")
    print(line)
    print(f"{'Case':<12}" + "".join(f"{r:>{col_w}}" for r in runs))
    print("-" * (12 + col_w * len(runs)))
    for case, row in f1_case.iterrows():
        vals = "".join(f"{v:>{col_w}.3f}" for v in row)
        print(f"{case:<12}{vals}")
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse and compare SCPD tool results from IR-Plag-Dataset."
    )
    parser.add_argument(
        "--tools", nargs="+", default=None, metavar="TOOL",
        help="Subset of runs or base tool names to include, "
             "e.g. --tools JPlag  or  --tools 'JPlag-Threshold-0.50-MinTokens-5'",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUT_DIR,
        help="Root output directory (default: out/)",
    )
    args = parser.parse_args()

    fig_dir = args.output_dir / "figures"
    tbl_dir = args.output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # ---- load ----
    print("Loading data...")
    data, pre_thresholds = load_data(TOOL_RUNS_CSVS, args.tools)
    if not data:
        sys.exit("ERROR: no run data found. Make sure each tool's runs CSV exists.")
    runs = list(data.keys())
    print(f"  Runs loaded: {runs}")
    for run_name, df in data.items():
        n_plag = int(df["is_plagiarized"].sum())
        t = pre_thresholds[run_name]
        print(f"  {run_name}: {len(df)} rows ({n_plag} plagiarised, {len(df)-n_plag} non-plag)"
              f"  threshold={t:.2f}")

    # ---- metrics ----
    print("\nComputing metrics...")
    global_metrics = compute_global_metrics(data, pre_thresholds)
    f1_level = compute_f1_by_group(data, global_metrics, "level", LEVEL_ORDER)
    f1_case  = compute_f1_by_group(data, global_metrics, "case",  CASE_ORDER)

    # ---- figures ----
    print("\nGenerating figures...")
    fig_similarity_distributions(data, global_metrics, fig_dir)
    fig_roc_curves(data, global_metrics, fig_dir)
    fig_metrics_bar(global_metrics, fig_dir)
    fig_f1_by_level_heatmap(f1_level, fig_dir)
    fig_f1_by_case_heatmap(f1_case, fig_dir)
    fig_plag_vs_nonplag(data, global_metrics, fig_dir)

    # ---- tables ----
    print("\nSaving tables...")
    save_tables(global_metrics, f1_level, f1_case, tbl_dir)

    # ---- summary ----
    print_summary(global_metrics, f1_level, f1_case)

    print(f"\nDone.  Results in {args.output_dir.resolve()}/")


if __name__ == "__main__":
    main()
