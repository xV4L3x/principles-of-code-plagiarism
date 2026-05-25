#!/usr/bin/env python3
"""
analyze.py — Multi-tool SCPD results analyzer.

Reads the standard CSV produced by each tool runner and produces:

  out/tables/
    global_metrics.csv      — precision, recall, F1, accuracy, AUC per tool
    f1_by_level.csv         — F1 per plagiarism level × tool
    f1_by_case.csv          — F1 per exercise case × tool

  out/figures/
    01_similarity_distributions.png  — box plots per tool, grouped by level
    02_roc_curves.png                — ROC curves for all tools
    03_metrics_bar.png               — precision / recall / F1 / accuracy bars
    04_f1_by_level_heatmap.png       — heatmap: tools × levels
    05_f1_by_case_heatmap.png        — heatmap: tools × cases
    06_similarity_plag_vs_nonplag.png— violin: plag vs non-plag per tool

Usage:
  .venv/bin/python analyze.py
  .venv/bin/python analyze.py --threshold 0.5
  .venv/bin/python analyze.py --tools JPlag Dolos
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

TOOL_CSVS: dict[str, Path] = {
    "JPlag":   HERE / "../jplag/out/jplag_results.csv",
    "Dolos":   HERE / "../dolos/out/dolos_results.csv",
    "SIM":     HERE / "../sim/out/sim_results.csv",
    "Plaggie": HERE / "../plaggie/out/plaggie_results.csv",
}

TOOL_COLORS: dict[str, str] = {
    "JPlag":   "#4C72B0",
    "Dolos":   "#DD8452",
    "SIM":     "#55A868",
    "Plaggie": "#C44E52",
}

LEVEL_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "non-plag"]
CASE_ORDER  = [f"case-{i:02d}" for i in range(1, 8)]
OUT_DIR     = HERE / "out"
DPI         = 150
FIGSIZE_WIDE = (14, 5)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(tool_csvs: dict[str, Path], requested: list[str] | None) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for name, path in tool_csvs.items():
        if requested and name not in requested:
            continue
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {name}", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        df["is_plagiarized"] = df["is_plagiarized"].astype(str).str.lower() == "true"
        df["predicted_plag"] = df["predicted_plag"].astype(str).str.lower() == "true"
        df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)
        data[name] = df
    return data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the threshold in [0,1] that maximises F1 (200-point grid)."""
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.0, 1.0, 201):
        pred = y_score >= t
        tp = float(np.sum(pred & y_true))
        fp = float(np.sum(pred & ~y_true))
        fn = float(np.sum(~pred & y_true))
        p  = _safe_div(tp, tp + fp)
        r  = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def metrics_at(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    pred = y_score >= threshold
    tp = float(np.sum(pred & y_true))
    fp = float(np.sum(pred & ~y_true))
    tn = float(np.sum(~pred & ~y_true))
    fn = float(np.sum(~pred & y_true))
    p  = _safe_div(tp, tp + fp)
    r  = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r)
    acc = _safe_div(tp + tn, tp + fp + tn + fn)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    return dict(threshold=threshold, tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                precision=p, recall=r, f1=f1, accuracy=acc, auc=auc)


def compute_global_metrics(data: dict[str, pd.DataFrame],
                            fixed_threshold: float | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tool, df in data.items():
        y_true  = df["is_plagiarized"].to_numpy()
        y_score = df["similarity"].to_numpy()
        t = fixed_threshold if fixed_threshold is not None else find_optimal_threshold(y_true, y_score)
        out[tool] = metrics_at(y_true, y_score, t)
    return out


def compute_f1_by_group(data: dict[str, pd.DataFrame],
                         global_metrics: dict[str, dict],
                         group_col: str,
                         group_order: list[str]) -> pd.DataFrame:
    """Return DataFrame[group × tool] of F1 scores."""
    rows: dict[str, dict[str, float]] = {g: {} for g in group_order}
    for tool, df in data.items():
        t = global_metrics[tool]["threshold"]
        for g in group_order:
            sub = df[df[group_col] == g]
            if sub.empty:
                rows[g][tool] = float("nan")
                continue
            y_true  = sub["is_plagiarized"].to_numpy()
            y_score = sub["similarity"].to_numpy()
            rows[g][tool] = metrics_at(y_true, y_score, t)["f1"]
    return pd.DataFrame(rows, index=list(data.keys())).T   # shape: groups × tools


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str, fig_dir: Path) -> None:
    path = fig_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.relative_to(HERE)}")


def _tool_palette(tools: list[str]) -> list[str]:
    return [TOOL_COLORS.get(t, "#888888") for t in tools]


# ---------------------------------------------------------------------------
# Figure 1 — similarity distributions by level (2×2 grid, one panel per tool)
# ---------------------------------------------------------------------------

def fig_similarity_distributions(
    data: dict[str, pd.DataFrame],
    global_metrics: dict[str, dict],
    fig_dir: Path,
) -> None:
    tools = list(data.keys())
    n = len(tools)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), sharey=False)
    axes_flat = np.array(axes).flatten()

    plag_palette = {True: "#E07B7B", False: "#7BA8E0"}

    for ax, tool in zip(axes_flat, tools):
        df = data[tool]
        t  = global_metrics[tool]["threshold"]
        f1 = global_metrics[tool]["f1"]
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
        ax.set_title(f"{tool}  (F1 = {f1:.3f} @ t={t:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Plagiarism level")
        ax.set_ylabel("Similarity score")
        ax.set_ylim(-0.05, 1.05)
        handles, labels = ax.get_legend_handles_labels()
        # rename legend entries
        label_map = {"True": "Plagiarised", "False": "Non-plag", "is_plagiarized": ""}
        new_labels = [label_map.get(l, l) for l in labels]
        ax.legend(handles, new_labels, title="", fontsize=8, loc="upper right")

    # hide unused panels
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

    for tool, df in data.items():
        y_true  = df["is_plagiarized"].to_numpy()
        y_score = df["similarity"].to_numpy()
        auc = global_metrics[tool]["auc"]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, color=TOOL_COLORS.get(tool), linewidth=2,
                label=f"{tool}  (AUC = {auc:.3f})")

        # mark operating point (optimal threshold)
        t  = global_metrics[tool]["threshold"]
        op_fpr = global_metrics[tool]["fp"] / (global_metrics[tool]["fp"] + global_metrics[tool]["tn"]) if (global_metrics[tool]["fp"] + global_metrics[tool]["tn"]) else 0
        op_tpr = global_metrics[tool]["recall"]
        ax.scatter([op_fpr], [op_tpr], color=TOOL_COLORS.get(tool), s=60, zorder=5)

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
    tools   = list(global_metrics.keys())
    metric_keys = ["precision", "recall", "f1", "accuracy"]
    metric_labels = ["Precision", "Recall", "F1", "Accuracy"]

    x      = np.arange(len(tools))
    n_met  = len(metric_keys)
    width  = 0.18
    offsets = np.linspace(-(n_met - 1) / 2 * width, (n_met - 1) / 2 * width, n_met)

    fig, ax = plt.subplots(figsize=(10, 5))

    metric_colors = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F"]
    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, metric_colors)):
        vals = [global_metrics[t][key] for t in tools]
        bars = ax.bar(x + offsets[i], vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tools, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Global performance metrics at optimal threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # annotate optimal thresholds below x-axis
    for xi, tool in zip(x, tools):
        t = global_metrics[tool]["threshold"]
        ax.text(xi, -0.08, f"t*={t:.2f}", ha="center", va="top",
                fontsize=8, color="grey", transform=ax.get_xaxis_transform())

    fig.tight_layout()
    _save(fig, "03_metrics_bar.png", fig_dir)


# ---------------------------------------------------------------------------
# Figure 4 — F1 by plagiarism level (heatmap)
# ---------------------------------------------------------------------------

def fig_f1_by_level_heatmap(
    f1_level: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(
        f1_level.T,
        annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "F1 score"},
    )
    ax.set_xlabel("Plagiarism level", fontsize=11)
    ax.set_ylabel("Tool", fontsize=11)
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
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(
        f1_case.T,
        annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "F1 score"},
    )
    ax.set_xlabel("Exercise case", fontsize=11)
    ax.set_ylabel("Tool", fontsize=11)
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
    tools = list(data.keys())
    frames = []
    for tool, df in data.items():
        tmp = df[["similarity", "is_plagiarized"]].copy()
        tmp["tool"] = tool
        frames.append(tmp)
    combined = pd.concat(frames, ignore_index=True)
    combined["Class"] = combined["is_plagiarized"].map({True: "Plagiarised", False: "Non-plagiarised"})

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.violinplot(
        data=combined,
        x="tool", y="similarity", hue="Class",
        split=True,
        inner="quartile",
        palette={"Plagiarised": "#E07B7B", "Non-plagiarised": "#7BA8E0"},
        order=tools,
        cut=0,
        ax=ax,
    )
    # draw optimal threshold for each tool
    for xi, tool in enumerate(tools):
        t = global_metrics[tool]["threshold"]
        ax.hlines(t, xi - 0.45, xi + 0.45, colors="black",
                  linestyles="dashed", linewidths=1.0, alpha=0.8)

    ax.set_xlabel("Tool", fontsize=11)
    ax.set_ylabel("Similarity score", fontsize=11)
    ax.set_title("Similarity score distributions: plagiarised vs non-plagiarised", fontsize=13, fontweight="bold")
    ax.legend(title="", fontsize=9, loc="upper right")
    ax.set_ylim(-0.05, 1.05)
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
    # Global metrics
    rows = []
    for tool, m in global_metrics.items():
        rows.append({
            "tool":      tool,
            "threshold": round(m["threshold"], 3),
            "TP": m["tp"], "FP": m["fp"], "TN": m["tn"], "FN": m["fn"],
            "precision": round(m["precision"], 4),
            "recall":    round(m["recall"],    4),
            "f1":        round(m["f1"],        4),
            "accuracy":  round(m["accuracy"],  4),
            "auc":       round(m["auc"],       4),
        })
    gm_df = pd.DataFrame(rows).set_index("tool")
    gm_df.to_csv(tbl_dir / "global_metrics.csv")
    print(f"  Saved {(tbl_dir / 'global_metrics.csv').relative_to(HERE)}")

    # F1 by level
    f1_level.round(4).to_csv(tbl_dir / "f1_by_level.csv")
    print(f"  Saved {(tbl_dir / 'f1_by_level.csv').relative_to(HERE)}")

    # F1 by case
    f1_case.round(4).to_csv(tbl_dir / "f1_by_case.csv")
    print(f"  Saved {(tbl_dir / 'f1_by_case.csv').relative_to(HERE)}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(global_metrics: dict[str, dict],
                  f1_level: pd.DataFrame,
                  f1_case: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("GLOBAL METRICS (at optimal threshold per tool)")
    print("=" * 70)
    header = f"{'Tool':<10} {'Thresh':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Acc':>7} {'AUC':>7}"
    print(header)
    print("-" * 70)
    for tool, m in global_metrics.items():
        print(f"{tool:<10} {m['threshold']:>7.3f} {m['precision']:>7.3f} "
              f"{m['recall']:>7.3f} {m['f1']:>7.3f} {m['accuracy']:>7.3f} {m['auc']:>7.3f}")

    print("\n" + "=" * 70)
    print("F1 BY PLAGIARISM LEVEL (at tool-specific optimal threshold)")
    print("=" * 70)
    print(f"{'Level':<10}" + "".join(f"{t:>10}" for t in f1_level.columns))
    print("-" * 70)
    for level, row in f1_level.iterrows():
        vals = "".join(f"{v:>10.3f}" for v in row)
        print(f"{level:<10}{vals}")

    print("\n" + "=" * 70)
    print("F1 BY CASE (at tool-specific optimal threshold)")
    print("=" * 70)
    print(f"{'Case':<12}" + "".join(f"{t:>10}" for t in f1_case.columns))
    print("-" * 70)
    for case, row in f1_case.iterrows():
        vals = "".join(f"{v:>10.3f}" for v in row)
        print(f"{case:<12}{vals}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse and compare SCPD tool results from IR-Plag-Dataset."
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Fixed decision threshold (default: auto per tool via F1 optimisation)",
    )
    parser.add_argument(
        "--tools", nargs="+", default=None, metavar="TOOL",
        help="Subset of tools to include, e.g. --tools JPlag Dolos",
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
    data = load_data(TOOL_CSVS, args.tools)
    if not data:
        sys.exit("ERROR: no tool data found.")
    tools = list(data.keys())
    print(f"  Tools: {tools}")
    for t, df in data.items():
        n_plag = int(df["is_plagiarized"].sum())
        print(f"  {t}: {len(df)} rows ({n_plag} plagiarised, {len(df)-n_plag} non-plag)")

    # ---- metrics ----
    print("\nComputing metrics...")
    global_metrics = compute_global_metrics(data, args.threshold)
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
