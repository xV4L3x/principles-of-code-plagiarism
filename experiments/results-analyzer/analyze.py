#!/usr/bin/env python3
"""
analyze.py — Multi-tool SCPD results analyzer.

Default mode:
  Selects the best non-degenerate run per tool (by MCC; change with --metric)
  and produces clean 7-series visualizations plus summary tables.

All-runs mode (--all-runs):
  Loads every run from every tool's runs CSV. Useful for per-tool sweep
  debugging. Skips figures 05 and 07 (best-per-level views).

Outputs:
  out/tables/
    best_per_tool.csv     — best run per tool + all metrics  (default mode)
    best_per_level.csv    — winning tool per plagiarism level (default mode)
    f1_by_level.csv       — F1 per level × tool
    f1_by_case.csv        — F1 per case  × tool
    global_metrics.csv    — full metrics per run/tool

  out/figures/
    01_roc_curves.png
    02_metrics_bar.png
    03_f1_by_level_heatmap.png
    04_f1_by_case_heatmap.png
    05_f1_by_level_lines.png    (default mode only)
    06_similarity_violin.png
    07_best_per_level_bar.png   (default mode only)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent

TOOL_RUNS_CSVS: dict[str, Path] = {
    "JPlag":     HERE / "../jplag/out/jplag_runs.csv",
    "Dolos":     HERE / "../dolos/out/dolos_runs.csv",
    "SIM":       HERE / "../sim/out/sim_runs.csv",
    "Plaggie":   HERE / "../plaggie/out/plaggie_runs.csv",
    "Oreo":      HERE / "../oreo/out/oreo_runs.csv",
    "CodeBERT":  HERE / "../codebert/out/codebert_runs.csv",
    "CodeLlama": HERE / "../codellama/out/codellama_runs.csv",
}

TOOL_COLORS: dict[str, str] = {
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
PLAG_LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6"]
CASE_ORDER  = [f"case-{i:02d}" for i in range(1, 8)]
OUT_DIR     = HERE / "out"
DPI         = 150


def _color(name: str) -> str:
    if name in TOOL_COLORS:
        return TOOL_COLORS[name]
    for base, color in TOOL_COLORS.items():
        if name.startswith(base):
            return color
    return "#888888"


# ---------------------------------------------------------------------------
# Best-run selection
# ---------------------------------------------------------------------------

def select_best_runs(
    tool_runs_csvs: dict[str, Path],
    metric: str,
    requested: list[str] | None,
) -> dict[str, pd.Series]:
    """Return {base_tool_name: best_non_degenerate_run_row}."""
    best: dict[str, pd.Series] = {}
    for base_name, runs_path in tool_runs_csvs.items():
        if requested and base_name not in requested:
            continue
        if not runs_path.exists():
            print(f"  INFO: {runs_path} not found — skipping {base_name}", file=sys.stderr)
            continue
        runs = pd.read_csv(runs_path)
        if runs.empty:
            continue
        degen = (runs["tn"] == 0) | (runs["tp"] == 0) | (runs["fp"] == 0)
        n_degen = int(degen.sum())
        valid = runs[~degen]
        if n_degen:
            print(f"  {base_name}: pruned {n_degen} degenerate run(s) (TN=0, TP=0, or FP=0)")
        if valid.empty:
            print(f"  WARNING: {base_name} — no non-degenerate runs, skipping", file=sys.stderr)
            continue
        if metric not in valid.columns:
            print(f"  WARNING: metric '{metric}' not found in {base_name} runs", file=sys.stderr)
            continue
        best_row = valid.loc[valid[metric].idxmax()]
        best[base_name] = best_row
        print(f"  {base_name}: best → {best_row['run_name']}  "
              f"{metric.upper()}={float(best_row[metric]):.4f}")
    return best


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["is_plagiarized"] = df["is_plagiarized"].astype(str).str.lower() == "true"
    df["predicted_plag"] = df["predicted_plag"].astype(str).str.lower() == "true"
    df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)
    return df


def load_best_predictions(
    best_runs: dict[str, pd.Series],
    tool_runs_csvs: dict[str, Path],
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    data: dict[str, pd.DataFrame] = {}
    thresholds: dict[str, float] = {}
    for base_name, row in best_runs.items():
        pred_path = tool_runs_csvs[base_name].parent / str(row["predictions_csv"])
        if not pred_path.exists():
            print(f"  WARNING: {pred_path} not found — skipping {base_name}", file=sys.stderr)
            continue
        data[base_name] = _load_pred(pred_path)
        thresholds[base_name] = float(row["threshold"])
    return data, thresholds


def load_all_predictions(
    tool_runs_csvs: dict[str, Path],
    requested: list[str] | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    data: dict[str, pd.DataFrame] = {}
    thresholds: dict[str, float] = {}
    for base_name, runs_path in tool_runs_csvs.items():
        if not runs_path.exists():
            print(f"  INFO: {runs_path} not found — skipping {base_name}", file=sys.stderr)
            continue
        runs = pd.read_csv(runs_path)
        degen = (runs["tn"] == 0) | (runs["tp"] == 0) | (runs["fp"] == 0)
        n_degen = int(degen.sum())
        if n_degen:
            print(f"  {base_name}: pruned {n_degen} degenerate run(s) (TN=0, TP=0, or FP=0)")
        for _, row in runs[~degen].iterrows():
            run_name = str(row["run_name"])
            if requested and run_name not in requested and base_name not in requested:
                continue
            pred_path = runs_path.parent / str(row["predictions_csv"])
            if not pred_path.exists():
                print(f"  WARNING: {pred_path} not found — skipping '{run_name}'", file=sys.stderr)
                continue
            data[run_name] = _load_pred(pred_path)
            thresholds[run_name] = float(row["threshold"])
    return data, thresholds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


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
    mcc_d = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc   = _safe_div(tp * tn - fp * fn, mcc_d)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    return dict(threshold=threshold,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                precision=p, recall=r, f1=f1, accuracy=acc, auc=auc, mcc=mcc)


def compute_global_metrics(
    data: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
) -> dict[str, dict]:
    return {
        name: metrics_at(
            df["is_plagiarized"].to_numpy(),
            df["similarity"].to_numpy(),
            thresholds[name],
        )
        for name, df in data.items()
    }


def compute_group_metrics(
    data: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
    group_col: str,
    group_order: list[str],
) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for name, df in data.items():
        t = thresholds[name]
        result[name] = {}
        for g in group_order:
            sub = df[df[group_col] == g]
            if sub.empty:
                result[name][g] = dict(f1=float("nan"), precision=float("nan"),
                                       recall=float("nan"), threshold=t,
                                       tp=0, fp=0, tn=0, fn=0,
                                       accuracy=float("nan"), auc=float("nan"),
                                       mcc=float("nan"))
            else:
                result[name][g] = metrics_at(
                    sub["is_plagiarized"].to_numpy(),
                    sub["similarity"].to_numpy(), t)
    return result


def f1_matrix(
    group_metrics: dict[str, dict[str, dict]],
    group_order: list[str],
) -> pd.DataFrame:
    """DataFrame with index=groups, columns=tool_names, values=F1."""
    return pd.DataFrame(
        index=group_order,
        data={
            name: [group_metrics[name].get(g, {}).get("f1", float("nan"))
                   for g in group_order]
            for name in group_metrics
        },
    )


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str, fig_dir: Path, fsuffix: str = "") -> None:
    stem, ext = name.rsplit(".", 1)
    path = fig_dir / f"{stem}{fsuffix}.{ext}"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.relative_to(HERE)}")


# ---------------------------------------------------------------------------
# Figure 01 — ROC curves
# ---------------------------------------------------------------------------

def fig_roc_curves(
    data: dict[str, pd.DataFrame],
    global_metrics: dict[str, dict],
    fig_dir: Path,
    suffix: str,
    fsuffix: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="Random (AUC = 0.50)")
    for name, df in data.items():
        y_true  = df["is_plagiarized"].to_numpy()
        y_score = df["similarity"].to_numpy()
        gm      = global_metrics[name]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, color=_color(name), linewidth=2.0,
                label=f"{name}  (AUC = {gm['auc']:.3f})")
        d = gm["fp"] + gm["tn"]
        ax.scatter([gm["fp"] / d if d else 0], [gm["recall"]],
                   color=_color(name), s=60, zorder=5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves" + suffix, fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "01_roc_curves.png", fig_dir, fsuffix)


# ---------------------------------------------------------------------------
# Figure 02 — Metrics bar chart
# ---------------------------------------------------------------------------

def fig_metrics_bar(
    global_metrics: dict[str, dict],
    fig_dir: Path,
    suffix: str,
    fsuffix: str = "",
) -> None:
    names         = list(global_metrics.keys())
    metric_keys   = ["precision", "recall", "f1", "accuracy", "mcc"]
    metric_labels = ["Precision", "Recall", "F1", "Accuracy", "MCC"]
    metric_colors = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F", "#8172B2"]
    n_met   = len(metric_keys)
    width   = 0.15
    x       = np.arange(len(names))
    offsets = np.linspace(-(n_met - 1) / 2 * width, (n_met - 1) / 2 * width, n_met)

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.8), 5))
    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, metric_colors)):
        vals = [global_metrics[n][key] for n in names]
        bars = ax.bar(x + offsets[i], vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Global performance metrics" + suffix, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    _save(fig, "02_metrics_bar.png", fig_dir, fsuffix)


# ---------------------------------------------------------------------------
# Figures 03 & 04 — F1 heatmaps
# ---------------------------------------------------------------------------

def fig_f1_by_level_heatmap(
    f1_lv: pd.DataFrame, fig_dir: Path, suffix: str, fsuffix: str = ""
) -> None:
    mat = f1_lv.reindex(LEVEL_ORDER)
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(mat.columns) * 0.55)))
    sns.heatmap(mat.T, annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "F1 score"})
    ax.set_xlabel("Plagiarism level", fontsize=11)
    ax.set_ylabel("Tool / Run", fontsize=11)
    ax.set_title("F1 by plagiarism level" + suffix, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, "03_f1_by_level_heatmap.png", fig_dir, fsuffix)


def fig_f1_by_case_heatmap(
    f1_case: pd.DataFrame, fig_dir: Path, suffix: str, fsuffix: str = ""
) -> None:
    mat = f1_case.reindex(CASE_ORDER)
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(mat.columns) * 0.55)))
    sns.heatmap(mat.T, annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "F1 score"})
    ax.set_xlabel("Exercise case", fontsize=11)
    ax.set_ylabel("Tool / Run", fontsize=11)
    ax.set_title("F1 by exercise case" + suffix, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, "04_f1_by_case_heatmap.png", fig_dir, fsuffix)


# ---------------------------------------------------------------------------
# Figure 05 — F1 vs obfuscation level (line chart, default mode only)
# ---------------------------------------------------------------------------

def fig_f1_by_level_lines(
    lv_metrics: dict[str, dict[str, dict]],
    fig_dir: Path,
    fsuffix: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, lm in lv_metrics.items():
        f1_vals = [lm.get(lv, {}).get("f1", float("nan")) for lv in PLAG_LEVELS]
        ax.plot(PLAG_LEVELS, f1_vals,
                marker="o", linewidth=2.0, markersize=6,
                color=_color(name), label=name)
    ax.set_xlabel("Plagiarism level  (obfuscation intensity →)", fontsize=11)
    ax.set_ylabel("F1 score", fontsize=11)
    ax.set_title("F1 vs obfuscation level (best run per tool)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "05_f1_by_level_lines.png", fig_dir, fsuffix)


# ---------------------------------------------------------------------------
# Figure 06 — Violin: plagiarised vs non-plagiarised
# ---------------------------------------------------------------------------

def fig_similarity_violin(
    data: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
    fig_dir: Path,
    suffix: str,
    fsuffix: str = "",
) -> None:
    names  = list(data.keys())
    frames = [df[["similarity", "is_plagiarized"]].assign(tool=name)
              for name, df in data.items()]
    combined = pd.concat(frames, ignore_index=True)
    combined["Class"] = combined["is_plagiarized"].map(
        {True: "Plagiarised", False: "Non-plagiarised"})

    fig, ax = plt.subplots(figsize=(max(11, len(names) * 1.5), 5))
    sns.violinplot(
        data=combined, x="tool", y="similarity", hue="Class",
        split=True, inner="quartile",
        palette={"Plagiarised": "#E07B7B", "Non-plagiarised": "#7BA8E0"},
        order=names, cut=0, ax=ax,
    )
    for xi, name in enumerate(names):
        t = thresholds[name]
        ax.hlines(t, xi - 0.45, xi + 0.45,
                  colors="black", linestyles="dashed", linewidths=1.0, alpha=0.8)
    ax.set_xlabel("Tool", fontsize=11)
    ax.set_ylabel("Similarity score", fontsize=11)
    ax.set_title("Similarity distribution: plagiarised vs non-plagiarised" + suffix,
                 fontsize=13, fontweight="bold")
    ax.legend(title="", fontsize=9, loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    _save(fig, "06_similarity_violin.png", fig_dir, fsuffix)


# ---------------------------------------------------------------------------
# Figure 07 — Best tool per level bar (default mode only)
# ---------------------------------------------------------------------------

def fig_best_per_level_bar(
    lv_metrics: dict[str, dict[str, dict]],
    fig_dir: Path,
    fsuffix: str = "",
) -> None:
    best_f1, best_tool, bar_colors = [], [], []
    for level in PLAG_LEVELS:
        scores = {name: lm.get(level, {}).get("f1", float("nan"))
                  for name, lm in lv_metrics.items()}
        valid = {n: v for n, v in scores.items() if not np.isnan(v)}
        if not valid:
            best_f1.append(0.0); best_tool.append("n/a"); bar_colors.append("#888888")
        else:
            winner = max(valid, key=valid.__getitem__)
            best_f1.append(valid[winner])
            best_tool.append(winner)
            bar_colors.append(_color(winner))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(PLAG_LEVELS, best_f1, color=bar_colors,
                  alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, tname, fval in zip(bars, best_tool, best_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{tname}\n{fval:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    seen = list(dict.fromkeys(t for t in best_tool if t != "n/a"))
    ax.legend(handles=[Patch(color=_color(t), label=t) for t in seen],
              fontsize=9, loc="lower left")
    ax.set_xlabel("Plagiarism level", fontsize=11)
    ax.set_ylabel("Best F1 score", fontsize=11)
    ax.set_title("Best F1 per obfuscation level", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "07_best_per_level_bar.png", fig_dir, fsuffix)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def save_tables(
    best_runs: dict[str, pd.Series] | None,
    global_metrics: dict[str, dict],
    lv_metrics: dict[str, dict[str, dict]],
    f1_lv: pd.DataFrame,
    f1_case: pd.DataFrame,
    tbl_dir: Path,
    fsuffix: str = "",
) -> None:
    def _csv(stem: str) -> Path:
        return tbl_dir / f"{stem}{fsuffix}.csv"

    if best_runs is not None:
        # best_per_tool.csv
        rows = []
        for base_name, row in best_runs.items():
            if base_name not in global_metrics:
                continue
            gm = global_metrics[base_name]
            rows.append({
                "tool":      base_name,
                "run_name":  row["run_name"],
                "threshold": round(float(row["threshold"]), 4),
                "TP": gm["tp"], "FP": gm["fp"], "TN": gm["tn"], "FN": gm["fn"],
                "precision": round(gm["precision"], 4),
                "recall":    round(gm["recall"],    4),
                "f1":        round(gm["f1"],        4),
                "accuracy":  round(gm["accuracy"],  4),
                "auc":       round(gm["auc"],       4),
                "mcc":       round(gm["mcc"],       4),
            })
        pd.DataFrame(rows).to_csv(_csv("best_per_tool"), index=False)
        print(f"  Saved {_csv('best_per_tool').relative_to(HERE)}")

        # best_per_level.csv
        level_rows = []
        for level in PLAG_LEVELS:
            scores = {name: lm.get(level, {}).get("f1", float("nan"))
                      for name, lm in lv_metrics.items()}
            valid = {n: v for n, v in scores.items() if not np.isnan(v)}
            if not valid:
                continue
            winner = max(valid, key=valid.__getitem__)
            wm = lv_metrics[winner].get(level, {})
            level_rows.append({
                "level":     level,
                "best_tool": winner,
                "run_name":  str(best_runs[winner]["run_name"]) if winner in best_runs else "",
                "f1":        round(float(wm.get("f1", float("nan"))),        4),
                "precision": round(float(wm.get("precision", float("nan"))), 4),
                "recall":    round(float(wm.get("recall", float("nan"))),    4),
            })
        pd.DataFrame(level_rows).to_csv(_csv("best_per_level"), index=False)
        print(f"  Saved {_csv('best_per_level').relative_to(HERE)}")

    # global_metrics.csv
    gm_rows = [{
        "run": name,
        "TP": m["tp"], "FP": m["fp"], "TN": m["tn"], "FN": m["fn"],
        "precision": round(m["precision"], 4),
        "recall":    round(m["recall"],    4),
        "f1":        round(m["f1"],        4),
        "accuracy":  round(m["accuracy"],  4),
        "auc":       round(m["auc"],       4),
        "mcc":       round(m["mcc"],       4),
    } for name, m in global_metrics.items()]
    pd.DataFrame(gm_rows).set_index("run").to_csv(_csv("global_metrics"))
    print(f"  Saved {_csv('global_metrics').relative_to(HERE)}")

    f1_lv.round(4).to_csv(_csv("f1_by_level"))
    print(f"  Saved {_csv('f1_by_level').relative_to(HERE)}")
    f1_case.round(4).to_csv(_csv("f1_by_case"))
    print(f"  Saved {_csv('f1_by_case').relative_to(HERE)}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(
    global_metrics: dict[str, dict],
    f1_lv: pd.DataFrame,
    f1_case: pd.DataFrame,
) -> None:
    names = list(global_metrics.keys())
    col_w = max(len(n) for n in names) + 2
    line  = "=" * (col_w + 56)

    print("\n" + line)
    print("GLOBAL METRICS")
    print(line)
    hdr = format("Run/Tool", f"<{col_w}") + " Prec    Rec     F1    Acc    AUC    MCC"
    print(hdr)
    print("-" * len(hdr))
    for name, m in global_metrics.items():
        print(format(name, f"<{col_w}")
              + format(m["precision"], " >7.3f")
              + format(m["recall"],    " >7.3f")
              + format(m["f1"],        " >7.3f")
              + format(m["accuracy"],  " >7.3f")
              + format(m["auc"],       " >7.3f")
              + format(m["mcc"],       " >7.3f"))

    def _row_vals(df: pd.DataFrame, idx: str) -> str:
        if idx not in df.index:
            return ""
        parts = []
        for n in names:
            v = df.at[idx, n] if n in df.columns else float("nan")
            parts.append(format(v, f">{col_w}.3f") if not np.isnan(float(v))
                         else format("n/a", f">{col_w}"))
        return "".join(parts)

    print("\n" + line)
    print("F1 BY PLAGIARISM LEVEL")
    print(line)
    print(format("Level", "<10") + "".join(format(n, f">{col_w}") for n in names))
    print("-" * (10 + col_w * len(names)))
    for lv in LEVEL_ORDER:
        vals = _row_vals(f1_lv, lv)
        if vals:
            print(format(lv, "<10") + vals)

    print("\n" + line)
    print("F1 BY CASE")
    print(line)
    print(format("Case", "<12") + "".join(format(n, f">{col_w}") for n in names))
    print("-" * (12 + col_w * len(names)))
    for case in CASE_ORDER:
        vals = _row_vals(f1_case, case)
        if vals:
            print(format(case, "<12") + vals)
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse SCPD tool results from IR-Plag-Dataset."
    )
    parser.add_argument(
        "--tools", nargs="+", default=None, metavar="TOOL",
        help="Restrict to base tool names (e.g. JPlag SIM) or run names (with --all-runs)",
    )
    parser.add_argument(
        "--metric", default="mcc", choices=["mcc", "f1", "auc", "accuracy"],
        help="Metric for best-run selection (default: mcc; ignored with --all-runs)",
    )
    parser.add_argument(
        "--all-runs", action="store_true",
        help="Load all runs without best-run selection (sweep debugging mode)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    fig_dir = args.output_dir / "figures"
    tbl_dir = args.output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    best_runs: dict[str, pd.Series] | None = None

    if args.all_runs:
        data, thresholds = load_all_predictions(TOOL_RUNS_CSVS, args.tools)
        suffix  = ""
        fsuffix = ("_" + "_".join(args.tools)) if args.tools else "_all"
    else:
        best_runs = select_best_runs(TOOL_RUNS_CSVS, args.metric, args.tools)
        data, thresholds = load_best_predictions(best_runs, TOOL_RUNS_CSVS)
        suffix  = " (best run per tool)"
        fsuffix = ""

    if not data:
        sys.exit("ERROR: no run data found. Make sure each tool's runs CSV exists.")

    names = list(data.keys())
    print(f"  Loaded: {names}")
    for name, df in data.items():
        n_plag = int(df["is_plagiarized"].sum())
        t      = thresholds[name]
        print(f"  {name}: {len(df)} rows, {n_plag} plagiarised, threshold={t:.2f}")

    print("\nComputing metrics...")
    global_metrics = compute_global_metrics(data, thresholds)
    lv_metrics     = compute_group_metrics(data, thresholds, "level", LEVEL_ORDER)
    case_metrics   = compute_group_metrics(data, thresholds, "case",  CASE_ORDER)
    f1_lv          = f1_matrix(lv_metrics,   LEVEL_ORDER)
    f1_case        = f1_matrix(case_metrics, CASE_ORDER)

    plag_lv = {
        name: {lv: lv_metrics[name].get(lv, {}) for lv in PLAG_LEVELS}
        for name in names
    }

    print("\nGenerating figures...")
    fig_roc_curves(data, global_metrics, fig_dir, suffix, fsuffix)
    fig_metrics_bar(global_metrics, fig_dir, suffix, fsuffix)
    fig_f1_by_level_heatmap(f1_lv,  fig_dir, suffix, fsuffix)
    fig_f1_by_case_heatmap(f1_case, fig_dir, suffix, fsuffix)
    if not args.all_runs:
        fig_f1_by_level_lines(plag_lv, fig_dir, fsuffix)
        fig_similarity_violin(data, thresholds, fig_dir, suffix, fsuffix)
        fig_best_per_level_bar(plag_lv, fig_dir, fsuffix)
    else:
        fig_similarity_violin(data, thresholds, fig_dir, suffix, fsuffix)

    print("\nSaving tables...")
    save_tables(best_runs, global_metrics, lv_metrics, f1_lv, f1_case, tbl_dir, fsuffix)

    print_summary(global_metrics, f1_lv, f1_case)
    print(f"\nDone.  Results in {args.output_dir.resolve()}/")


if __name__ == "__main__":
    main()
