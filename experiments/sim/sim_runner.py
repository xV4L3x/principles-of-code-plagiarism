#!/usr/bin/env python3
"""
sim_runner.py — Evaluate SIM over IR-Plag-Dataset.

Each invocation is a named run encoding its parameters. Results are written to:
  out/<run_name>_results.csv         — per-submission predictions
  out/sim_runs.csv                   — one summary row per run (metrics + params)

Score caching: directional (orig_in_sub, sub_in_orig) scores are cached per
(case, min_run) in out/case-XX-minrun-R_scores.csv. Runs that share the same
min_run but differ only in metric or threshold reuse the cache automatically.
Pass --force to re-run SIM even when a cache exists.

Usage:
  python sim_runner.py
  python sim_runner.py --min-run 10 --metric ORIG_IN_SUB --threshold 0.6
  python sim_runner.py --min-run 5 --metric AVG   # reuses cached scores
  python sim_runner.py --min-run 5 --force        # re-runs SIM
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

DATASET_ROOT = Path(__file__).parent.parent / "IR-Plag-Dataset"
OUT_DIR = Path(__file__).parent / "out"
SIM_BIN_DEFAULT = Path(__file__).parent / "sim_java"

RUNS_CSV = OUT_DIR / "sim_runs.csv"
RUNS_FIELDNAMES = [
    "run_name", "min_run", "threshold", "metric",
    "tp", "fp", "tn", "fn",
    "precision", "recall", "f1", "accuracy", "auc", "mcc",
    "predictions_csv",
]
PREDICTIONS_FIELDNAMES = [
    "case", "level", "submission_id", "similarity", "is_plagiarized", "predicted_plag",
]
SCORE_CACHE_FIELDNAMES = [
    "level", "sub_id", "is_plag", "orig_in_sub", "sub_in_orig",
]

DEFAULT_MIN_RUN = 5
DEFAULT_METRIC = "MAX"
DEFAULT_THRESHOLD = 0.5

SIM_METRICS = ["MAX", "AVG", "SUB_IN_ORIG", "ORIG_IN_SUB"]
SIM_TIMEOUT_S = 30


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def find_java_files(directory: Path) -> list[Path]:
    return list(directory.rglob("*.java"))


def collect_submissions(case_dir: Path) -> list[tuple[str, str, bool, list[Path]]]:
    """Returns list of (level, sub_id, is_plagiarized, [java_files]), sorted."""
    subs: list[tuple[str, str, bool, list[Path]]] = []

    for level_dir in sorted((case_dir / "plagiarized").iterdir()):
        if not level_dir.is_dir() or level_dir.name.startswith("."):
            continue
        level = level_dir.name
        for sub_dir in sorted(level_dir.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            files = find_java_files(sub_dir)
            if files:
                subs.append((level, sub_dir.name, True, files))

    for sub_dir in sorted((case_dir / "non-plagiarized").iterdir()):
        if not sub_dir.is_dir() or sub_dir.name.startswith("."):
            continue
        files = find_java_files(sub_dir)
        if files:
            subs.append(("non-plag", sub_dir.name, False, files))

    return subs


# ---------------------------------------------------------------------------
# SIM invocation and parsing
# ---------------------------------------------------------------------------

def run_sim(sim_bin: Path, file_a: Path, file_b: Path, min_run: int) -> tuple[str, bool]:
    cmd = [str(sim_bin), "-p", "-T", "-r", str(min_run), str(file_a), str(file_b)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=SIM_TIMEOUT_S)
    if result.returncode != 0:
        err = (result.stdout + result.stderr).strip()
        print(f"    SIM error: {err[:300]}", file=sys.stderr)
        return "", False
    return result.stdout, True


def parse_sim_percentages(stdout: str, orig_file: Path, sub_file: Path) -> tuple[float, float]:
    """
    Returns (orig_in_sub, sub_in_orig) as fractions [0-1].

    SIM output (with -p -T):
      /path/orig.java consists for X % of /path/sub.java material  → orig_in_sub
      /path/sub.java  consists for Y % of /path/orig.java material  → sub_in_orig
    """
    orig_in_sub = 0.0
    sub_in_orig = 0.0
    orig_str = str(orig_file)
    sub_str = str(sub_file)

    for line in stdout.splitlines():
        m = re.search(r'consists\s+for\s+(\d+(?:\.\d+)?)\s*%', line)
        if not m:
            continue
        pct = float(m.group(1)) / 100.0
        if line.strip().startswith(orig_str):
            orig_in_sub = pct
        elif line.strip().startswith(sub_str):
            sub_in_orig = pct

    return orig_in_sub, sub_in_orig


def apply_metric(orig_in_sub: float, sub_in_orig: float, metric: str) -> float:
    if metric == "MAX":
        return max(orig_in_sub, sub_in_orig)
    if metric == "AVG":
        return (orig_in_sub + sub_in_orig) / 2.0
    if metric == "SUB_IN_ORIG":
        return sub_in_orig
    if metric == "ORIG_IN_SUB":
        return orig_in_sub
    raise ValueError(f"Unknown metric: {metric}")


def scores_for_submission(
    sim_bin: Path,
    orig_file: Path,
    sub_files: list[Path],
    min_run: int,
) -> tuple[float, float]:
    """Run SIM for each java file in the submission, return max directional scores."""
    best_orig_in_sub = 0.0
    best_sub_in_orig = 0.0
    for sub_file in sub_files:
        stdout, ok = run_sim(sim_bin, orig_file, sub_file, min_run)
        if ok:
            a, b = parse_sim_percentages(stdout, orig_file, sub_file)
            best_orig_in_sub = max(best_orig_in_sub, a)
            best_sub_in_orig = max(best_sub_in_orig, b)
    return best_orig_in_sub, best_sub_in_orig


# ---------------------------------------------------------------------------
# Score cache  (keyed by case + min_run)
# ---------------------------------------------------------------------------

def cache_path(case_name: str, min_run: int) -> Path:
    return OUT_DIR / f"{case_name}-minrun-{min_run}_scores.csv"


def load_score_cache(case_name: str, min_run: int) -> list[dict] | None:
    p = cache_path(case_name, min_run)
    if not p.exists():
        return None
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def save_score_cache(case_name: str, min_run: int, rows: list[dict]) -> None:
    p = cache_path(case_name, min_run)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_CACHE_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict:
    predicted = y_score >= threshold
    tp = int(np.sum(predicted & y_true))
    fp = int(np.sum(predicted & ~y_true))
    tn = int(np.sum(~predicted & ~y_true))
    fn = int(np.sum(~predicted & y_true))

    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)
    f1        = _safe_div(2 * precision * recall, precision + recall)
    accuracy  = _safe_div(tp + tn, len(y_true))

    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = 0.0

    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = _safe_div(tp * tn - fp * fn, denom)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
        "auc":       round(auc, 4),
        "mcc":       round(mcc, 4),
    }


# ---------------------------------------------------------------------------
# Runs CSV (upsert)
# ---------------------------------------------------------------------------

def append_run(run_row: dict) -> None:
    rows: list[dict] = []
    if RUNS_CSV.exists():
        with open(RUNS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("run_name") != run_row["run_name"]]
    rows.append(run_row)
    with open(RUNS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_cases(args: argparse.Namespace) -> list[Path]:
    cases = sorted(
        d for d in args.dataset.iterdir()
        if d.is_dir() and d.name.startswith("case-")
    )
    if args.cases:
        selected = set(args.cases)
        cases = [c for c in cases if c.name in selected]
        if not cases:
            sys.exit(f"ERROR: None of {args.cases} found in {args.dataset}")
    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SIM over IR-Plag-Dataset (run-based, one execution per invocation)."
    )
    parser.add_argument("--sim-bin", type=Path, default=SIM_BIN_DEFAULT,
                        help="Path to sim_java binary")
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT)
    parser.add_argument("--cases", nargs="+", default=None, metavar="CASE")
    parser.add_argument("--min-run", type=int, default=DEFAULT_MIN_RUN,
                        help=f"Minimum token run length for SIM -r flag (default: {DEFAULT_MIN_RUN})")
    parser.add_argument("--metric", default=DEFAULT_METRIC, choices=SIM_METRICS,
                        help="How to combine directional scores (default: MAX)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity cutoff for predicted_plag (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--force", action="store_true",
                        help="Re-run SIM even when a cached score file exists")
    args = parser.parse_args()

    if not args.sim_bin.exists():
        sys.exit(
            f"ERROR: sim_java binary not found at {args.sim_bin}\n"
            "Compile from https://github.com/sauloq/sim and place at experiments/sim/sim_java"
        )
    if not args.dataset.exists():
        sys.exit(f"ERROR: Dataset not found at {args.dataset}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_name = (
        f"SIM-Threshold-{args.threshold:.2f}"
        f"-MinRun-{args.min_run}"
        f"-Metric-{args.metric}"
    )
    predictions_csv = OUT_DIR / f"{run_name}_results.csv"

    print(f"\nRun: {run_name}")
    print(f"  min_run={args.min_run}  threshold={args.threshold:.2f}  metric={args.metric}")
    print(f"  Output: {predictions_csv.name}")

    cases = _get_cases(args)
    all_rows: list[dict] = []

    for case_dir in cases:
        case_name = case_dir.name
        print(f"\n{'='*50}\n{case_name}\n{'='*50}")

        orig_files = find_java_files(case_dir / "original")
        if not orig_files:
            print(f"  WARNING: no .java in original/ — skipping", file=sys.stderr)
            continue
        orig_file = orig_files[0]

        cached = None if args.force else load_score_cache(case_name, args.min_run)
        if cached is not None:
            print(f"  Using cached scores ({len(cached)} entries, min_run={args.min_run})")
            for entry in cached:
                orig_in_sub = float(entry["orig_in_sub"])
                sub_in_orig = float(entry["sub_in_orig"])
                is_plag = entry["is_plag"] == "True"
                sim = apply_metric(orig_in_sub, sub_in_orig, args.metric)
                predicted = sim >= args.threshold
                all_rows.append({
                    "case": case_name,
                    "level": entry["level"],
                    "submission_id": entry["sub_id"],
                    "similarity": round(sim, 4),
                    "is_plagiarized": is_plag,
                    "predicted_plag": predicted,
                })
                flag = "PLAG" if is_plag else "    "
                key = f"{entry['level']}_{entry['sub_id']}"
                print(f"  [{flag}] {key:<25} sim={sim:.4f}  pred={'Y' if predicted else 'N'}")
        else:
            submissions = collect_submissions(case_dir)
            print(f"  Submissions: {len(submissions)}  |  Running SIM with -r {args.min_run}")
            score_cache_rows: list[dict] = []
            for level, sub_id, is_plag, sub_files in submissions:
                try:
                    orig_in_sub, sub_in_orig = scores_for_submission(
                        args.sim_bin, orig_file, sub_files, args.min_run
                    )
                except subprocess.TimeoutExpired:
                    print(f"  TIMEOUT for {level}/{sub_id}", file=sys.stderr)
                    orig_in_sub, sub_in_orig = 0.0, 0.0

                score_cache_rows.append({
                    "level": level, "sub_id": sub_id, "is_plag": is_plag,
                    "orig_in_sub": round(orig_in_sub, 6),
                    "sub_in_orig": round(sub_in_orig, 6),
                })

                sim = apply_metric(orig_in_sub, sub_in_orig, args.metric)
                predicted = sim >= args.threshold
                all_rows.append({
                    "case": case_name,
                    "level": level,
                    "submission_id": sub_id,
                    "similarity": round(sim, 4),
                    "is_plagiarized": is_plag,
                    "predicted_plag": predicted,
                })
                flag = "PLAG" if is_plag else "    "
                key = f"{level}_{sub_id}"
                print(f"  [{flag}] {key:<25} sim={sim:.4f}  pred={'Y' if predicted else 'N'}")

            save_score_cache(case_name, args.min_run, score_cache_rows)
            print(f"  Score cache saved → {cache_path(case_name, args.min_run).name}")

    if not all_rows:
        print("\nNo results produced.", file=sys.stderr)
        sys.exit(1)

    with open(predictions_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PREDICTIONS_FIELDNAMES)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPredictions → {predictions_csv.name}  ({len(all_rows)} rows)")

    y_true  = np.array([r["is_plagiarized"] for r in all_rows], dtype=bool)
    y_score = np.array([r["similarity"] for r in all_rows], dtype=float)
    m = compute_metrics(y_true, y_score, args.threshold)

    run_row = {
        "run_name":        run_name,
        "min_run":         args.min_run,
        "threshold":       args.threshold,
        "metric":          args.metric,
        **m,
        "predictions_csv": predictions_csv.name,
    }
    append_run(run_row)

    print(
        f"Metrics — "
        f"Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
        f"F1={m['f1']:.4f}  Accuracy={m['accuracy']:.4f}  "
        f"AUC={m['auc']:.4f}  MCC={m['mcc']:.4f}"
    )
    print(f"Run logged → {RUNS_CSV.name}")


if __name__ == "__main__":
    main()
