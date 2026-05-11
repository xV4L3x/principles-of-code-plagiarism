#!/usr/bin/env python3
"""
sim_runner.py — Evaluate SIM over IR-Plag-Dataset.

Normal mode (default):
  Runs SIM with a fixed --min-run and --metric, writes the standard CSV
  and one raw .txt per case.

Sweep mode (--sweep):
  Re-runs SIM for every value in --sweep-runs (different -r values require
  actual re-execution), then evaluates all combinations of
  (min_run × metric × threshold) and picks the one with the highest F1.
  Writes out/sweep_results.csv (all combos) and out/sweep_best.txt (summary).

Layout:
  experiments/
    sim/
      sim_runner.py         ← this file
      sim_java              ← compiled binary (see README)
      out/
        sim_results.csv     ← normal mode output
        case-01_raw.txt     ← raw SIM stdout per case (normal mode)
        sweep_results.csv   ← sweep mode: all (min_run, metric, threshold) rows
        sweep_best.txt      ← sweep mode: human-readable summary

Usage:
  python sim_runner.py
  python sim_runner.py --min-run 10 --metric SUB_IN_ORIG --threshold 0.8
  python sim_runner.py --sweep
  python sim_runner.py --sweep --sweep-runs 3 5 8 10 15 20
"""

import argparse
import csv
import re
import subprocess
import sys
from itertools import product
from pathlib import Path

DATASET_ROOT = Path(__file__).parent.parent / "IR-Plag-Dataset"
OUT_DIR = Path(__file__).parent / "out"
OUTPUT_CSV = OUT_DIR / "sim_results.csv"
SIM_BIN_DEFAULT = Path(__file__).parent / "sim_java"

DEFAULT_MIN_RUN = 5
DEFAULT_METRIC = "MAX"       # MAX | AVG | SUB_IN_ORIG | ORIG_IN_SUB
DEFAULT_THRESHOLD = 0.5

SWEEP_MIN_RUNS = [3, 5, 8, 10, 15, 20]
SWEEP_METRICS = ["MAX", "AVG", "SUB_IN_ORIG", "ORIG_IN_SUB"]
SWEEP_THRESHOLDS = [round(v / 100, 2) for v in range(5, 100, 5)]  # 0.05..0.95

SIM_TIMEOUT_S = 30


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def find_java_files(directory: Path) -> list[Path]:
    return list(directory.rglob("*.java"))


def collect_submissions(case_dir: Path) -> dict[str, tuple[str, str, bool, list[Path]]]:
    """Returns {key: (level, sub_id, is_plagiarized, [java_files])}."""
    subs: dict[str, tuple[str, str, bool, list[Path]]] = {}

    for level_dir in sorted((case_dir / "plagiarized").iterdir()):
        if not level_dir.is_dir() or level_dir.name.startswith("."):
            continue
        level = level_dir.name
        for sub_dir in sorted(level_dir.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            files = find_java_files(sub_dir)
            if files:
                subs[f"plag_{level}_{sub_dir.name}"] = (level, sub_dir.name, True, files)

    for sub_dir in sorted((case_dir / "non-plagiarized").iterdir()):
        if not sub_dir.is_dir() or sub_dir.name.startswith("."):
            continue
        files = find_java_files(sub_dir)
        if files:
            subs[f"nonplag_{sub_dir.name}"] = ("non-plag", sub_dir.name, False, files)

    return subs


# ---------------------------------------------------------------------------
# SIM invocation
# ---------------------------------------------------------------------------

def run_sim(sim_bin: Path, file_a: Path, file_b: Path, min_run: int) -> tuple[str, bool]:
    cmd = [str(sim_bin), "-p", "-T", "-r", str(min_run), str(file_a), str(file_b)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=SIM_TIMEOUT_S)
    if result.returncode != 0:
        err = (result.stdout + result.stderr).strip()
        print(f"    SIM error: {err[:300]}", file=sys.stderr)
        return "", False
    return result.stdout, True


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_sim_percentages(stdout: str, orig_file: Path, sub_file: Path) -> tuple[float, float]:
    """
    Returns (orig_in_sub, sub_in_orig) as fractions [0-1].

    SIM output (with -p -T):
      /path/orig.java consists for X % of /path/sub.java material   → orig_in_sub
      /path/sub.java  consists for Y % of /path/orig.java material   → sub_in_orig

    Empty stdout → both 0.0 (no matching runs found).
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
        # The subject of "consists for X% of Y" is the file at the start of the line
        if line.strip().startswith(orig_str):
            orig_in_sub = pct
        elif line.strip().startswith(sub_str):
            sub_in_orig = pct

    return orig_in_sub, sub_in_orig


def apply_metric(orig_in_sub: float, sub_in_orig: float, metric: str) -> float:
    """Combine two directional SIM scores into a single similarity value."""
    if metric == "MAX":
        return max(orig_in_sub, sub_in_orig)
    if metric == "AVG":
        return (orig_in_sub + sub_in_orig) / 2.0
    if metric == "SUB_IN_ORIG":
        return sub_in_orig
    if metric == "ORIG_IN_SUB":
        return orig_in_sub
    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Per-submission similarity
# ---------------------------------------------------------------------------

def similarity_for_submission(
    sim_bin: Path,
    orig_file: Path,
    sub_files: list[Path],
    min_run: int,
) -> tuple[float, float, str]:
    """
    Run SIM pairwise for each java file in the submission against orig_file.
    Returns (max_orig_in_sub, max_sub_in_orig, raw_stdout_concatenated).
    If a submission has multiple java files, we take the max over all files.
    """
    best_orig_in_sub = 0.0
    best_sub_in_orig = 0.0
    raw_parts: list[str] = []

    for sub_file in sub_files:
        stdout, ok = run_sim(sim_bin, orig_file, sub_file, min_run)
        raw_parts.append(f"# sim_java -r {min_run} {orig_file.name} {sub_file.name}\n{stdout}")
        if ok:
            a, b = parse_sim_percentages(stdout, orig_file, sub_file)
            best_orig_in_sub = max(best_orig_in_sub, a)
            best_sub_in_orig = max(best_sub_in_orig, b)

    return best_orig_in_sub, best_sub_in_orig, "\n".join(raw_parts)


# ---------------------------------------------------------------------------
# Evaluation helpers (used in sweep mode)
# ---------------------------------------------------------------------------

def compute_metrics(is_plag_list: list[bool], similarities: list[float], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for is_plag, sim in zip(is_plag_list, similarities):
        pred = sim >= threshold
        if pred and is_plag:
            tp += 1
        elif pred and not is_plag:
            fp += 1
        elif not pred and not is_plag:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(is_plag_list) if is_plag_list else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ---------------------------------------------------------------------------
# Normal mode
# ---------------------------------------------------------------------------

def run_normal(args: argparse.Namespace) -> None:
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _get_cases(args)
    rows: list[dict] = []

    for case_dir in cases:
        case_name = case_dir.name
        print(f"\n{'='*50}\n{case_name}\n{'='*50}")

        orig_files = find_java_files(case_dir / "original")
        if not orig_files:
            print(f"  WARNING: no .java in original/ — skipping", file=sys.stderr)
            continue
        orig_file = orig_files[0]
        print(f"  Original : {orig_file.name}")

        submissions = collect_submissions(case_dir)
        print(f"  Submissions: {len(submissions)}  |  min_run={args.min_run}  metric={args.metric}")

        case_raw: list[str] = [f"=== {case_name} (original: {orig_file}) ===\n"]

        for key, (level, sub_id, is_plag, sub_files) in sorted(submissions.items()):
            try:
                orig_in_sub, sub_in_orig, raw = similarity_for_submission(
                    args.sim_bin, orig_file, sub_files, args.min_run
                )
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT for {key}", file=sys.stderr)
                orig_in_sub, sub_in_orig, raw = 0.0, 0.0, f"# TIMEOUT: {key}\n"

            case_raw.append(raw)
            sim = apply_metric(orig_in_sub, sub_in_orig, args.metric)
            predicted = sim >= args.threshold
            rows.append({
                "case": case_name, "level": level, "submission_id": sub_id,
                "similarity": round(sim, 4), "is_plagiarized": is_plag,
                "predicted_plag": predicted,
            })
            flag = "PLAG" if is_plag else "    "
            print(f"  [{flag}] {key:<25} sim={sim:.4f}  pred={'Y' if predicted else 'N'}")

        raw_path = out_dir / f"{case_name}_raw.txt"
        raw_path.write_text("\n".join(case_raw))
        print(f"  Raw output saved → {raw_path}")

    if not rows:
        print("\nNo results produced.", file=sys.stderr)
        sys.exit(1)

    _write_csv(args.output, rows)
    print(f"\nDone. {len(rows)} rows → {args.output}")
    print(f"Raw outputs in {out_dir}/")


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace) -> None:
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _get_cases(args)
    min_runs = args.sweep_runs

    # data[min_run] = list of {case, level, sub_id, is_plag, orig_in_sub, sub_in_orig}
    data: dict[int, list[dict]] = {r: [] for r in min_runs}

    for min_run in min_runs:
        print(f"\n{'='*50}\nSweeping min_run={min_run}\n{'='*50}")

        for case_dir in cases:
            case_name = case_dir.name
            orig_files = find_java_files(case_dir / "original")
            if not orig_files:
                continue
            orig_file = orig_files[0]
            submissions = collect_submissions(case_dir)
            print(f"  {case_name}  ({len(submissions)} submissions)", flush=True)

            for key, (level, sub_id, is_plag, sub_files) in sorted(submissions.items()):
                try:
                    orig_in_sub, sub_in_orig, _ = similarity_for_submission(
                        args.sim_bin, orig_file, sub_files, min_run
                    )
                except subprocess.TimeoutExpired:
                    orig_in_sub, sub_in_orig = 0.0, 0.0

                data[min_run].append({
                    "case": case_name, "level": level, "submission_id": sub_id,
                    "is_plagiarized": is_plag,
                    "orig_in_sub": orig_in_sub, "sub_in_orig": sub_in_orig,
                })

    # Evaluate all (min_run, metric, threshold) combinations
    print(f"\n{'='*50}\nEvaluating combinations...\n{'='*50}")
    combo_rows: list[dict] = []

    for min_run, metric, threshold in product(min_runs, SWEEP_METRICS, SWEEP_THRESHOLDS):
        rows = data[min_run]
        if not rows:
            continue
        is_plag_list = [r["is_plagiarized"] for r in rows]
        sims = [apply_metric(r["orig_in_sub"], r["sub_in_orig"], metric) for r in rows]
        m = compute_metrics(is_plag_list, sims, threshold)
        combo_rows.append({
            "min_run": min_run, "metric": metric, "threshold": threshold,
            **{k: round(v, 4) for k, v in m.items()},
        })

    # Sort by F1 descending
    combo_rows.sort(key=lambda r: r["f1"], reverse=True)

    # Write sweep CSV
    sweep_csv = out_dir / "sweep_results.csv"
    sweep_fields = ["min_run", "metric", "threshold", "f1", "accuracy", "precision", "recall",
                    "tp", "fp", "tn", "fn"]
    with open(sweep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sweep_fields)
        w.writeheader()
        w.writerows(combo_rows)
    print(f"Sweep CSV → {sweep_csv}  ({len(combo_rows)} combinations)")

    # Print top-20 and write summary
    header = f"{'min_run':>8}  {'metric':<12}  {'threshold':>9}  {'F1':>6}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for row in combo_rows[:20]:
        lines.append(
            f"{row['min_run']:>8}  {row['metric']:<12}  {row['threshold']:>9.2f}"
            f"  {row['f1']:>6.4f}  {row['accuracy']:>6.4f}"
            f"  {row['precision']:>6.4f}  {row['recall']:>6.4f}"
        )
    lines.append(sep)
    best = combo_rows[0]
    lines.append(
        f"\nBEST  →  min_run={best['min_run']}  metric={best['metric']}"
        f"  threshold={best['threshold']:.2f}"
        f"  F1={best['f1']:.4f}  Accuracy={best['accuracy']:.4f}"
    )

    summary = "\n".join(lines)
    print("\n" + summary)

    best_path = out_dir / "sweep_best.txt"
    best_path.write_text(summary + "\n")
    print(f"\nSummary saved → {best_path}")

    # Offer to write the winning combo as the standard CSV
    print(f"\nTo run the winner: python sim_runner.py "
          f"--min-run {best['min_run']} --metric {best['metric']} "
          f"--threshold {best['threshold']:.2f}")


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


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["case", "level", "submission_id", "similarity", "is_plagiarized", "predicted_plag"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SIM over IR-Plag-Dataset — normal or sweep mode."
    )
    parser.add_argument("--sim-bin", type=Path, default=SIM_BIN_DEFAULT,
                        help="Path to sim_java binary")
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV,
                        help="Output CSV (normal mode). Raw .txt and sweep files go alongside it.")
    parser.add_argument("--cases", nargs="+", default=None, metavar="CASE")

    # Normal mode options
    parser.add_argument("--min-run", type=int, default=DEFAULT_MIN_RUN,
                        help=f"Minimum token run length for SIM (default: {DEFAULT_MIN_RUN})")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        choices=SWEEP_METRICS,
                        help="How to combine the two directional scores "
                             "(MAX | AVG | SUB_IN_ORIG | ORIG_IN_SUB, default: MAX)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity threshold for predicted_plag (default: {DEFAULT_THRESHOLD})")

    # Sweep mode
    parser.add_argument("--sweep", action="store_true",
                        help="Try all (min_run × metric × threshold) combinations and "
                             "report the best by F1")
    parser.add_argument("--sweep-runs", nargs="+", type=int, default=SWEEP_MIN_RUNS,
                        metavar="N",
                        help=f"min_run values to try in sweep mode "
                             f"(default: {SWEEP_MIN_RUNS})")

    args = parser.parse_args()

    if not args.sim_bin.exists():
        sys.exit(
            f"ERROR: sim_java binary not found at {args.sim_bin}\n"
            "Compile from https://github.com/sauloq/sim and place at experiments/sim/sim_java"
        )
    if not args.dataset.exists():
        sys.exit(f"ERROR: Dataset not found at {args.dataset}")

    if args.sweep:
        run_sweep(args)
    else:
        run_normal(args)


if __name__ == "__main__":
    main()
