#!/usr/bin/env python3
"""
dolos_runner.py — Evaluate Dolos over IR-Plag-Dataset.

Each invocation is a named **run** that encodes the parameter combination
(kgram, window, threshold, metric) in the run name.  Results are written to:
  - A per-run predictions CSV in out/
  - A row appended (or updated) in out/dolos_runs.csv

Report directories are keyed by (kgram, window) so that multiple runs that
differ only in metric or threshold reuse the same Dolos execution, saving time.
Pass --force to re-run Dolos even when a cached report exists.

Layout:
  experiments/
    dolos/
      dolos_runner.py                                         ← this file
      node_modules/                                           ← npm install @dodona/dolos
      .nvmrc                                                  ← pins Node 22
      out/
        dolos_runs.csv                                        ← one row per run (params + metrics)
        Dolos-Threshold-0.25-KGram-23-Window-17-Metric-ORIG_IN_SUB_results.csv
        case-01-kgram-23-window-17_report/                   ← raw Dolos CSV report per case
        ...

Usage:
  python dolos_runner.py
  python dolos_runner.py --kgram 10 --metric MAX --threshold 0.4
  python dolos_runner.py --metric ORIG_IN_SUB --threshold 0.25
  python dolos_runner.py --kgram 10 --force
  python dolos_runner.py --cases case-01 case-02
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

DATASET_ROOT = Path(__file__).parent.parent / "IR-Plag-Dataset"
OUT_DIR = Path(__file__).parent / "out"
DOLOS_BIN = Path(__file__).parent / "node_modules" / ".bin" / "dolos"

DEFAULT_KGRAM     = 23
DEFAULT_WINDOW    = 17
DEFAULT_METRIC    = "COMBINED"
DEFAULT_THRESHOLD = 0.5

VALID_METRICS = ["COMBINED", "MAX", "AVG", "SUB_IN_ORIG", "ORIG_IN_SUB"]

DOLOS_TIMEOUT_S = 120

RUNS_CSV = OUT_DIR / "dolos_runs.csv"
RUNS_FIELDNAMES = [
    "run_name", "kgram", "window", "threshold", "metric",
    "tp", "fp", "tn", "fn",
    "precision", "recall", "f1", "accuracy", "auc", "mcc",
    "predictions_csv",
]

PREDICTIONS_FIELDNAMES = [
    "case", "level", "submission_id", "similarity", "is_plagiarized", "predicted_plag",
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score
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
    return dict(
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        precision=round(p, 4), recall=round(r, 4),
        f1=round(f1, 4), accuracy=round(acc, 4), auc=round(auc, 4),
        mcc=round(mcc, 4),
    )


# ---------------------------------------------------------------------------
# Runs CSV helpers
# ---------------------------------------------------------------------------

def append_run(run_row: dict) -> None:
    """Append (or overwrite if run_name already exists) a row in dolos_runs.csv."""
    existing: list[dict] = []
    if RUNS_CSV.exists():
        with open(RUNS_CSV, newline="") as f:
            existing = list(csv.DictReader(f))

    replaced = False
    for i, row in enumerate(existing):
        if row["run_name"] == run_row["run_name"]:
            existing[i] = run_row
            replaced = True
            break
    if not replaced:
        existing.append(run_row)

    with open(RUNS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(existing)

    action = "Updated" if replaced else "Appended"
    print(f"  {action} run '{run_row['run_name']}' in {RUNS_CSV}")


# ---------------------------------------------------------------------------
# Node / Dolos binary helpers
# ---------------------------------------------------------------------------

def _find_node() -> str:
    nvm_versions = Path.home() / ".nvm" / "versions" / "node"
    if nvm_versions.is_dir():
        for v22 in sorted(nvm_versions.glob("v22.*"), reverse=True):
            node = v22 / "bin" / "node"
            if node.exists():
                return str(node)
    return "node"


def _dolos_cmd(kgram: int, window: int, report_dir: Path, java_files: list[Path]) -> list[str]:
    node = _find_node()
    return [
        node, str(DOLOS_BIN), "run",
        "-l", "java",
        "-f", "csv",
        "-k", str(kgram),
        "-w", str(window),
        "-L", "0",
        "-o", str(report_dir),
        *[str(f) for f in java_files],
    ]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def find_java_files(directory: Path) -> list[Path]:
    return list(directory.rglob("*.java"))


def collect_case_files(
    case_dir: Path,
) -> dict[str, tuple[str, str, bool, list[Path]]]:
    """
    Returns {folder_key: (level, sub_id, is_plagiarized, [java_files])}.
    folder_key = 'plag_L3_02' or 'nonplag_04'.
    """
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
                key = f"plag_{level}_{sub_dir.name}"
                subs[key] = (level, sub_dir.name, True, files)

    for sub_dir in sorted((case_dir / "non-plagiarized").iterdir()):
        if not sub_dir.is_dir() or sub_dir.name.startswith("."):
            continue
        files = find_java_files(sub_dir)
        if files:
            key = f"nonplag_{sub_dir.name}"
            subs[key] = ("non-plag", sub_dir.name, False, files)

    return subs


def _path_to_folder_key(path: str, case_dir: Path) -> str | None:
    try:
        rel = Path(path).relative_to(case_dir)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    if parts[0] == "original":
        return "original"
    if parts[0] == "plagiarized" and len(parts) >= 3:
        return f"plag_{parts[1]}_{parts[2]}"
    if parts[0] == "non-plagiarized" and len(parts) >= 2:
        return f"nonplag_{parts[1]}"
    return None


# ---------------------------------------------------------------------------
# Dolos invocation
# ---------------------------------------------------------------------------

def run_dolos(
    java_files: list[Path],
    report_dir: Path,
    kgram: int,
    window: int,
) -> bool:
    import shutil
    import subprocess
    if report_dir.exists():
        shutil.rmtree(report_dir)
    cmd = _dolos_cmd(kgram, window, report_dir, java_files)
    print(f"  $ {' '.join(cmd[:8])} ... ({len(java_files)} files)", flush=True)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DOLOS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {DOLOS_TIMEOUT_S}s", file=sys.stderr)
        return False
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        print(f"  Dolos error:\n{output[-2000:]}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def _parse_files_csv(report_dir: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    files_csv = report_dir / "files.csv"
    if not files_csv.exists():
        return result
    with open(files_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                result[row["path"]] = int(row["amountOfKgrams"])
            except (KeyError, ValueError):
                pass
    return result


def _parse_pairs_csv(report_dir: Path) -> list[dict]:
    pairs_csv = report_dir / "pairs.csv"
    if not pairs_csv.exists():
        return []
    with open(pairs_csv, newline="") as f:
        return list(csv.DictReader(f))


def apply_metric(
    orig_covered: int,
    sub_covered: int,
    orig_total: int,
    sub_total: int,
    combined: float,
    metric: str,
) -> float:
    """Compute a single similarity score from the raw Dolos pair data."""
    if metric == "COMBINED":
        return combined
    orig_frac = orig_covered / orig_total if orig_total > 0 else 0.0
    sub_frac  = sub_covered  / sub_total  if sub_total  > 0 else 0.0
    if metric == "ORIG_IN_SUB":
        return orig_frac
    if metric == "SUB_IN_ORIG":
        return sub_frac
    if metric == "MAX":
        return max(orig_frac, sub_frac)
    if metric == "AVG":
        return (orig_frac + sub_frac) / 2.0
    raise ValueError(f"Unknown metric: {metric}")


def extract_submission_sims(
    report_dir: Path,
    case_dir: Path,
    metric: str,
) -> dict[str, float]:
    """
    Parse the Dolos report and return {folder_key: similarity} for every
    submission. Only pairs involving at least one file from 'original/' are
    considered. For multi-file submissions, the max similarity is returned.
    """
    file_kgrams = _parse_files_csv(report_dir)
    pairs = _parse_pairs_csv(report_dir)

    orig_paths: set[str] = {
        p for p in file_kgrams if _path_to_folder_key(p, case_dir) == "original"
    }

    sub_sims: dict[str, float] = {}

    for row in pairs:
        left_path  = row.get("leftFilePath",  "")
        right_path = row.get("rightFilePath", "")

        if left_path in orig_paths:
            orig_path, sub_path = left_path, right_path
            orig_covered = int(row.get("leftCovered",  0))
            sub_covered  = int(row.get("rightCovered", 0))
        elif right_path in orig_paths:
            orig_path, sub_path = right_path, left_path
            orig_covered = int(row.get("rightCovered", 0))
            sub_covered  = int(row.get("leftCovered",  0))
        else:
            continue

        folder_key = _path_to_folder_key(sub_path, case_dir)
        if folder_key is None or folder_key == "original":
            continue

        orig_total = file_kgrams.get(orig_path, 0)
        sub_total  = file_kgrams.get(sub_path,  0)
        combined   = float(row.get("similarity", 0.0))

        sim = apply_metric(orig_covered, sub_covered, orig_total, sub_total, combined, metric)
        sub_sims[folder_key] = max(sub_sims.get(folder_key, 0.0), sim)

    return sub_sims


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Dolos over IR-Plag-Dataset and write a named results run."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT,
                        help="Path to IR-Plag-Dataset directory")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity cutoff for predicted_plag (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--kgram", type=int, default=DEFAULT_KGRAM,
                        help=f"k-gram length for fingerprinting (default: {DEFAULT_KGRAM})")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"Winnowing window size in kgrams (default: {DEFAULT_WINDOW})")
    parser.add_argument("--metric", default=DEFAULT_METRIC, choices=VALID_METRICS,
                        help="Similarity metric — COMBINED (Dolos default), MAX, AVG, "
                             "SUB_IN_ORIG, ORIG_IN_SUB (default: COMBINED)")
    parser.add_argument("--cases", nargs="+", default=None, metavar="CASE",
                        help="Run only these cases, e.g. --cases case-01 case-03")
    parser.add_argument("--force", action="store_true",
                        help="Re-run Dolos even when a cached report exists for this kgram+window")
    args = parser.parse_args()

    run_name = (f"Dolos-Threshold-{args.threshold:.2f}"
                f"-KGram-{args.kgram}"
                f"-Window-{args.window}"
                f"-Metric-{args.metric}")
    predictions_filename = f"{run_name}_results.csv"
    predictions_csv = OUT_DIR / predictions_filename

    print("=" * 60)
    print(f"Run: {run_name}")
    print(f"  threshold  = {args.threshold}")
    print(f"  kgram      = {args.kgram}")
    print(f"  window     = {args.window}")
    print(f"  metric     = {args.metric}")
    print(f"  output     = {predictions_csv}")
    print("=" * 60)

    if not DOLOS_BIN.exists():
        sys.exit(
            f"ERROR: Dolos not found at {DOLOS_BIN}\n"
            "Run: cd experiments/dolos && npm install @dodona/dolos\n"
            "Requires Node.js 22 — use nvm: nvm use 22"
        )
    if not args.dataset.exists():
        sys.exit(f"ERROR: Dataset not found at {args.dataset}")

    cases = sorted(
        d for d in args.dataset.iterdir()
        if d.is_dir() and d.name.startswith("case-")
    )
    if args.cases:
        selected = set(args.cases)
        cases = [c for c in cases if c.name in selected]
        if not cases:
            sys.exit(f"ERROR: None of {args.cases} found in {args.dataset}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for case_dir in cases:
        case_name = case_dir.name
        print(f"\n{'='*50}\n{case_name}\n{'='*50}")

        orig_files = find_java_files(case_dir / "original")
        if not orig_files:
            print(f"  WARNING: no .java in original/ — skipping", file=sys.stderr)
            continue

        subs = collect_case_files(case_dir)
        all_java = orig_files + [f for _, _, _, files in subs.values() for f in files]

        # Report directory is keyed by (kgram, window) so different metric/threshold
        # runs can reuse the same Dolos execution.
        report_dir = OUT_DIR / f"{case_name}-kgram-{args.kgram}-window-{args.window}_report"

        if report_dir.exists() and not args.force:
            print(f"  Reusing cached report: {report_dir.name}")
        else:
            print(f"  Files: {len(all_java)} total  |  kgram={args.kgram}  window={args.window}")
            ok = run_dolos(all_java, report_dir, args.kgram, args.window)
            if not ok:
                print(f"  Skipping {case_name} due to Dolos error.")
                continue

        sub_sims = extract_submission_sims(report_dir, case_dir, args.metric)
        matched = sum(1 for v in sub_sims.values() if v > 0.0)
        print(f"  Similarity found for {matched}/{len(subs)} submissions vs original  "
              f"[metric={args.metric}]")

        for key, (level, sub_id, is_plag, _) in sorted(subs.items()):
            sim = sub_sims.get(key, 0.0)
            predicted = sim >= args.threshold
            rows.append({
                "case": case_name, "level": level, "submission_id": sub_id,
                "similarity": round(sim, 4), "is_plagiarized": is_plag,
                "predicted_plag": predicted,
            })
            flag = "PLAG" if is_plag else "    "
            print(f"  [{flag}] {key:<25} sim={sim:.4f}  pred={'Y' if predicted else 'N'}")

    if not rows:
        print("\nNo results produced. Check Dolos errors above.", file=sys.stderr)
        sys.exit(1)

    # Write predictions CSV
    with open(predictions_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTIONS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPredictions written → {predictions_csv}  ({len(rows)} rows)")

    # Compute metrics and append to runs CSV
    y_true  = np.array([r["is_plagiarized"] for r in rows], dtype=bool)
    y_score = np.array([r["similarity"]     for r in rows], dtype=float)
    m = compute_metrics(y_true, y_score, args.threshold)

    run_row = {
        "run_name":  run_name,
        "kgram":     args.kgram,
        "window":    args.window,
        "threshold": args.threshold,
        "metric":    args.metric,
        **m,
        "predictions_csv": predictions_filename,
    }
    append_run(run_row)

    print(f"\nMetrics @ threshold={args.threshold:.2f}:")
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  Accuracy={m['accuracy']:.4f}  "
          f"AUC={m['auc']:.4f}  MCC={m['mcc']:.4f}")
    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
