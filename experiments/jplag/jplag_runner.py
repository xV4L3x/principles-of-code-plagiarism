#!/usr/bin/env python3
"""
jplag_runner.py — Evaluate JPlag over IR-Plag-Dataset.

Strategy per case:
  1. Copy original + all plagiarized + non-plagiarized submissions into a
     temp directory (each in its own subfolder, uniquely named).
  2. Run JPlag once with --shown-comparisons -1 (store all pairs).
  3. Parse the report ZIP: extract similarity between "original" and each
     other submission (use MAX metric, as per IR-Plag evaluation protocol).
  4. Apply the specified threshold and emit one CSV row per submission.
  5. Compute metrics and append a summary row to out/jplag_runs.csv so
     different parameter configurations are tracked side-by-side.

Layout:
  experiments/
    jplag/
      jplag_runner.py                                    ← this file
      jplag.jar                                          ← download from github.com/jplag/JPlag/releases
      out/
        jplag_runs.csv                                   ← one row per run (params + metrics)
        JPlag-Threshold-0.50-MinTokens-5_results.csv    ← predictions for that run
        case-01_report.zip                               ← raw JPlag report per case
        ...

Usage:
  python jplag_runner.py
  python jplag_runner.py --min-tokens 3
  python jplag_runner.py --threshold 0.7
  python jplag_runner.py --threshold 0.6 --min-tokens 3 --cases case-01 case-02
  python jplag_runner.py --jar /path/to/other.jar
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np

DATASET_ROOT = Path(__file__).parent.parent / "IR-Plag-Dataset"
OUT_DIR = Path(__file__).parent / "out"
JPLAG_JAR_DEFAULT = Path(__file__).parent / "jplag.jar"
JPLAG_TIMEOUT_S = 300  # per case

RUNS_CSV = OUT_DIR / "jplag_runs.csv"
RUNS_FIELDNAMES = [
    "run_name", "min_tokens", "threshold", "similarity_metric",
    "tp", "fp", "tn", "fn",
    "precision", "recall", "f1", "accuracy", "auc", "mcc",
    "predictions_csv",
]

PREDICTIONS_FIELDNAMES = ["case", "level", "submission_id", "similarity", "is_plagiarized", "predicted_plag"]


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
# Dataset helpers
# ---------------------------------------------------------------------------

def find_java_files(directory: Path) -> list[Path]:
    return list(directory.rglob("*.java"))


def copy_submission(src: Path, dest: Path) -> int:
    """Copy all .java files from src into dest. Returns number of files copied."""
    dest.mkdir(parents=True, exist_ok=True)
    files = find_java_files(src)
    for f in files:
        shutil.copy(f, dest / f.name)
    return len(files)


def prepare_submissions(case_dir: Path, work_dir: Path) -> dict[str, tuple[str, str, bool]]:
    """
    Populate work_dir with one subfolder per submission.
    Returns: {folder_name: (level, submission_id, is_plagiarized)}
    """
    meta: dict[str, tuple[str, str, bool]] = {}

    # Original (the reference file)
    n = copy_submission(case_dir / "original", work_dir / "original")
    if n == 0:
        print(f"  WARNING: no .java files in {case_dir / 'original'}", file=sys.stderr)
    meta["original"] = ("original", "original", False)

    # Plagiarized submissions, one folder per submission
    plag_root = case_dir / "plagiarized"
    for level_dir in sorted(plag_root.iterdir()):
        if not level_dir.is_dir() or level_dir.name.startswith("."):
            continue
        level = level_dir.name  # L1 .. L6
        for sub_dir in sorted(level_dir.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            folder = f"plag_{level}_{sub_dir.name}"
            copy_submission(sub_dir, work_dir / folder)
            meta[folder] = (level, sub_dir.name, True)

    # Non-plagiarized submissions
    nonplag_root = case_dir / "non-plagiarized"
    for sub_dir in sorted(nonplag_root.iterdir()):
        if not sub_dir.is_dir() or sub_dir.name.startswith("."):
            continue
        folder = f"nonplag_{sub_dir.name}"
        copy_submission(sub_dir, work_dir / folder)
        meta[folder] = ("non-plag", sub_dir.name, False)

    return meta


# ---------------------------------------------------------------------------
# JPlag invocation
# ---------------------------------------------------------------------------

def run_jplag(jar: Path, submission_dir: Path, report_stem: Path, min_tokens: int) -> bool:
    """
    Invoke JPlag. report_stem is the output path without extension;
    JPlag will create report_stem.zip (or .jplag depending on version).
    Returns True on success.
    """
    cmd = [
        "java", "-jar", str(jar),
        "--language", "java",
        "--min-tokens", str(min_tokens),
        "--mode", "RUN",
        "--result-file", str(report_stem),
        "--shown-comparisons", "-1",
        str(submission_dir),
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=JPLAG_TIMEOUT_S,
    )
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        print(f"  JPlag output:\n{output[-2000:]}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def _find_report_zip(stem: Path) -> Path | None:
    """Find the report file JPlag actually wrote (tries several extensions)."""
    for ext in (".zip", ".jplag", ""):
        p = stem.with_suffix(ext) if ext else stem
        if p.exists() and zipfile.is_zipfile(p):
            return p
    return None


def _parse_similarity(comp: dict, metric: str = "MAX") -> float:
    """
    Extract a single similarity value from a JPlag comparison dict.

    metric: "MAX" uses the direction that favours the smaller (plagiarising) file.
            "AVG" uses the average of both directions — less aggressive, may
            reduce false positives on structurally similar non-plag submissions.
    """
    sims = comp.get("similarities", {})
    if sims:
        if metric == "AVG":
            for key in ("AVG", "avg", "average"):
                if key in sims:
                    return float(sims[key])
            # Fall back to arithmetic mean of all available values
            return float(sum(sims.values()) / len(sims))
        else:  # MAX (default)
            for key in ("MAX", "max"):
                if key in sims:
                    return float(sims[key])
            return float(max(sims.values()))
    # Older format: flat similarity field, possibly in 0-100 range
    raw = comp.get("similarity", comp.get("percent", 0.0))
    val = float(raw)
    return val / 100.0 if val > 1.0 else val


def extract_similarities(report_stem: Path, metric: str = "MAX") -> dict[tuple[str, str], float]:
    """
    Parse all comparisons from the JPlag report ZIP (v5 format).
    Returns {(subA, subB): similarity} with both orderings stored.
    """
    sims: dict[tuple[str, str], float] = {}

    report_path = _find_report_zip(report_stem)
    if report_path is None:
        print(f"  ERROR: no report ZIP found near {report_stem}", file=sys.stderr)
        return sims

    _skip = {"overview.json", "options.json", "submissionFileIndex.json", "README.txt"}

    with zipfile.ZipFile(report_path) as zf:
        names = set(zf.namelist())

        comp_filenames: set[str] = set()
        if "overview.json" in names:
            with zf.open("overview.json") as f:
                overview = json.load(f)

            index = overview.get("submission_ids_to_comparison_file_name", {})
            for inner in index.values():
                comp_filenames.update(inner.values())

            for comp in overview.get("top_comparisons", []):
                sub_a = comp["first_submission"]
                sub_b = comp["second_submission"]
                sim = _parse_similarity(comp, metric)
                sims[(sub_a, sub_b)] = sim
                sims[(sub_b, sub_a)] = sim

        if not comp_filenames:
            comp_filenames = {
                n for n in names
                if n.endswith(".json") and "/" not in n and n not in _skip
            }

        for fname in comp_filenames:
            if fname not in names:
                continue
            with zf.open(fname) as f:
                comp = json.load(f)
            sub_a = comp.get("id1", comp.get("first_submission", ""))
            sub_b = comp.get("id2", comp.get("second_submission", ""))
            if not sub_a or not sub_b:
                continue
            sim = _parse_similarity(comp, metric)
            sims[(sub_a, sub_b)] = sim
            sims[(sub_b, sub_a)] = sim

    return sims


# ---------------------------------------------------------------------------
# Runs CSV helpers
# ---------------------------------------------------------------------------

def append_run(run_row: dict) -> None:
    """Append (or overwrite if run_name already exists) a row in jplag_runs.csv."""
    existing: list[dict] = []
    if RUNS_CSV.exists():
        with open(RUNS_CSV, newline="") as f:
            existing = list(csv.DictReader(f))

    # Replace existing row with the same run_name, or append
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run JPlag over IR-Plag-Dataset and write results CSV."
    )
    parser.add_argument("--jar", type=Path, default=JPLAG_JAR_DEFAULT,
                        help="Path to JPlag fat JAR (default: experiments/jplag/jplag.jar)")
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT,
                        help="Path to IR-Plag-Dataset directory")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Similarity threshold for predicted_plag (default: 0.5)")
    parser.add_argument("--min-tokens", type=int, default=5,
                        help="Minimum token length for JPlag matching (default: 5)")
    parser.add_argument("--similarity-metric", choices=["MAX", "AVG"], default="MAX",
                        help="Similarity metric extracted from JPlag report (default: MAX). "
                             "MAX favours the smaller plagiarising file; "
                             "AVG uses both directions and may reduce false positives.")
    parser.add_argument("--cases", nargs="+", default=None, metavar="CASE",
                        help="Run only these cases, e.g. --cases case-01 case-03")
    args = parser.parse_args()

    run_name = (f"JPlag-Threshold-{args.threshold:.2f}"
                f"-MinTokens-{args.min_tokens}"
                f"-Metric-{args.similarity_metric}")
    predictions_filename = f"{run_name}_results.csv"
    predictions_csv = OUT_DIR / predictions_filename

    print("=" * 60)
    print(f"Run: {run_name}")
    print(f"  threshold          = {args.threshold}")
    print(f"  min-tokens         = {args.min_tokens}")
    print(f"  similarity-metric  = {args.similarity_metric}")
    print(f"  output             = {predictions_csv}")
    print("=" * 60)

    if not args.jar.exists():
        sys.exit(
            f"ERROR: JPlag JAR not found at {args.jar}\n"
            "Download the fat JAR from https://github.com/jplag/JPlag/releases\n"
            "and place it at experiments/jplag/jplag.jar (or pass --jar <path>)."
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

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            submission_dir = tmp_path / "submissions"
            submission_dir.mkdir()
            report_stem = tmp_path / "report"

            meta = prepare_submissions(case_dir, submission_dir)
            total_subs = len(meta) - 1  # exclude "original"
            print(f"  Submissions prepared: {total_subs} (+ original)")

            try:
                ok = run_jplag(args.jar, submission_dir, report_stem, args.min_tokens)
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT after {JPLAG_TIMEOUT_S}s — skipping {case_name}", file=sys.stderr)
                continue

            if not ok:
                print(f"  Skipping {case_name} due to JPlag error.")
                continue

            # Persist raw report ZIP before the temp dir is deleted
            raw_zip = _find_report_zip(report_stem)
            if raw_zip is not None:
                dest = OUT_DIR / f"{case_name}_report.zip"
                shutil.copy(raw_zip, dest)
                print(f"  Raw report saved → {dest}")

            sims = extract_similarities(report_stem, args.similarity_metric)
            n_pairs = len(sims) // 2
            print(f"  Parsed {n_pairs} comparison pairs from report")

            matched = 0
            for folder, (level, sub_id, is_plag) in sorted(meta.items()):
                if folder == "original":
                    continue
                sim = sims.get(("original", folder), sims.get((folder, "original"), 0.0))
                if sim > 0.0:
                    matched += 1
                predicted = sim >= args.threshold
                rows.append({
                    "case": case_name,
                    "level": level,
                    "submission_id": sub_id,
                    "similarity": round(sim, 4),
                    "is_plagiarized": is_plag,
                    "predicted_plag": predicted,
                })
                flag = "PLAG" if is_plag else "    "
                print(f"  [{flag}] {folder:<25} sim={sim:.4f}  pred={'Y' if predicted else 'N'}")

            print(f"  Similarity found for {matched}/{total_subs} submissions vs original")

    if not rows:
        print("\nNo results produced. Check JPlag errors above.", file=sys.stderr)
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
        "run_name":          run_name,
        "min_tokens":        args.min_tokens,
        "threshold":         args.threshold,
        "similarity_metric": args.similarity_metric,
        **m,
        "predictions_csv":   predictions_filename,
    }
    append_run(run_row)

    print(f"\nMetrics @ threshold={args.threshold:.2f}:")
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  Accuracy={m['accuracy']:.4f}  AUC={m['auc']:.4f}  MCC={m['mcc']:.4f}")
    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
