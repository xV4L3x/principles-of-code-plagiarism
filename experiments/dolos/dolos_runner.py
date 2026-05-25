#!/usr/bin/env python3
"""
dolos_runner.py — Evaluate Dolos over IR-Plag-Dataset.

Normal mode (default):
  Runs Dolos once per case with a fixed --kgram-length and --metric, writes
  the standard CSV and preserves the raw Dolos report directory per case.

Sweep mode (--sweep):
  Re-runs Dolos for every value in --sweep-kgrams (different -k values require
  actual re-execution), then evaluates all (kgram × metric × threshold)
  combinations and picks the one with the highest F1.
  Writes out/sweep_results.csv and out/sweep_best.txt.

Layout:
  experiments/
    dolos/
      dolos_runner.py         ← this file
      node_modules/           ← npm install @dodona/dolos (requires Node 22)
      .nvmrc                  ← pins Node 22
      out/
        dolos_results.csv         ← normal mode output
        case-01_report/           ← raw Dolos CSV report per case
        sweep_results.csv         ← sweep mode: all combinations
        sweep_best.txt            ← sweep mode: human-readable summary

Usage:
  python dolos_runner.py
  python dolos_runner.py --kgram 10 --metric MAX --threshold 0.7
  python dolos_runner.py --sweep
  python dolos_runner.py --sweep --sweep-kgrams 5 10 15 23
"""

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from itertools import product
from pathlib import Path

DATASET_ROOT = Path(__file__).parent.parent / "IR-Plag-Dataset"
OUT_DIR = Path(__file__).parent / "out"
OUTPUT_CSV = OUT_DIR / "dolos_results.csv"
DOLOS_BIN = Path(__file__).parent / "node_modules" / ".bin" / "dolos"

DEFAULT_KGRAM = 23
DEFAULT_WINDOW = 17
DEFAULT_METRIC = "COMBINED"    # COMBINED | MAX | AVG | SUB_IN_ORIG | ORIG_IN_SUB
DEFAULT_THRESHOLD = 0.5

SWEEP_KGRAMS = [5, 10, 15, 23]
SWEEP_METRICS = ["COMBINED", "MAX", "AVG", "SUB_IN_ORIG", "ORIG_IN_SUB"]
SWEEP_THRESHOLDS = [round(v / 100, 2) for v in range(5, 100, 5)]  # 0.05..0.95

DOLOS_TIMEOUT_S = 120


# ---------------------------------------------------------------------------
# Node / Dolos binary helpers
# ---------------------------------------------------------------------------

def _find_node() -> str:
    """
    Return the path to a Node.js binary that can run Dolos.
    Prefers nvm-managed Node 22 (required for tree-sitter native module),
    falls back to system node.
    """
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
        "-L", "0",           # no result limit — return all pairs
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
    """
    Map a file path from the Dolos report back to a submission folder_key.
    .../case-01/plagiarized/L3/02/file.java  →  'plag_L3_02'
    .../case-01/non-plagiarized/04/file.java →  'nonplag_04'
    .../case-01/original/file.java           →  'original'
    Returns None if the path doesn't match known structure.
    """
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
    # Dolos creates the output directory itself; it errors if the dir already exists.
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
    """Returns {file_path: amountOfKgrams}."""
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
    """Returns list of raw row dicts from pairs.csv."""
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
    sub_frac = sub_covered / sub_total if sub_total > 0 else 0.0
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
    submission. Only pairs that involve at least one file from 'original/'
    are considered. For submissions with multiple files, the max similarity
    across all their file-original pairs is returned.
    """
    file_kgrams = _parse_files_csv(report_dir)
    pairs = _parse_pairs_csv(report_dir)

    orig_paths: set[str] = {
        p for p in file_kgrams if _path_to_folder_key(p, case_dir) == "original"
    }

    sub_sims: dict[str, float] = {}

    for row in pairs:
        left_path = row.get("leftFilePath", "")
        right_path = row.get("rightFilePath", "")

        if left_path in orig_paths:
            orig_path, sub_path = left_path, right_path
            orig_covered = int(row.get("leftCovered", 0))
            sub_covered = int(row.get("rightCovered", 0))
        elif right_path in orig_paths:
            orig_path, sub_path = right_path, left_path
            orig_covered = int(row.get("rightCovered", 0))
            sub_covered = int(row.get("leftCovered", 0))
        else:
            continue

        folder_key = _path_to_folder_key(sub_path, case_dir)
        if folder_key is None or folder_key == "original":
            continue

        orig_total = file_kgrams.get(orig_path, 0)
        sub_total = file_kgrams.get(sub_path, 0)
        combined = float(row.get("similarity", 0.0))

        sim = apply_metric(orig_covered, sub_covered, orig_total, sub_total, combined, metric)
        sub_sims[folder_key] = max(sub_sims.get(folder_key, 0.0), sim)

    return sub_sims


def extract_raw_pair_data(
    report_dir: Path,
    case_dir: Path,
) -> list[dict]:
    """
    Like extract_submission_sims but returns the raw (orig_covered,
    sub_covered, orig_total, sub_total, combined_sim) per pair — used in
    sweep mode to avoid recomputing metrics for every threshold.
    Aggregates max over multiple files per submission using COMBINED.
    """
    file_kgrams = _parse_files_csv(report_dir)
    pairs = _parse_pairs_csv(report_dir)

    orig_paths: set[str] = {
        p for p in file_kgrams if _path_to_folder_key(p, case_dir) == "original"
    }

    # best_data[folder_key] = best raw record (maximise combined_sim)
    best_data: dict[str, dict] = {}

    for row in pairs:
        left_path = row.get("leftFilePath", "")
        right_path = row.get("rightFilePath", "")

        if left_path in orig_paths:
            orig_path, sub_path = left_path, right_path
            orig_covered = int(row.get("leftCovered", 0))
            sub_covered = int(row.get("rightCovered", 0))
        elif right_path in orig_paths:
            orig_path, sub_path = right_path, left_path
            orig_covered = int(row.get("rightCovered", 0))
            sub_covered = int(row.get("leftCovered", 0))
        else:
            continue

        folder_key = _path_to_folder_key(sub_path, case_dir)
        if folder_key is None or folder_key == "original":
            continue

        combined = float(row.get("similarity", 0.0))
        existing = best_data.get(folder_key)
        if existing is None or combined > existing["combined_sim"]:
            best_data[folder_key] = {
                "orig_covered": orig_covered,
                "sub_covered": sub_covered,
                "orig_total": file_kgrams.get(orig_path, 0),
                "sub_total": file_kgrams.get(sub_path, 0),
                "combined_sim": combined,
            }

    return best_data


# ---------------------------------------------------------------------------
# Evaluation helpers (sweep mode)
# ---------------------------------------------------------------------------

def compute_metrics(
    is_plag_list: list[bool],
    similarities: list[float],
    threshold: float,
) -> dict:
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

        subs = collect_case_files(case_dir)
        all_java = orig_files + [f for _, _, _, files in subs.values() for f in files]
        print(f"  Files: {len(all_java)} total  |  kgram={args.kgram}  metric={args.metric}")

        report_dir = out_dir / f"{case_name}_report"

        ok = run_dolos(all_java, report_dir, args.kgram, args.window)
        if not ok:
            print(f"  Skipping {case_name} due to Dolos error.")
            continue

        sub_sims = extract_submission_sims(report_dir, case_dir, args.metric)
        matched = sum(1 for v in sub_sims.values() if v > 0.0)
        print(f"  Similarity found for {matched}/{len(subs)} submissions vs original")

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

    _write_csv(args.output, rows)
    print(f"\nDone. {len(rows)} rows → {args.output}")
    print(f"Raw reports in {out_dir}/")


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace) -> None:
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = _get_cases(args)
    kgrams = args.sweep_kgrams

    # data[kgram] = list of {case, level, sub_id, is_plag, orig_covered, sub_covered,
    #                         orig_total, sub_total, combined_sim}
    data: dict[int, list[dict]] = {k: [] for k in kgrams}

    for kgram in kgrams:
        print(f"\n{'='*50}\nSweeping kgram={kgram}\n{'='*50}")
        for case_dir in cases:
            case_name = case_dir.name
            orig_files = find_java_files(case_dir / "original")
            if not orig_files:
                continue
            subs = collect_case_files(case_dir)
            all_java = orig_files + [f for _, _, _, files in subs.values() for f in files]
            print(f"  {case_name}  ({len(subs)} submissions)", flush=True)

            with tempfile.TemporaryDirectory() as tmp:
                report_dir = Path(tmp) / "report"
                ok = run_dolos(all_java, report_dir, kgram, args.window)
                if not ok:
                    continue
                best_data = extract_raw_pair_data(report_dir, case_dir)

            for key, (level, sub_id, is_plag, _) in subs.items():
                raw = best_data.get(key, {})
                data[kgram].append({
                    "case": case_name, "level": level, "submission_id": sub_id,
                    "is_plagiarized": is_plag,
                    "orig_covered": raw.get("orig_covered", 0),
                    "sub_covered": raw.get("sub_covered", 0),
                    "orig_total": raw.get("orig_total", 0),
                    "sub_total": raw.get("sub_total", 0),
                    "combined_sim": raw.get("combined_sim", 0.0),
                })

    print(f"\n{'='*50}\nEvaluating combinations...\n{'='*50}")
    combo_rows: list[dict] = []

    for kgram, metric, threshold in product(kgrams, SWEEP_METRICS, SWEEP_THRESHOLDS):
        rows = data[kgram]
        if not rows:
            continue
        is_plag_list = [r["is_plagiarized"] for r in rows]
        sims = [
            apply_metric(
                r["orig_covered"], r["sub_covered"],
                r["orig_total"], r["sub_total"],
                r["combined_sim"], metric,
            )
            for r in rows
        ]
        m = compute_metrics(is_plag_list, sims, threshold)
        combo_rows.append({
            "kgram": kgram, "metric": metric, "threshold": threshold,
            **{k: round(v, 4) for k, v in m.items()},
        })

    combo_rows.sort(key=lambda r: r["f1"], reverse=True)

    sweep_csv = out_dir / "sweep_results.csv"
    sweep_fields = ["kgram", "metric", "threshold", "f1", "accuracy", "precision", "recall",
                    "tp", "fp", "tn", "fn"]
    with open(sweep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sweep_fields)
        w.writeheader()
        w.writerows(combo_rows)
    print(f"Sweep CSV → {sweep_csv}  ({len(combo_rows)} combinations)")

    header = f"{'kgram':>6}  {'metric':<12}  {'threshold':>9}  {'F1':>6}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for row in combo_rows[:20]:
        lines.append(
            f"{row['kgram']:>6}  {row['metric']:<12}  {row['threshold']:>9.2f}"
            f"  {row['f1']:>6.4f}  {row['accuracy']:>6.4f}"
            f"  {row['precision']:>6.4f}  {row['recall']:>6.4f}"
        )
    lines.append(sep)
    best = combo_rows[0]
    lines.append(
        f"\nBEST  →  kgram={best['kgram']}  metric={best['metric']}"
        f"  threshold={best['threshold']:.2f}"
        f"  F1={best['f1']:.4f}  Accuracy={best['accuracy']:.4f}"
    )

    summary = "\n".join(lines)
    print("\n" + summary)

    best_path = out_dir / "sweep_best.txt"
    best_path.write_text(summary + "\n")
    print(f"\nSummary saved → {best_path}")
    print(
        f"\nTo run the winner: python dolos_runner.py "
        f"--kgram {best['kgram']} --metric {best['metric']} "
        f"--threshold {best['threshold']:.2f}"
    )


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
        description="Run Dolos over IR-Plag-Dataset — normal or sweep mode."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT,
                        help="Path to IR-Plag-Dataset directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV,
                        help="Output CSV (normal mode). Reports and sweep files go alongside it.")
    parser.add_argument("--cases", nargs="+", default=None, metavar="CASE",
                        help="Run only these cases, e.g. --cases case-01 case-03")

    # Normal mode options
    parser.add_argument("--kgram", type=int, default=DEFAULT_KGRAM,
                        help=f"k-gram length for fingerprinting (default: {DEFAULT_KGRAM})")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"Window size in kgrams for winnowing (default: {DEFAULT_WINDOW})")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        choices=SWEEP_METRICS,
                        help="Similarity metric — COMBINED (Dolos default), MAX, AVG, "
                             "SUB_IN_ORIG, ORIG_IN_SUB (default: COMBINED)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity threshold for predicted_plag (default: {DEFAULT_THRESHOLD})")

    # Sweep mode
    parser.add_argument("--sweep", action="store_true",
                        help="Try all (kgram × metric × threshold) combinations and report the best by F1")
    parser.add_argument("--sweep-kgrams", nargs="+", type=int, default=SWEEP_KGRAMS,
                        metavar="K",
                        help=f"kgram values to sweep (default: {SWEEP_KGRAMS})")

    args = parser.parse_args()

    if not DOLOS_BIN.exists():
        sys.exit(
            f"ERROR: Dolos not found at {DOLOS_BIN}\n"
            "Run: cd experiments/dolos && npm install @dodona/dolos\n"
            "Requires Node.js 22 — use nvm: nvm use 22"
        )
    if not args.dataset.exists():
        sys.exit(f"ERROR: Dataset not found at {args.dataset}")

    if args.sweep:
        run_sweep(args)
    else:
        run_normal(args)


if __name__ == "__main__":
    main()
