#!/usr/bin/env python3
"""
jplag_runner.py — Evaluate JPlag over IR-Plag-Dataset.

Strategy per case:
  1. Copy original + all plagiarized + non-plagiarized submissions into a
     temp directory (each in its own subfolder, uniquely named).
  2. Run JPlag once with --shown-comparisons -1 (store all pairs).
  3. Parse the report ZIP: extract similarity between "original" and each
     other submission (use MAX metric, as per IR-Plag evaluation protocol).
  4. Emit one CSV row per submission.

Layout:
  experiments/
    jplag/
      jplag_runner.py         ← this file
      jplag.jar               ← download from github.com/jplag/JPlag/releases
      out/
        jplag_results.csv     ← parsed results (standard CSV format)
        case-01_report.zip    ← raw JPlag report per case (for manual inspection)
        case-02_report.zip
        ...

Usage:
  python jplag_runner.py
  python jplag_runner.py --threshold 0.7 --cases case-01 case-02
  python jplag_runner.py --jar /path/to/other.jar --output /path/to/out.csv
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

DATASET_ROOT = Path(__file__).parent.parent / "IR-Plag-Dataset"
OUT_DIR = Path(__file__).parent / "out"
OUTPUT_CSV = OUT_DIR / "jplag_results.csv"
JPLAG_JAR_DEFAULT = Path(__file__).parent / "jplag.jar"
MIN_TOKENS = 5
JPLAG_TIMEOUT_S = 300  # per case


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

def run_jplag(jar: Path, submission_dir: Path, report_stem: Path) -> bool:
    """
    Invoke JPlag. report_stem is the output path without extension;
    JPlag will create report_stem.zip (or .jplag depending on version).
    Returns True on success.
    """
    cmd = [
        "java", "-jar", str(jar),
        "--language", "java",
        "--min-tokens", str(MIN_TOKENS),
        "--mode", "RUN",           # v5: RUN | VIEW | RUN_AND_VIEW (no RUN_AND_EXIT)
        "--result-file", str(report_stem),
        "--shown-comparisons", "-1",   # store all pairs, not just top-N
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


def _parse_similarity(comp: dict) -> float:
    """
    Extract a single similarity value from a JPlag comparison dict.
    Prefers MAX (one direction that favours the smaller, plagiarising file),
    falls back to AVG, then to any value present.
    """
    sims = comp.get("similarities", {})
    if sims:
        for key in ("MAX", "max", "AVG", "avg", "average"):
            if key in sims:
                return float(sims[key])
        return float(max(sims.values()))
    # Older format: flat similarity field, possibly in 0-100 range
    raw = comp.get("similarity", comp.get("percent", 0.0))
    val = float(raw)
    return val / 100.0 if val > 1.0 else val


def extract_similarities(report_stem: Path) -> dict[tuple[str, str], float]:
    """
    Parse all comparisons from the JPlag report ZIP (v5 format).
    Returns {(subA, subB): similarity} with both orderings stored.

    v5 ZIP layout:
      overview.json                         ← has top_comparisons + index
      <subA>-<subB>.json                    ← individual comparison files (root level)
      files/<sub>/...                       ← submission sources (ignored)
    """
    sims: dict[tuple[str, str], float] = {}

    report_path = _find_report_zip(report_stem)
    if report_path is None:
        print(f"  ERROR: no report ZIP found near {report_stem}", file=sys.stderr)
        return sims

    _skip = {"overview.json", "options.json", "submissionFileIndex.json", "README.txt"}

    with zipfile.ZipFile(report_path) as zf:
        names = set(zf.namelist())

        # Collect all comparison filenames from the authoritative index in overview
        comp_filenames: set[str] = set()
        if "overview.json" in names:
            with zf.open("overview.json") as f:
                overview = json.load(f)

            # submission_ids_to_comparison_file_name: {subB: {subA: filename}, ...}
            index = overview.get("submission_ids_to_comparison_file_name", {})
            for inner in index.values():
                comp_filenames.update(inner.values())

            # Also seed from top_comparisons (covers cases where index is absent)
            for comp in overview.get("top_comparisons", []):
                sub_a = comp["first_submission"]
                sub_b = comp["second_submission"]
                sim = _parse_similarity(comp)
                sims[(sub_a, sub_b)] = sim
                sims[(sub_b, sub_a)] = sim

        # Fall back: treat any root-level .json not in the known metadata set as a comparison
        if not comp_filenames:
            comp_filenames = {
                n for n in names
                if n.endswith(".json") and "/" not in n and n not in _skip
            }

        # Parse each comparison file (id1/id2 keys in v5)
        for fname in comp_filenames:
            if fname not in names:
                continue
            with zf.open(fname) as f:
                comp = json.load(f)
            sub_a = comp.get("id1", comp.get("first_submission", ""))
            sub_b = comp.get("id2", comp.get("second_submission", ""))
            if not sub_a or not sub_b:
                continue
            sim = _parse_similarity(comp)
            sims[(sub_a, sub_b)] = sim
            sims[(sub_b, sub_a)] = sim

    return sims


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run JPlag over IR-Plag-Dataset and write results CSV."
    )
    parser.add_argument("--jar", type=Path, default=JPLAG_JAR_DEFAULT,
                        help="Path to JPlag fat JAR (default: experiments/jplag.jar)")
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT,
                        help="Path to IR-Plag-Dataset directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV,
                        help="Output CSV path (default: experiments/jplag/out/jplag_results.csv); "
                             "raw ZIP reports are saved alongside it")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Similarity threshold for predicted_plag (default: 0.5; "
                             "use evaluate.py to find optimal threshold)")
    parser.add_argument("--cases", nargs="+", default=None, metavar="CASE",
                        help="Run only these cases, e.g. --cases case-01 case-03")
    args = parser.parse_args()

    if not args.jar.exists():
        sys.exit(
            f"ERROR: JPlag JAR not found at {args.jar}\n"
            "Download the fat JAR from https://github.com/jplag/JPlag/releases\n"
            "and place it at experiments/jplag.jar (or pass --jar <path>)."
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

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

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
                ok = run_jplag(args.jar, submission_dir, report_stem)
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT after {JPLAG_TIMEOUT_S}s — skipping {case_name}", file=sys.stderr)
                continue

            if not ok:
                print(f"  Skipping {case_name} due to JPlag error.")
                continue

            # Persist raw report ZIP before the temp dir is deleted
            raw_zip = _find_report_zip(report_stem)
            if raw_zip is not None:
                dest = out_dir / f"{case_name}_report.zip"
                shutil.copy(raw_zip, dest)
                print(f"  Raw report saved → {dest}")

            sims = extract_similarities(report_stem)
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

    fieldnames = ["case", "level", "submission_id", "similarity", "is_plagiarized", "predicted_plag"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows written to {args.output}")
    print(f"Raw reports in {out_dir}/")


if __name__ == "__main__":
    main()
