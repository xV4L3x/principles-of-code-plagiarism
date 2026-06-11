#!/usr/bin/env python3
"""
plaggie_runner.py — Evaluate Plaggie over IR-Plag-Dataset.

Each invocation is a named run encoding its parameters. Results are written to:
  out/<run_name>_results.csv         — per-submission predictions
  out/plaggie_runs.csv               — one summary row per run (metrics + params)

Score caching: directional (orig_in_sub, sub_in_orig) scores are cached per
(case, min_tokens) in out/case-XX-mintokens-T_scores.csv. Runs that share the
same min_tokens but differ only in metric or threshold reuse the cache
automatically. Pass --force to re-run Plaggie even when a cache exists.

Strategy per case:
  1. Copy original + all plagiarized + non-plagiarized into a temp directory.
  2. Run Plaggie once per case with -s0.0 -nohtml; capture stdout.
  3. Parse stdout: extract (simA, simB) for each pair involving "original".
  4. Save directional scores to cache; apply metric; write predictions row.

Build note:
  Plaggie is distributed as source only (SourceForge). The --build flag
  downloads, patches (removes a hardcoded args line), compiles, and jars it.
  Run once before first use:  python plaggie_runner.py --build

Usage:
  python plaggie_runner.py --build
  python plaggie_runner.py
  python plaggie_runner.py --min-tokens 5 --metric ORIG_IN_SUB --threshold 0.6
  python plaggie_runner.py --min-tokens 3 --metric AVG   # reuses cached scores
  python plaggie_runner.py --min-tokens 3 --force        # re-runs Plaggie
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

DATASET_ROOT = Path(__file__).parent.parent / "IR-Plag-Dataset"
OUT_DIR      = Path(__file__).parent / "out"
JAR_PATH     = Path(__file__).parent / "plaggie.jar"
SRC_DIR      = Path(__file__).parent / "plaggie-src"

RUNS_CSV = OUT_DIR / "plaggie_runs.csv"
RUNS_FIELDNAMES = [
    "run_name", "min_tokens", "threshold", "metric",
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

DEFAULT_MIN_TOKENS = 3
DEFAULT_METRIC     = "MAX"
DEFAULT_THRESHOLD  = 0.5
PLAGGIE_METRICS    = ["MAX", "AVG", "ORIG_IN_SUB", "SUB_IN_ORIG", "PRODUCT"]
PLAGGIE_TIMEOUT_S  = 300

DOWNLOAD_URL  = "https://sourceforge.net/projects/plaggie/files/latest/download"
HARDCODED_PAT = re.compile(
    r'\s*args\s*=\s*new\s+String\[\]\s*\{[^}]*LawanSubba[^}]*\}\s*;\s*\n'
)

PROPERTIES_TEMPLATE = """\
plag.parser.plaggie.codeTokenizer=plag.parser.java.JavaTokenizer
plag.parser.plaggie.filenameFilter=plag.parser.java.JavaFilenameFilter
plag.parser.plaggie.minimumMatchLength={min_tokens}
plag.parser.plaggie.minimumSubmissionSimilarityValue=0.0
plag.parser.plaggie.maximumDetectionResultsToReport=100000
plag.parser.plaggie.useRecursive=true
plag.parser.plaggie.severalSubmissionDirectories=false
plag.parser.plaggie.submissionDirectory=round1
plag.parser.plaggie.excludeInterfaces=false
plag.parser.plaggie.excludeFiles=
plag.parser.plaggie.excludeSubdirectories=
plag.parser.plaggie.templates=
plag.parser.plaggie.minimumFileSimilarityValueToReport=0.0
plag.parser.plaggie.htmlReport=false
plag.parser.plaggie.htmlDir=html
plag.parser.plaggie.blacklistFile=
plag.parser.plaggie.showAllBlacklistedResults=false
plag.parser.plaggie.printTokenLists=false
plag.parser.plaggie.fileDetectionReports=none
plag.parser.plaggie.debugMessages=false
plag.parser.plaggie.cacheTokenLists=true
plag.parser.plaggie.createResultFile=false
plag.parser.plaggie.readResultsFromFile=false
plag.parser.plaggie.resultFile=results.data
"""


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def build_jar() -> None:
    print("Downloading Plaggie from SourceForge...", flush=True)
    tmp_zip = Path(tempfile.mktemp(suffix=".zip"))
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, tmp_zip)
    except Exception as exc:
        sys.exit(f"ERROR: download failed: {exc}")

    print(f"Extracting to {SRC_DIR}...", flush=True)
    if SRC_DIR.exists():
        shutil.rmtree(SRC_DIR)
    SRC_DIR.mkdir(parents=True)
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(SRC_DIR)
    tmp_zip.unlink(missing_ok=True)

    plaggie_main = SRC_DIR / "src/plag/parser/plaggie/Plaggie.java"
    if not plaggie_main.exists():
        sys.exit(f"ERROR: Plaggie.java not found at {plaggie_main}")
    src_text = plaggie_main.read_text(encoding="latin-1")
    patched, n = HARDCODED_PAT.subn("", src_text)
    if n:
        plaggie_main.write_text(patched, encoding="latin-1")
        print(f"  Patched Plaggie.java ({n} hardcoded-args line removed)")
    else:
        print("  Plaggie.java already clean (no hardcoded-args pattern found)")

    print("Compiling Plaggie...", flush=True)
    bin_dir = SRC_DIR / "bin"
    bin_dir.mkdir(exist_ok=True)
    java_files = [str(f) for f in (SRC_DIR / "src").rglob("*.java")]
    sources_file = SRC_DIR / "sources.txt"
    sources_file.write_text("\n".join(java_files))
    result = subprocess.run(
        ["javac", "-encoding", "ISO-8859-1",
         "-sourcepath", str(SRC_DIR / "src"),
         "-d", str(bin_dir), f"@{sources_file}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        sys.exit("ERROR: compilation failed")
    print(f"  Compiled {len(java_files)} source files")

    print(f"Creating {JAR_PATH}...", flush=True)
    result = subprocess.run(
        ["jar", "cf", str(JAR_PATH), "-C", str(bin_dir), "."],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit("ERROR: jar creation failed")
    print(f"  Created {JAR_PATH} ({JAR_PATH.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _find_java_files(directory: Path) -> list[Path]:
    return list(directory.rglob("*.java"))


def _copy_submission(src: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    files = _find_java_files(src)
    for f in files:
        shutil.copy(f, dest / f.name)
    return len(files)


def prepare_submissions(
    case_dir: Path, submissions_dir: Path
) -> dict[str, tuple[str, str, bool]]:
    """
    Populate submissions_dir with one subfolder per submission.
    Returns {folder_name: (level, submission_id, is_plagiarized)}.
    """
    meta: dict[str, tuple[str, str, bool]] = {}

    n = _copy_submission(case_dir / "original", submissions_dir / "original")
    if n == 0:
        print(f"  WARNING: no .java files in {case_dir / 'original'}", file=sys.stderr)
    meta["original"] = ("original", "original", False)

    for level_dir in sorted((case_dir / "plagiarized").iterdir()):
        if not level_dir.is_dir() or level_dir.name.startswith("."):
            continue
        level = level_dir.name
        for sub_dir in sorted(level_dir.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            folder = f"plag_{level}_{sub_dir.name}"
            _copy_submission(sub_dir, submissions_dir / folder)
            meta[folder] = (level, sub_dir.name, True)

    for sub_dir in sorted((case_dir / "non-plagiarized").iterdir()):
        if not sub_dir.is_dir() or sub_dir.name.startswith("."):
            continue
        folder = f"nonplag_{sub_dir.name}"
        _copy_submission(sub_dir, submissions_dir / folder)
        meta[folder] = ("non-plag", sub_dir.name, False)

    return meta


# ---------------------------------------------------------------------------
# Plaggie invocation
# ---------------------------------------------------------------------------

def run_plaggie(submissions_dir: Path, work_dir: Path) -> tuple[bool, str]:
    cmd = [
        "java", "-cp", str(JAR_PATH),
        "plag.parser.plaggie.Plaggie",
        "-s0.0", "-nohtml",
        str(submissions_dir),
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=PLAGGIE_TIMEOUT_S, cwd=str(work_dir),
    )
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        print(f"  Plaggie error:\n{output[-2000:]}", file=sys.stderr)
        return False, ""
    return True, result.stdout


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_SIM_A_RE = re.compile(r"^Similarity A:([\d.]+)")
_SIM_B_RE = re.compile(r"^Similarity B:([\d.]+)")
_SEP_RE   = re.compile(r"^[=\-]{8,}")


def _submission_key(file_path: str, submissions_dir: Path) -> str | None:
    try:
        rel = Path(file_path).relative_to(submissions_dir)
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def parse_output(
    stdout: str, submissions_dir: Path
) -> dict[str, tuple[float, float]]:
    """
    Parse Plaggie stdout. Returns {folder_name: (orig_in_sub, sub_in_orig)}
    for all pairs involving 'original'.
    """
    pairs: dict[str, tuple[float, float]] = {}
    lines = stdout.splitlines()
    i = 0

    while i < len(lines):
        if not lines[i].startswith("=" * 8):
            i += 1
            continue

        sim_a = sim_b = 0.0
        files_a: list[str] = []
        files_b: list[str] = []
        section: str | None = None
        i += 1

        while i < len(lines) and not lines[i].startswith("=" * 8):
            line = lines[i]
            m = _SIM_A_RE.match(line)
            if m:
                sim_a = float(m.group(1))
                i += 1
                continue
            m = _SIM_B_RE.match(line)
            if m:
                sim_b = float(m.group(1))
                i += 1
                continue
            if line.startswith("Files in submission A:"):
                section = "A"; i += 1; continue
            if line.startswith("Files in submission B:"):
                section = "B"; i += 1; continue
            if _SEP_RE.match(line):
                i += 1; continue
            stripped = line.strip()
            if stripped:
                if section == "A":
                    files_a.append(stripped)
                elif section == "B":
                    files_b.append(stripped)
            i += 1

        key_a = next((_submission_key(fp, submissions_dir) for fp in files_a), None)
        key_b = next((_submission_key(fp, submissions_dir) for fp in files_b), None)
        if key_a is None or key_b is None:
            continue

        if key_a == "original":
            pairs[key_b] = (sim_a, sim_b)
        elif key_b == "original":
            pairs[key_a] = (sim_b, sim_a)

    return pairs


def apply_metric(orig_in_sub: float, sub_in_orig: float, metric: str) -> float:
    if metric == "MAX":
        return max(orig_in_sub, sub_in_orig)
    if metric == "AVG":
        return (orig_in_sub + sub_in_orig) / 2.0
    if metric == "ORIG_IN_SUB":
        return orig_in_sub
    if metric == "SUB_IN_ORIG":
        return sub_in_orig
    if metric == "PRODUCT":
        return orig_in_sub * sub_in_orig
    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Score cache  (keyed by case + min_tokens)
# ---------------------------------------------------------------------------

def cache_path(case_name: str, min_tokens: int) -> Path:
    return OUT_DIR / f"{case_name}-mintokens-{min_tokens}_scores.csv"


def load_score_cache(case_name: str, min_tokens: int) -> list[dict] | None:
    p = cache_path(case_name, min_tokens)
    if not p.exists():
        return None
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def save_score_cache(case_name: str, min_tokens: int, rows: list[dict]) -> None:
    p = cache_path(case_name, min_tokens)
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
        description="Run Plaggie over IR-Plag-Dataset (run-based, one execution per invocation)."
    )
    parser.add_argument("--build", action="store_true",
                        help="Download, compile, and create plaggie.jar, then exit")
    parser.add_argument("--dataset",    type=Path,  default=DATASET_ROOT)
    parser.add_argument("--cases",      nargs="+",  default=None, metavar="CASE")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity cutoff for predicted_plag (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--min-tokens", type=int,   default=DEFAULT_MIN_TOKENS,
                        help=f"Minimum matching token sequence length (default: {DEFAULT_MIN_TOKENS})")
    parser.add_argument("--metric",     default=DEFAULT_METRIC, choices=PLAGGIE_METRICS,
                        help="Similarity metric (default: MAX)")
    parser.add_argument("--force",      action="store_true",
                        help="Re-run Plaggie even when a cached score file exists")
    args = parser.parse_args()

    if args.build:
        build_jar()
        print("\nBuild complete. Run: python plaggie_runner.py")
        return

    if not JAR_PATH.exists():
        sys.exit(
            f"ERROR: plaggie.jar not found at {JAR_PATH}\n"
            "Build it first:  python plaggie_runner.py --build"
        )
    if not args.dataset.exists():
        sys.exit(f"ERROR: Dataset not found at {args.dataset}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_name = (
        f"Plaggie-Threshold-{args.threshold:.2f}"
        f"-MinTokens-{args.min_tokens}"
        f"-Metric-{args.metric}"
    )
    predictions_csv = OUT_DIR / f"{run_name}_results.csv"

    print(f"\nRun: {run_name}")
    print(f"  min_tokens={args.min_tokens}  threshold={args.threshold:.2f}  metric={args.metric}")
    print(f"  Output: {predictions_csv.name}")

    cases = _get_cases(args)
    all_rows: list[dict] = []

    for case_dir in cases:
        case_name = case_dir.name
        print(f"\n{'='*50}\n{case_name}\n{'='*50}")

        cached = None if args.force else load_score_cache(case_name, args.min_tokens)

        if cached is not None:
            print(f"  Using cached scores ({len(cached)} entries, min_tokens={args.min_tokens})")
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
            with tempfile.TemporaryDirectory() as tmp:
                work_dir = Path(tmp)
                submissions_dir = work_dir / "submissions"
                submissions_dir.mkdir()

                (work_dir / "plaggie.properties").write_text(
                    PROPERTIES_TEMPLATE.format(min_tokens=args.min_tokens)
                )

                meta = prepare_submissions(case_dir, submissions_dir)
                total_subs = len(meta) - 1
                print(f"  Submissions prepared: {total_subs} (+ original)")

                try:
                    ok, stdout = run_plaggie(submissions_dir, work_dir)
                except subprocess.TimeoutExpired:
                    print(f"  TIMEOUT after {PLAGGIE_TIMEOUT_S}s — skipping {case_name}",
                          file=sys.stderr)
                    continue

                if not ok:
                    print(f"  Skipping {case_name} due to Plaggie error.")
                    continue

                pairs = parse_output(stdout, submissions_dir)
                print(f"  Parsed {len(pairs)} pairs involving original")

                score_cache_rows: list[dict] = []
                for folder, (level, sub_id, is_plag) in sorted(meta.items()):
                    if folder == "original":
                        continue
                    if folder in pairs:
                        orig_in_sub, sub_in_orig = pairs[folder]
                    else:
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

                save_score_cache(case_name, args.min_tokens, score_cache_rows)
                print(f"  Score cache saved → {cache_path(case_name, args.min_tokens).name}")

    if not all_rows:
        print("\nNo results produced. Check Plaggie errors above.", file=sys.stderr)
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
        "min_tokens":      args.min_tokens,
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
