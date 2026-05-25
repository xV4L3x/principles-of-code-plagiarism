#!/usr/bin/env python3
"""
plaggie_runner.py — Evaluate Plaggie over IR-Plag-Dataset.

Strategy per case:
  1. Copy original + all plagiarized + non-plagiarized submissions into a
     temp directory (each in its own subfolder, uniquely named).
  2. Run Plaggie once per case with -s0.0 -nohtml; capture stdout.
  3. Parse stdout: extract (simA, simB) for each pair involving "original".
     simA = fraction of submission A tokens found in submission B;
     simB = vice versa.
  4. Compute the chosen similarity metric and emit one CSV row per submission.

Build note:
  Plaggie is distributed as source only (SourceForge). The --build flag
  downloads, patches (removes a hardcoded args line), compiles, and jars it.
  Run once before the first use:  python plaggie_runner.py --build

Layout:
  experiments/
    plaggie/
      plaggie_runner.py         ← this file
      plaggie.properties        ← configuration template (auto-generated per run)
      plaggie.jar               ← built by --build; listed in .gitignore
      plaggie-src/              ← patched source tree; listed in .gitignore
      out/
        plaggie_results.csv     ← standard CSV for evaluate.py

Usage:
  python plaggie_runner.py --build
  python plaggie_runner.py
  python plaggie_runner.py --min-tokens 5 --metric MAX --threshold 0.6
  python plaggie_runner.py --cases case-01 case-03
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

DATASET_ROOT = Path(__file__).parent.parent / "IR-Plag-Dataset"
OUT_DIR      = Path(__file__).parent / "out"
OUTPUT_CSV   = OUT_DIR / "plaggie_results.csv"
JAR_PATH     = Path(__file__).parent / "plaggie.jar"
SRC_DIR      = Path(__file__).parent / "plaggie-src"

DEFAULT_MIN_TOKENS = 3
DEFAULT_METRIC     = "MAX"
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
    """Download Plaggie 1.1 from SourceForge, patch, compile, and jar."""
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

    # Patch Plaggie.java: remove the hardcoded args line added by a SourceForge contributor
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

    # Compile
    print("Compiling Plaggie...", flush=True)
    bin_dir = SRC_DIR / "bin"
    bin_dir.mkdir(exist_ok=True)
    java_files = [str(f) for f in (SRC_DIR / "src").rglob("*.java")]
    sources_file = SRC_DIR / "sources.txt"
    sources_file.write_text("\n".join(java_files))
    result = subprocess.run(
        [
            "javac",
            "-encoding", "ISO-8859-1",
            "-sourcepath", str(SRC_DIR / "src"),
            "-d", str(bin_dir),
            f"@{sources_file}",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        sys.exit("ERROR: compilation failed")
    print(f"  Compiled {len(java_files)} source files (warnings suppressed)")

    # Package JAR
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


def prepare_submissions(case_dir: Path, submissions_dir: Path) -> dict[str, tuple[str, str, bool]]:
    """
    Populate submissions_dir with one subfolder per submission.
    Returns: {folder_name: (level, submission_id, is_plagiarized)}
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
    """
    Run Plaggie with -s0.0 -nohtml from work_dir (which contains plaggie.properties).
    Returns (success, stdout_text).
    """
    cmd = [
        "java",
        "-cp", str(JAR_PATH),
        "plag.parser.plaggie.Plaggie",
        "-s0.0",
        "-nohtml",
        str(submissions_dir),
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=PLAGGIE_TIMEOUT_S,
        cwd=str(work_dir),
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
    """Extract the submission folder name (first path component after submissions_dir)."""
    try:
        rel = Path(file_path).relative_to(submissions_dir)
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def parse_output(stdout: str, submissions_dir: Path) -> dict[str, tuple[float, float]]:
    """
    Parse Plaggie stdout report.

    Plaggie prints one block per pair:
        ========...
        Similarity A:<float>    ← fraction of sub_A tokens found in sub_B
        Similarity B:<float>    ← fraction of sub_B tokens found in sub_A
        --------...
        Files in submission A:
        <abs/path/to/file.java>
        --------...
        Files in submission B:
        <abs/path/to/file.java>

    Returns {sub_key: (orig_in_sub, sub_in_orig)} where orig_in_sub is the
    fraction of original tokens that appear in the submission, and sub_in_orig
    is the reverse.  Only pairs involving "original" are returned.
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
                section = "A"
                i += 1
                continue
            if line.startswith("Files in submission B:"):
                section = "B"
                i += 1
                continue
            if _SEP_RE.match(line):
                i += 1
                continue
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
            # simA = orig_in_sub, simB = sub_in_orig
            pairs[key_b] = (sim_a, sim_b)
        elif key_b == "original":
            # simA = sub_in_orig, simB = orig_in_sub  → swap
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Plaggie over IR-Plag-Dataset and write results CSV."
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Download, compile, and create plaggie.jar, then exit",
    )
    parser.add_argument("--dataset",   type=Path,  default=DATASET_ROOT)
    parser.add_argument("--output",    type=Path,  default=OUTPUT_CSV)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Similarity threshold for predicted_plag (default: 0.5)")
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS,
                        help="Minimum matching token sequence length (default: 3)")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        choices=["MAX", "AVG", "ORIG_IN_SUB", "SUB_IN_ORIG", "PRODUCT"],
                        help="Similarity metric (default: MAX)")
    parser.add_argument("--cases", nargs="+", default=None, metavar="CASE",
                        help="Run only these cases, e.g. --cases case-01 case-03")
    args = parser.parse_args()

    if args.build:
        build_jar()
        print("\nBuild complete. Run: python plaggie_runner.py")
        return

    if not JAR_PATH.exists():
        sys.exit(
            f"ERROR: plaggie.jar not found at {JAR_PATH}\n"
            "Build it first:\n"
            "  python plaggie_runner.py --build"
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
            work_dir = Path(tmp)
            submissions_dir = work_dir / "submissions"
            submissions_dir.mkdir()

            # Write per-run properties (allows --min-tokens to work)
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

            matched = 0
            for folder, (level, sub_id, is_plag) in sorted(meta.items()):
                if folder == "original":
                    continue
                if folder in pairs:
                    orig_in_sub, sub_in_orig = pairs[folder]
                    sim = apply_metric(orig_in_sub, sub_in_orig, args.metric)
                    matched += 1
                else:
                    sim = 0.0

                predicted = sim >= args.threshold
                rows.append({
                    "case":           case_name,
                    "level":          level,
                    "submission_id":  sub_id,
                    "similarity":     round(sim, 4),
                    "is_plagiarized": is_plag,
                    "predicted_plag": predicted,
                })
                flag = "PLAG" if is_plag else "    "
                print(
                    f"  [{flag}] {folder:<25} sim={sim:.4f}  pred={'Y' if predicted else 'N'}"
                )

            print(f"  Similarity found for {matched}/{total_subs} submissions vs original")

    if not rows:
        print("\nNo results produced. Check Plaggie errors above.", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "case", "level", "submission_id", "similarity",
        "is_plagiarized", "predicted_plag",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows written to {args.output}")
    print(f"Raw report: stdout only (no files saved)")


if __name__ == "__main__":
    main()
