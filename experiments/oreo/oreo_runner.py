#!/usr/bin/env python3
"""
oreo_runner.py — Evaluate Oreo over IR-Plag-Dataset.

Oreo (Saini et al., ESEC/FSE 2018) is a 3-phase hybrid clone detector for Java:
  Phase 1  Metric extraction  — Java JAR extracts 24 code metrics per method (local)
  Phase 2  Candidate pairing  — SourcererCC finds token-similar pairs (Docker)
  Phase 3  ML classification  — Siamese Keras network scores candidates (Docker)

Phases 2 and 3 always run inside Docker to faithfully reproduce the original
environment (Java 11, Python 3.6, TensorFlow 1.5, Keras 2.1.3).

Detection is method-level. Similarity is reported as:
  matched_original_methods / total_original_methods
where a method is "matched" if Oreo predicted it as a clone of a method in the
submission being evaluated.

── Prerequisites ────────────────────────────────────────────────────────────────

  1. Docker running (Docker Desktop or OrbStack).

  2. Build the image once:
       docker build --platform=linux/amd64 -t oreo-runner experiments/oreo/

  3. oreo-artifact already cloned at experiments/oreo/oreo-artifact/
     (contains pre-built java-parser JAR and the trained Keras model).

── Output layout ────────────────────────────────────────────────────────────────

  experiments/oreo/
    out/
      oreo_runs.csv          ← one row per run (params + metrics)
      oreo_scores.csv        ← score cache (all submissions, reused for threshold sweeps)
      Oreo-Threshold-0.50_results.csv   ← per-run predictions CSV
      work/
        flat/                ← 2-level input tree for Oreo metric extractor
        blocks.file          ← Phase 1 output (24-metric vectors per method)
        predictions/         ← Phase 3 output (clone pair .txt files)

── Usage ────────────────────────────────────────────────────────────────────────

  # Full pipeline (Phase 1 local + Phase 2+3 in Docker)
  python oreo_runner.py

  # Phase 1 only (metric extraction, no Docker needed)
  python oreo_runner.py --phase1-only

  # Resume: skip Phase 1 if blocks.file already exists
  python oreo_runner.py --skip-phase1

  # Threshold sweep — reuses score cache, no Docker needed
  python oreo_runner.py --threshold 0.3
  python oreo_runner.py --threshold 0.7

  # Re-run from scratch even if score cache exists
  python oreo_runner.py --force

  # Specific cases only
  python oreo_runner.py --cases case-01 case-02

  # Custom oreo-artifact path
  python oreo_runner.py --oreo-dir /other/path/oreo-artifact
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

DATASET_ROOT = Path(__file__).parent.parent / "IR-Plag-Dataset"
OUT_DIR      = Path(__file__).parent / "out"
WORK_DIR     = OUT_DIR / "work"
FLAT_DIR     = WORK_DIR / "flat"
DEFAULT_OREO_DIR = Path(__file__).parent / "oreo-artifact"

DEFAULT_THRESHOLD = 0.5
PHASE1_TIMEOUT_S  = 300
PHASE2_TIMEOUT_S  = 1800   # SourcererCC can be slow
PHASE3_TIMEOUT_S  = 600

RUNS_CSV   = OUT_DIR / "oreo_runs.csv"
SCORES_CSV = OUT_DIR / "oreo_scores.csv"

RUNS_FIELDNAMES = [
    "run_name", "threshold",
    "tp", "fp", "tn", "fn",
    "precision", "recall", "f1", "accuracy", "auc", "mcc",
    "predictions_csv",
]
PREDICTIONS_FIELDNAMES = [
    "case", "level", "submission_id", "similarity", "is_plagiarized", "predicted_plag",
]
SCORES_FIELDNAMES = ["case", "level", "sub_id", "is_plag", "similarity"]


# ─────────────────────────────────────────────────────────────────────────────
# Paths inside oreo-artifact
# ─────────────────────────────────────────────────────────────────────────────

def _paths(oreo_dir: Path) -> dict:
    oreo = oreo_dir / "oreo"
    return {
        "oreo":            oreo,
        # Phase 1
        "metric_jar":      oreo / "java-parser" / "dist" / "metricCalculator.jar",
        "phase1_script":   oreo / "python_scripts" / "metricCalculationWorkManager.py",
        "phase1_outdir":   oreo / "python_scripts" / "1_metric_output",
        "phase1_output":   oreo / "python_scripts" / "1_metric_output" / "mlcc_input.file",
        # Phase 2
        "cd_dir":          oreo / "clone-detector",
        "cd_jar":          oreo / "clone-detector" / "dist" / "indexbased.SearchManager.jar",
        "cd_input":        oreo / "clone-detector" / "input" / "dataset" / "blocks.file",
        "cd_candidates":   oreo / "results" / "candidates",
        "cd_cleanup":      oreo / "clone-detector" / "cleanup.sh",
        # Phase 3
        "predictor":       oreo / "python_scripts" / "Predictor.py",
        "model":           oreo / "ml_model" / "oreo_model_fse.h5",
        "predictions":     oreo / "results" / "predictions",
    }


def check_phase1(oreo_dir: Path) -> bool:
    p = _paths(oreo_dir)
    if not p["metric_jar"].exists():
        print(f"  ERROR: metricCalculator.jar not found: {p['metric_jar']}", file=sys.stderr)
        return False
    if not p["phase1_script"].exists():
        print(f"  ERROR: metricCalculationWorkManager.py not found: {p['phase1_script']}",
              file=sys.stderr)
        return False
    return True


def check_docker() -> bool:
    """Verify Docker is available and the oreo-runner image exists."""
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        print("ERROR: Docker is not running. Start Docker Desktop or OrbStack.",
              file=sys.stderr)
        return False
    check = subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE],
                           capture_output=True)
    if check.returncode != 0:
        print(f"ERROR: Docker image '{DOCKER_IMAGE}' not found.", file=sys.stderr)
        print("  Build it with:", file=sys.stderr)
        print(f"    docker build --platform=linux/amd64 -t {DOCKER_IMAGE} experiments/oreo/",
              file=sys.stderr)
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_java_files(directory: Path) -> list[Path]:
    return list(directory.rglob("*.java"))


def collect_case_files(case_dir: Path) -> dict[str, tuple[str, str, bool, list[Path]]]:
    """Returns {folder_key: (level, sub_id, is_plagiarized, [java_files])}."""
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


def flat_key(case_name: str, folder_key: str) -> str:
    """Name of the flat subdirectory for a given case + folder_key."""
    return f"{case_name}_{folder_key}"


def prepare_flat_tree(cases: list[Path]) -> dict[str, dict[str, tuple[str, str, bool]]]:
    """
    Build the 2-level flat input tree that Oreo requires:
      flat/<case>_<folder_key>/*.java

    Returns case_meta: {case_name: {folder_key: (level, sub_id, is_plag)}}
    """
    if FLAT_DIR.exists():
        shutil.rmtree(FLAT_DIR)
    FLAT_DIR.mkdir(parents=True)

    case_meta: dict[str, dict[str, tuple[str, str, bool]]] = {}

    for case_dir in cases:
        case_name = case_dir.name
        case_meta[case_name] = {}

        # Original reference files
        orig_files = find_java_files(case_dir / "original")
        if not orig_files:
            print(f"  WARNING: no .java files in {case_dir/'original'}", file=sys.stderr)
        else:
            dest = FLAT_DIR / flat_key(case_name, "original")
            dest.mkdir()
            for f in orig_files:
                shutil.copy(f, dest / f.name)
            case_meta[case_name]["original"] = ("original", "original", False)

        # Submissions
        subs = collect_case_files(case_dir)
        for key, (level, sub_id, is_plag, files) in subs.items():
            dest = FLAT_DIR / flat_key(case_name, key)
            dest.mkdir()
            for f in files:
                shutil.copy(f, dest / f.name)
            case_meta[case_name][key] = (level, sub_id, is_plag)

        n = len(subs)
        print(f"  {case_name}: {len(orig_files)} original file(s), {n} submissions → {n+1} flat dirs")

    return case_meta


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Metric extraction
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1(oreo_dir: Path) -> bool:
    """
    Run metricCalculationWorkManager.py from python_scripts/ with FLAT_DIR as input.
    Output: python_scripts/1_metric_output/mlcc_input.file → WORK_DIR/blocks.file
    """
    p = _paths(oreo_dir)
    script_dir = p["phase1_script"].parent

    # Clean previous output so we don't merge with stale data
    if p["phase1_outdir"].exists():
        shutil.rmtree(p["phase1_outdir"])

    cmd = [
        sys.executable,
        str(p["phase1_script"]),
        "1",                        # 1 parallel process
        "d",                        # input type: directory
        str(FLAT_DIR.resolve()),    # absolute path to flat input tree
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)

    # Must run from python_scripts/ — script uses __file__-relative paths for output
    result = subprocess.run(
        cmd,
        cwd=str(script_dir),
        capture_output=True,
        text=True,
        timeout=PHASE1_TIMEOUT_S,
    )
    # Phase 1 runs async (Popen inside) — the script returns before the JAR finishes.
    # Wait by polling for the output file.
    import time
    for _ in range(60):  # up to 5 minutes
        if p["phase1_output"].exists():
            break
        time.sleep(5)

    if not p["phase1_output"].exists():
        # Show error info
        out = (result.stdout + result.stderr).strip()
        if out:
            print(f"  Phase 1 script output:\n{out[-2000:]}", file=sys.stderr)
        # Also check metric.err for JAR errors
        metric_err = script_dir / "metric.err"
        if metric_err.exists():
            err_text = metric_err.read_text().strip()
            if err_text:
                print(f"  metric.err:\n{err_text[-1000:]}", file=sys.stderr)
        print(f"  ERROR: blocks.file not produced at {p['phase1_output']}", file=sys.stderr)
        return False

    dest = WORK_DIR / "blocks.file"
    shutil.copy(p["phase1_output"], dest)
    n_lines = sum(1 for _ in open(dest))
    print(f"  Phase 1 OK: {n_lines} method entries → {dest}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# (Python Phase 2 / Phase 3 fallbacks removed — Docker is the only path)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Docker mode — Phase 2 + Phase 3 in the original environment
# ─────────────────────────────────────────────────────────────────────────────

DOCKER_IMAGE = "oreo-runner"
DOCKERFILE_DIR = Path(__file__).parent  # experiments/oreo/


def run_docker(oreo_dir: Path) -> bool:
    """
    Run Phase 2 (SourcererCC, Java 8) + Phase 3 (TF 1.5 / Keras 2.1.3)
    inside a Docker container.

    Volume layout inside the container:
      /oreo-artifact/    ← oreo-artifact (rw: SourcererCC writes state)
      /data/blocks.file  ← Phase 1 output (ro)
      /data/output/      ← predictions written here
    """
    blocks_file = WORK_DIR / "blocks.file"
    pred_dest   = WORK_DIR / "predictions"

    if not blocks_file.exists():
        print(f"  ERROR: {blocks_file} not found. Run Phase 1 first.", file=sys.stderr)
        return False

    pred_dest.mkdir(exist_ok=True)

    # Build image if not already present
    check = subprocess.run(
        ["docker", "image", "inspect", DOCKER_IMAGE],
        capture_output=True,
    )
    if check.returncode != 0:
        print(f"  Building Docker image '{DOCKER_IMAGE}'…", flush=True)
        build = subprocess.run(
            ["docker", "build", "--platform=linux/amd64",
             "-t", DOCKER_IMAGE, str(DOCKERFILE_DIR)],
            check=False,
        )
        if build.returncode != 0:
            print("  ERROR: docker build failed.", file=sys.stderr)
            return False
        print("  Image built.")
    else:
        print(f"  Using existing Docker image '{DOCKER_IMAGE}'.")

    cmd = [
        "docker", "run", "--rm",
        "--platform=linux/amd64",
        "-v", f"{oreo_dir.resolve()}:/oreo-artifact",
        "-v", f"{blocks_file.resolve()}:/data/blocks.file:ro",
        "-v", f"{pred_dest.resolve()}:/data/output",
        DOCKER_IMAGE,
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, timeout=PHASE2_TIMEOUT_S + PHASE3_TIMEOUT_S)
    if result.returncode != 0:
        print(f"  Docker run failed (exit {result.returncode}).", file=sys.stderr)
        return False

    n = len(list(pred_dest.glob("*.txt")))
    if n == 0:
        print("  WARNING: no prediction .txt files in output.", file=sys.stderr)
    else:
        print(f"  Docker OK: {n} prediction file(s) → {pred_dest}")
    return True




# ─────────────────────────────────────────────────────────────────────────────
# Parsing blocks.file and predictions
# ─────────────────────────────────────────────────────────────────────────────

def parse_blocks_file(blocks_file: Path) -> dict[str, tuple[str, str]]:
    """
    Parse blocks.file produced by Phase 1.
    Returns {function_id: (flat_folder_name, java_filename)}.

    Line format (comma + @#@ separated):
      <flat_folder>,<filename>,<start>,<end>,<method_name>,<tokens>,<unique>,<hash>,<proj_id>,<func_id>
      @#@ <metric1>,...,<metric24>
      @#@ <calls>
      @#@ ...
    """
    id_to_folder: dict[str, tuple[str, str]] = {}

    with open(blocks_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on @#@ to get the header part
            header = line.split("@#@")[0]
            fields = header.split(",")
            if len(fields) < 10:
                continue
            flat_folder = fields[0].strip()
            filename    = fields[1].strip()
            func_id     = fields[9].strip()
            id_to_folder[func_id] = (flat_folder, filename)

    return id_to_folder


def parse_predictions(pred_dir: Path) -> list[tuple[str, str, str, str, str, str]]:
    """
    Parse all Oreo prediction .txt files.
    Returns list of (flat_folder1, file1, start1, flat_folder2, file2, start2).

    Oreo output format (one method pair per line):
      <flat_folder1>,<file1>,<start1>,<end1>,<flat_folder2>,<file2>,<start2>,<end2>
    """
    results = []
    for txt in sorted(pred_dir.glob("*.txt")):
        with open(txt, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = line.split(",")
                if len(fields) >= 8:
                    results.append((
                        fields[0].strip(),  # flat_folder1
                        fields[1].strip(),  # file1
                        fields[2].strip(),  # start_line1
                        fields[4].strip(),  # flat_folder2
                        fields[5].strip(),  # file2
                        fields[6].strip(),  # start_line2
                    ))
    return results


def flat_folder_to_case_and_key(flat_folder: str, cases: list[str]) -> tuple[str, str] | None:
    """
    Reverse-map a flat folder name back to (case_name, folder_key).
    flat_folder = '<case_name>_<folder_key>', e.g. 'case-01_plag_L2_03'
    """
    for case_name in cases:
        prefix = case_name + "_"
        if flat_folder.startswith(prefix):
            folder_key = flat_folder[len(prefix):]
            return (case_name, folder_key)
    return None


def aggregate_similarities(
    blocks_file: Path,
    pred_dir: Path,
    case_meta: dict[str, dict[str, tuple[str, str, bool]]],
) -> dict[str, dict[str, float]]:
    """
    For each case and submission, compute:
      similarity = matched_original_methods / total_original_methods

    where a method is "matched" if Oreo predicted it as a clone of at least
    one method in the submission being evaluated.

    Oreo prediction format:
      flat_folder1, file1, start1, end1, flat_folder2, file2, start2, end2
    """
    # --- Count total original methods per case from blocks.file ---
    print("  Counting original methods from blocks.file…", flush=True)
    # orig_methods[case_name] = set of (file, start_line) identifying each method
    orig_methods: dict[str, set[tuple[str, str]]] = {}
    with open(blocks_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            header = line.split("@#@")[0]
            fields = header.split(",")
            if len(fields) < 4:
                continue
            flat_folder = fields[0].strip()
            if not flat_folder.endswith("_original"):
                continue
            case_name = flat_folder[: -len("_original")]
            file_name  = fields[1].strip()
            start_line = fields[2].strip()
            orig_methods.setdefault(case_name, set()).add((file_name, start_line))
    for case_name, methods in orig_methods.items():
        print(f"    {case_name}: {len(methods)} original method(s)")

    # --- Parse predictions ---
    print("  Parsing predictions…", flush=True)
    pairs = parse_predictions(pred_dir)
    print(f"    {len(pairs)} predicted clone pairs total")

    # --- Build per-case, per-submission matched method sets ---
    # matched[(case_name, folder_key)] = set of (orig_file, orig_start) that matched
    matched: dict[tuple[str, str], set[tuple[str, str]]] = {}

    orig_suffix = "_original"
    for (f1, file1, start1, f2, file2, start2) in pairs:
        for orig_flat, orig_f, orig_s, sub_flat in [
            (f1, file1, start1, f2),
            (f2, file2, start2, f1),
        ]:
            if not orig_flat.endswith(orig_suffix):
                continue
            case_name = orig_flat[: -len(orig_suffix)]
            if not sub_flat.startswith(case_name + "_"):
                continue  # cross-case pair — skip
            folder_key = sub_flat[len(case_name) + 1:]
            if folder_key == "original":
                continue
            matched.setdefault((case_name, folder_key), set()).add((orig_f, orig_s))

    # --- Build result ---
    result: dict[str, dict[str, float]] = {}
    total_matched_subs = 0
    for case_name, subs in case_meta.items():
        result[case_name] = {}
        total_orig = len(orig_methods.get(case_name, set()))
        for folder_key in subs:
            if folder_key == "original":
                continue
            n_matched = len(matched.get((case_name, folder_key), set()))
            sim = n_matched / total_orig if total_orig > 0 else 0.0
            result[case_name][folder_key] = sim
            if sim > 0:
                total_matched_subs += 1

    print(f"    Submissions with sim > 0: {total_matched_subs}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def compute_metrics(y_true: list[int], y_score: list[float], threshold: float) -> dict:
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)
    f1        = _safe_div(2 * precision * recall, precision + recall)
    accuracy  = _safe_div(tp + tn, tp + fp + tn + fn)

    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = _safe_div(tp * tn - fp * fn, denom)

    try:
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "accuracy":  round(accuracy,  4),
        "auc":       round(auc,       4),
        "mcc":       round(mcc,       4),
    }


def append_run(run_row: dict) -> None:
    rows: list[dict] = []
    if RUNS_CSV.exists():
        with open(RUNS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("run_name") != run_row["run_name"]]
    rows.append(run_row)
    with open(RUNS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Oreo over IR-Plag-Dataset (run-based, score-cached).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--oreo-dir", type=Path, default=DEFAULT_OREO_DIR,
                        help=f"Path to oreo-artifact directory (default: {DEFAULT_OREO_DIR})")
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT,
                        help="Path to IR-Plag-Dataset")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Similarity threshold for predicted_plag (default: 0.5)")
    parser.add_argument("--cases", nargs="+", default=None, metavar="CASE",
                        help="Run only these cases, e.g. --cases case-01 case-03")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Reuse existing out/work/blocks.file (flat/ must also exist)")
    parser.add_argument("--phase1-only", action="store_true",
                        help="Run only Phase 1 (metric extraction). No Docker needed.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run pipeline even if score cache (oreo_scores.csv) exists.")
    args = parser.parse_args()

    if not args.oreo_dir.exists():
        sys.exit(
            f"ERROR: oreo-artifact not found at {args.oreo_dir}\n"
            "  Clone with: git clone https://github.com/Mondego/oreo-artifact oreo-artifact\n"
            "  (place it at experiments/oreo/oreo-artifact/)"
        )
    if not args.dataset.exists():
        sys.exit(f"ERROR: Dataset not found at {args.dataset}")

    cases = _get_cases(args)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_name       = f"Oreo-Threshold-{args.threshold:.2f}"
    predictions_csv = OUT_DIR / f"{run_name}_results.csv"

    # ── Phase 1 only (debug shortcut — always re-runs, no cache) ─────────────
    if args.phase1_only:
        if not check_phase1(args.oreo_dir):
            sys.exit("Phase 1 prerequisites not met.")
        print(f"\n{'='*50}\nPhase 1 — Metric extraction\n{'='*50}")
        print("Building flat input tree…")
        prepare_flat_tree(cases)
        try:
            ok = run_phase1(args.oreo_dir)
        except subprocess.TimeoutExpired:
            sys.exit(f"ERROR: Phase 1 timed out after {PHASE1_TIMEOUT_S}s")
        if not ok:
            sys.exit("Phase 1 failed.")
        print(f"\nPhase 1 complete. blocks.file at: {WORK_DIR / 'blocks.file'}")
        print("Stopping after Phase 1 (--phase1-only).")
        return

    # ── Score cache ───────────────────────────────────────────────────────────
    if SCORES_CSV.exists() and not args.force:
        print(f"Loading score cache: {SCORES_CSV}")
        with open(SCORES_CSV, newline="") as f:
            cached_rows = list(csv.DictReader(f))
        if args.cases:
            case_names = {c.name for c in cases}
            cached_rows = [r for r in cached_rows if r["case"] in case_names]
        if not cached_rows:
            sys.exit(
                f"ERROR: score cache exists but contains no data for {args.cases}.\n"
                "  Run with --force to regenerate."
            )
        print(f"  {len(cached_rows)} cached similarity scores loaded.")
    else:
        # ── Phase 1 — Metric extraction (local) ──────────────────────────────
        case_meta: dict[str, dict[str, tuple[str, str, bool]]] = {}

        if args.skip_phase1:
            blocks_file = WORK_DIR / "blocks.file"
            if not blocks_file.exists():
                sys.exit(f"ERROR: --skip-phase1 but {blocks_file} not found.")
            print("Skipping Phase 1 (reusing existing blocks.file + flat/).")
            for case_dir in cases:
                case_name = case_dir.name
                case_meta[case_name] = {"original": ("original", "original", False)}
                for key, (level, sub_id, is_plag, _) in collect_case_files(case_dir).items():
                    case_meta[case_name][key] = (level, sub_id, is_plag)
        else:
            if not check_phase1(args.oreo_dir):
                sys.exit("Phase 1 prerequisites not met.")
            print(f"\n{'='*50}\nPhase 1 — Metric extraction\n{'='*50}")
            print("Building flat input tree…")
            case_meta = prepare_flat_tree(cases)
            try:
                ok = run_phase1(args.oreo_dir)
            except subprocess.TimeoutExpired:
                sys.exit(f"ERROR: Phase 1 timed out after {PHASE1_TIMEOUT_S}s")
            if not ok:
                sys.exit("Phase 1 failed.")
            print(f"\nPhase 1 complete. blocks.file at: {WORK_DIR / 'blocks.file'}")

        # ── Phase 2 + Phase 3 — Docker ────────────────────────────────────────
        if not check_docker():
            sys.exit(1)
        print(f"\n{'='*50}\nDocker — Phase 2 (SourcererCC) + Phase 3 (Siamese)\n{'='*50}")
        try:
            ok = run_docker(args.oreo_dir)
        except subprocess.TimeoutExpired:
            sys.exit("ERROR: Docker run timed out.")
        if not ok:
            sys.exit("Docker run failed.")

        # ── Aggregate ─────────────────────────────────────────────────────────
        print(f"\n{'='*50}\nAggregating results\n{'='*50}")
        blocks_file = WORK_DIR / "blocks.file"
        pred_dir    = WORK_DIR / "predictions"

        if not blocks_file.exists():
            sys.exit(f"ERROR: {blocks_file} not found. Run Phase 1 first.")
        if not pred_dir.exists() or not any(pred_dir.glob("*.txt")):
            sys.exit(f"ERROR: no prediction files in {pred_dir}. Run Docker phase first.")

        sims = aggregate_similarities(blocks_file, pred_dir, case_meta)

        # ── Write score cache ─────────────────────────────────────────────────
        cached_rows = []
        for case_name in sorted(sims):
            for folder_key, sim in sorted(sims[case_name].items()):
                info = case_meta[case_name].get(folder_key)
                if info is None:
                    continue
                level, sub_id, is_plag = info
                cached_rows.append({
                    "case":       case_name,
                    "level":      level,
                    "sub_id":     sub_id,
                    "is_plag":    is_plag,
                    "similarity": round(sim, 4),
                })
        with open(SCORES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SCORES_FIELDNAMES)
            writer.writeheader()
            writer.writerows(cached_rows)
        print(f"\nScore cache written: {SCORES_CSV}  ({len(cached_rows)} rows)")

    # ── Apply threshold + write predictions CSV ───────────────────────────────
    pred_rows: list[dict] = []
    y_true:    list[int]   = []
    y_score:   list[float] = []

    for row in cached_rows:
        sim     = float(row["similarity"])
        is_plag = str(row["is_plag"]).strip().lower() in ("true", "1", "yes")
        predicted = sim >= args.threshold
        pred_rows.append({
            "case":           row["case"],
            "level":          row["level"],
            "submission_id":  row["sub_id"],
            "similarity":     sim,
            "is_plagiarized": is_plag,
            "predicted_plag": predicted,
        })
        y_true.append(int(is_plag))
        y_score.append(sim)

    if not pred_rows:
        print("\nNo results produced.", file=sys.stderr)
        sys.exit(1)

    with open(predictions_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTIONS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(pred_rows)

    # ── Metrics + run record ──────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_score, args.threshold)
    run_row = {
        "run_name":       run_name,
        "threshold":      args.threshold,
        **metrics,
        "predictions_csv": predictions_csv.name,
    }
    append_run(run_row)

    print(f"\n{run_name}")
    print(f"  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
          f"F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
          f"AUC={metrics['auc']:.4f}  MCC={metrics['mcc']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  "
          f"TN={metrics['tn']}  FN={metrics['fn']}")
    print(f"  predictions → {predictions_csv.name}")
    print(f"  run record  → {RUNS_CSV.name}")


if __name__ == "__main__":
    main()
