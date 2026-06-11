#!/usr/bin/env python3
"""
codebert_runner.py — Evaluate CodeBERT (or GraphCodeBERT) over IR-Plag-Dataset.

Zero-shot approach: raw CodeBERT embeddings → cosine similarity. No anonymization,
no whitening. Each invocation is a named run; results are written to:
  out/<run_name>_results.csv         — per-submission predictions
  out/codebert_runs.csv              — one summary row per run (metrics + params)

Score caching: per-submission cosine similarities are cached per
(case, model_short, max_tokens, stride, pooling) in:
  out/case-XX-<model_short>-maxlen<N>-stride<S>-pooling-<p>_scores.csv
Runs that share the same model/pooling/max_tokens/stride but differ only in
threshold reuse the cache automatically. Pass --force to re-run inference.

Token limit: CodeBERT's position embeddings are hard-capped at 512. Files
exceeding --max-tokens are split into overlapping windows (stride = --stride);
each window is embedded independently and the final vector is the mean of all
window vectors.

Usage:
  python codebert_runner.py                           # default: codebert-base, mean, t=0.5
  python codebert_runner.py --pooling cls
  python codebert_runner.py --model microsoft/graphcodebert-base
  python codebert_runner.py --threshold 0.95          # reuses cached scores
  python codebert_runner.py --cases case-01 --device cpu
  python codebert_runner.py --force                   # re-run inference
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

DATASET_ROOT = Path(__file__).parent.parent / "IR-Plag-Dataset"
OUT_DIR      = Path(__file__).parent / "out"

RUNS_CSV = OUT_DIR / "codebert_runs.csv"
RUNS_FIELDNAMES = [
    "run_name", "model", "pooling", "threshold", "max_tokens", "stride",
    "tp", "fp", "tn", "fn",
    "precision", "recall", "f1", "accuracy", "auc", "mcc",
    "predictions_csv",
]
PREDICTIONS_FIELDNAMES = [
    "case", "level", "submission_id", "similarity", "is_plagiarized", "predicted_plag",
]
SCORE_CACHE_FIELDNAMES = ["level", "sub_id", "is_plag", "similarity"]

DEFAULT_MODEL     = "microsoft/codebert-base"
DEFAULT_POOLING   = "mean"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_TOK   = 512
DEFAULT_STRIDE    = 256


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def find_java_files(directory: Path) -> list[Path]:
    return list(directory.rglob("*.java"))


def collect_case_files(
    case_dir: Path,
) -> list[tuple[str, str, bool, list[Path]]]:
    """Returns [(level, sub_id, is_plagiarized, [java_files])], sorted."""
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
# Model name helpers
# ---------------------------------------------------------------------------

def model_short(model_name: str) -> str:
    """Derive a filesystem-safe short name from a HuggingFace model path."""
    return Path(model_name).name


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def load_model(
    model_name: str, device: str, cache_dir: str | None
) -> tuple[AutoTokenizer, AutoModel]:
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    print(f"  Loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    print(f"  Loading model: {model_name}", flush=True)
    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model on {device} ({n_params:.0f}M parameters)")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_text(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_tokens: int,
    stride: int,
    pooling: str,
) -> torch.Tensor:
    """
    Tokenize text and return a 1-D embedding tensor on CPU.

    pooling="mean": mean of all token hidden states in each window.
    pooling="cls":  [CLS] token hidden state only.

    Files exceeding max_tokens are split into overlapping windows; the final
    vector is the mean of per-window embeddings.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n_tokens = len(ids)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    inner = max_tokens - 2

    def pool_hidden(hidden: torch.Tensor, p: str) -> torch.Tensor:
        if p == "cls":
            return hidden[0, 0, :]
        return hidden[0, :, :].mean(dim=0)

    if n_tokens <= inner:
        input_ids = torch.tensor([[cls_id] + ids + [sep_id]], device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids)
        return pool_hidden(out.last_hidden_state, pooling).cpu()

    window_vecs: list[torch.Tensor] = []
    start = 0
    while start < n_tokens:
        chunk = ids[start: start + inner]
        input_ids = torch.tensor([[cls_id] + chunk + [sep_id]], device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids)
        window_vecs.append(pool_hidden(out.last_hidden_state, pooling).cpu())
        if start + inner >= n_tokens:
            break
        start += stride

    print(f"  WINDOWED: {n_tokens} tokens → {len(window_vecs)} window(s)", file=sys.stderr)
    return torch.stack(window_vecs).mean(dim=0)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return max(0.0, float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()))


def similarity_for_submission(
    orig_emb: torch.Tensor,
    sub_files: list[Path],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_tokens: int,
    stride: int,
    pooling: str,
) -> float:
    """Embed each file in sub_files and return the MAX cosine similarity vs orig_emb."""
    best = 0.0
    for f in sub_files:
        text = f.read_text(errors="replace")
        emb = embed_text(text, tokenizer, model, device, max_tokens, stride, pooling)
        best = max(best, cosine_sim(orig_emb, emb))
    return best


# ---------------------------------------------------------------------------
# Score cache  (keyed by case × model × max_tokens × stride × pooling)
# ---------------------------------------------------------------------------

def cache_path(case_name: str, m_short: str, max_tokens: int, stride: int, pooling: str) -> Path:
    return OUT_DIR / f"{case_name}-{m_short}-maxlen{max_tokens}-stride{stride}-pooling-{pooling}_scores.csv"


def load_score_cache(
    case_name: str, m_short: str, max_tokens: int, stride: int, pooling: str
) -> list[dict] | None:
    p = cache_path(case_name, m_short, max_tokens, stride, pooling)
    if not p.exists():
        return None
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def save_score_cache(
    case_name: str, m_short: str, max_tokens: int, stride: int, pooling: str,
    rows: list[dict],
) -> None:
    p = cache_path(case_name, m_short, max_tokens, stride, pooling)
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CodeBERT/GraphCodeBERT over IR-Plag-Dataset (zero-shot, run-based).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",    type=Path, default=DATASET_ROOT)
    parser.add_argument("--cases",      nargs="+", default=None, metavar="CASE")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--pooling",    default=DEFAULT_POOLING, choices=["mean", "cls"],
                        help="Pooling strategy: mean (all tokens) or cls ([CLS] token only)")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity cutoff for predicted_plag (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--max-tokens", type=int,   default=DEFAULT_MAX_TOK, dest="max_tokens")
    parser.add_argument("--stride",     type=int,   default=DEFAULT_STRIDE)
    parser.add_argument("--device",     default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--model-cache",type=str,   default=None, dest="model_cache")
    parser.add_argument("--force",      action="store_true",
                        help="Re-run inference even when a cached score file exists")
    args = parser.parse_args()

    if not args.dataset.exists():
        sys.exit(f"ERROR: Dataset not found at {args.dataset}")

    if args.model_cache:
        os.environ["TRANSFORMERS_CACHE"] = args.model_cache

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m_short = model_short(args.model)
    run_name = (
        f"CodeBERT-Threshold-{args.threshold:.2f}"
        f"-Model-{m_short}"
        f"-Pooling-{args.pooling}"
    )
    predictions_csv = OUT_DIR / f"{run_name}_results.csv"

    print(f"\nRun: {run_name}")
    print(f"  model={args.model}  pooling={args.pooling}  "
          f"threshold={args.threshold:.2f}  max_tokens={args.max_tokens}  stride={args.stride}")
    print(f"  Output: {predictions_csv.name}")

    cases = _get_cases(args)

    # Load model only if at least one case needs inference
    needs_inference = args.force or any(
        load_score_cache(case_dir.name, m_short, args.max_tokens, args.stride, args.pooling) is None
        for case_dir in cases
    )

    tokenizer = model = None
    if needs_inference:
        device = _resolve_device(args.device)
        print(f"\nUsing device: {device}")
        print("Loading model…")
        tokenizer, model = load_model(args.model, device, args.model_cache)
    else:
        device = "cpu"
        print("\nAll cases cached — skipping model load.")

    all_rows: list[dict] = []

    for case_dir in cases:
        case_name = case_dir.name
        print(f"\n{'='*50}\n{case_name}\n{'='*50}")

        cached = None if args.force else load_score_cache(
            case_name, m_short, args.max_tokens, args.stride, args.pooling
        )

        if cached is not None:
            print(f"  Using cached scores ({len(cached)} entries)")
            for entry in cached:
                sim = float(entry["similarity"])
                is_plag = entry["is_plag"] == "True"
                predicted = sim >= args.threshold
                all_rows.append({
                    "case": case_name,
                    "level": entry["level"],
                    "submission_id": entry["sub_id"],
                    "similarity": sim,
                    "is_plagiarized": is_plag,
                    "predicted_plag": predicted,
                })
                flag = "PLAG" if is_plag else "    "
                key = f"{entry['level']}_{entry['sub_id']}"
                print(f"  [{flag}] {key:<25} sim={sim:.4f}  pred={'Y' if predicted else 'N'}")
        else:
            orig_files = find_java_files(case_dir / "original")
            if not orig_files:
                print(f"  WARNING: no .java in original/ — skipping", file=sys.stderr)
                continue

            orig_text = "\n".join(f.read_text(errors="replace") for f in orig_files)
            print(f"  Embedding original ({len(orig_files)} file(s))…", flush=True)
            orig_emb = embed_text(
                orig_text, tokenizer, model, device,
                args.max_tokens, args.stride, args.pooling
            )

            submissions = collect_case_files(case_dir)
            print(f"  Submissions: {len(submissions)}", flush=True)

            score_cache_rows: list[dict] = []
            for level, sub_id, is_plag, sub_files in submissions:
                sim = similarity_for_submission(
                    orig_emb, sub_files, tokenizer, model, device,
                    args.max_tokens, args.stride, args.pooling
                )
                sim = round(sim, 4)
                score_cache_rows.append({
                    "level": level, "sub_id": sub_id,
                    "is_plag": is_plag, "similarity": sim,
                })
                predicted = sim >= args.threshold
                all_rows.append({
                    "case": case_name,
                    "level": level,
                    "submission_id": sub_id,
                    "similarity": sim,
                    "is_plagiarized": is_plag,
                    "predicted_plag": predicted,
                })
                flag = "PLAG" if is_plag else "    "
                key = f"{level}_{sub_id}"
                print(f"  [{flag}] {key:<25} sim={sim:.4f}  pred={'Y' if predicted else 'N'}")

            save_score_cache(
                case_name, m_short, args.max_tokens, args.stride, args.pooling,
                score_cache_rows
            )
            p = cache_path(case_name, m_short, args.max_tokens, args.stride, args.pooling)
            print(f"  Score cache saved → {p.name}")

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
        "model":           args.model,
        "pooling":         args.pooling,
        "threshold":       args.threshold,
        "max_tokens":      args.max_tokens,
        "stride":          args.stride,
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
