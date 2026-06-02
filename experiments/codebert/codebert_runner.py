#!/usr/bin/env python3
"""
codebert_runner.py — Evaluate CodeBERT (or GraphCodeBERT) over IR-Plag-Dataset.

Approach: zero-shot CLS embedding similarity (no fine-tuning).
  1. Load microsoft/codebert-base (or any HuggingFace code model via --model).
  2. For each case, embed the original Java source once.
  3. Embed each submission's Java source.
  4. Compute cosine similarity between the two CLS vectors.
  5. Write the standard 6-column CSV for evaluate.py.

Token limit: CodeBERT's position embeddings are hard-capped at 512. Files exceeding
--max-tokens are handled via a sliding window: the code is split into overlapping
chunks of size max_tokens (stride controlled by --stride), each chunk is embedded
independently, and the final vector is the mean of all window CLS vectors.

── Prerequisites ────────────────────────────────────────────────────────────────

  pip install -r requirements.txt

  First run downloads the model (~500 MB) to ~/.cache/huggingface/.

── Usage ────────────────────────────────────────────────────────────────────────

  # Full run, auto device
  python codebert_runner.py

  # CPU only
  python codebert_runner.py --device cpu

  # GraphCodeBERT
  python codebert_runner.py --model microsoft/graphcodebert-base \\
      --output out/graphcodebert_results.csv

  # Specific cases
  python codebert_runner.py --cases case-01 case-02

  # Custom threshold (default 0.5; use analyze.py for optimal F1)
  python codebert_runner.py --threshold 0.5
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DATASET_ROOT = Path(__file__).parent.parent / "IR-Plag-Dataset"
OUT_DIR      = Path(__file__).parent / "out"
OUTPUT_CSV   = OUT_DIR / "codebert_results.csv"

DEFAULT_MODEL     = "microsoft/codebert-base"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_TOK   = 512
DEFAULT_STRIDE    = 256   # sliding-window overlap (tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers (identical pattern to all other runners)
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
    fieldnames = ["case", "level", "submission_id", "similarity",
                  "is_plagiarized", "predicted_plag"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Source reading
# ─────────────────────────────────────────────────────────────────────────────

def read_source(path: Path) -> str:
    return path.read_text(errors="replace")


def concat_sources(files: list[Path]) -> str:
    return "\n".join(read_source(f) for f in files)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def get_cls_embedding(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_tokens: int,
    stride: int,
    source_path: Path | None = None,
) -> torch.Tensor:
    """
    Tokenize text and return a CLS embedding as a 1-D CPU tensor.

    If the token count fits within max_tokens, a single forward pass is used.
    If it exceeds max_tokens, a sliding window of size max_tokens with the given
    stride is applied: each window produces a CLS vector, and the final embedding
    is the mean of all window vectors.

    The [CLS] and [SEP] special tokens are added per window so each chunk is a
    complete RoBERTa input sequence.
    """
    # Tokenize without special tokens to get raw token ids for windowing
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n_tokens = len(ids)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    inner = max_tokens - 2  # slots available after [CLS] and [SEP]

    if n_tokens <= inner:
        # Single forward pass — fits in one window
        input_ids = torch.tensor([[cls_id] + ids + [sep_id]], device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids)
        return out.last_hidden_state[:, 0, :].squeeze(0).cpu()

    # Sliding window — average CLS vectors across all windows
    label = str(source_path) if source_path else "<text>"
    window_vecs: list[torch.Tensor] = []
    start = 0
    while start < n_tokens:
        chunk = ids[start : start + inner]
        input_ids = torch.tensor([[cls_id] + chunk + [sep_id]], device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids)
        window_vecs.append(out.last_hidden_state[:, 0, :].squeeze(0).cpu())
        if start + inner >= n_tokens:
            break
        start += stride

    n_windows = len(window_vecs)
    print(f"  WINDOWED {label}: {n_tokens} tokens → {n_windows} window(s) of {max_tokens}",
          file=sys.stderr)
    return torch.stack(window_vecs).mean(dim=0)


def embed_files(
    files: list[Path],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_tokens: int,
    stride: int,
) -> torch.Tensor:
    """Concatenate all files and return a single CLS embedding."""
    text = concat_sources(files)
    return get_cls_embedding(text, tokenizer, model, device, max_tokens, stride,
                             source_path=files[0] if files else None)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return max(0.0, float(sim))


def similarity_for_submission(
    orig_embedding: torch.Tensor,
    sub_files: list[Path],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_tokens: int,
    stride: int,
) -> float:
    """
    Embed each file in sub_files individually and return the MAX cosine similarity
    against orig_embedding. Mirrors the multi-file MAX strategy from sim_runner.py.
    """
    best = 0.0
    for f in sub_files:
        text = read_source(f)
        emb = get_cls_embedding(text, tokenizer, model, device, max_tokens, stride,
                                source_path=f)
        best = max(best, cosine_similarity(orig_embedding, emb))
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CodeBERT/GraphCodeBERT over IR-Plag-Dataset and write results CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT,
                        help="Path to IR-Plag-Dataset")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV,
                        help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Similarity threshold for predicted_plag (default: 0.5)")
    parser.add_argument("--cases", nargs="+", default=None, metavar="CASE",
                        help="Run only these cases, e.g. --cases case-01 case-03")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HuggingFace model name (default: {DEFAULT_MODEL}). "
                             "Also supports microsoft/graphcodebert-base.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to run on (default: auto → cuda > mps > cpu)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOK,
                        dest="max_tokens",
                        help="Maximum token count per file; raises an error if exceeded "
                             f"(default: {DEFAULT_MAX_TOK})")
    parser.add_argument("--model-cache", type=str, default=None, dest="model_cache",
                        help="HuggingFace cache directory (sets TRANSFORMERS_CACHE)")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, dest="stride",
                        help=f"Sliding-window stride in tokens for files exceeding --max-tokens "
                             f"(default: {DEFAULT_STRIDE}). Each window produces a CLS vector; "
                             "the final embedding is the mean of all window vectors.")
    args = parser.parse_args()

    if not args.dataset.exists():
        sys.exit(f"ERROR: Dataset not found at {args.dataset}")

    if args.model_cache:
        os.environ["TRANSFORMERS_CACHE"] = args.model_cache

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    cases = _get_cases(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model…")
    tokenizer, model = load_model(args.model, device, args.model_cache)

    rows: list[dict] = []

    for case_dir in cases:
        case_name = case_dir.name
        print(f"\n{'='*50}\n{case_name}\n{'='*50}", flush=True)

        orig_files = find_java_files(case_dir / "original")
        if not orig_files:
            print(f"  WARNING: no .java files in {case_dir/'original'} — skipping",
                  file=sys.stderr)
            continue

        print(f"  Embedding original ({len(orig_files)} file(s))…", flush=True)
        orig_embedding = embed_files(orig_files, tokenizer, model, device,
                                     args.max_tokens, args.stride)

        subs = collect_case_files(case_dir)
        print(f"  {len(subs)} submissions")

        for key, (level, sub_id, is_plag, sub_files) in sorted(subs.items()):
            sim = similarity_for_submission(
                orig_embedding, sub_files, tokenizer, model, device,
                args.max_tokens, args.stride,
            )
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
            print(f"  [{flag}] {key:<30} sim={sim:.4f}  pred={'Y' if predicted else 'N'}")

    if not rows:
        print("\nNo results produced.", file=sys.stderr)
        sys.exit(1)

    _write_csv(args.output, rows)
    print(f"\nDone. {len(rows)} rows → {args.output}")


if __name__ == "__main__":
    main()
