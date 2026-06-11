#!/usr/bin/env python3
"""
codellama_runner.py — Evaluate CodeLlama (instruct) over IR-Plag-Dataset.

Approach: zero-shot prompt-based binary classification via next-token logits.
  1. Load codellama/CodeLlama-7b-Instruct-hf (or any instruct model via --model).
  2. For each case, read the original Java source once.
  3. For each submission, build a prompt containing both code snippets and run a
     single forward pass.
  4. Extract the logits for the tokens "YES" and "NO" from the final position and
     compute P(YES) / (P(YES) + P(NO)) as the similarity score in [0, 1].
  5. Write the standard 6-column CSV for analyze.py.

This mirrors the approach of Brach et al. 2024 (GPT-4o for SCPD), applied to an
open-source LLM. Unlike CodeBERT (embedding-based), this uses the model's
generative reasoning to judge plagiarism, which is the defining trait of
LLM-based SCPD.

Token budget: If original + submission exceed --max-context tokens, each side is
truncated symmetrically to half the available budget (after prompt overhead).

── Backends ─────────────────────────────────────────────────────────────────────

  Two backends are supported:

  transformers (default)
    Uses HuggingFace transformers + PyTorch. Works on CUDA, MPS, and CPU.
    Model loads in float16 (~14 GB for 7B). --load-in-4bit reduces to ~4 GB
    but requires bitsandbytes, which is CUDA-only.

  mlx  (--mlx flag)
    Uses Apple's mlx-lm framework. Apple Silicon ONLY. Loads a pre-quantized
    4-bit model from HuggingFace (~4 GB download). Recommended for MacBook
    testing. Incompatible with --load-in-4bit (quantization is built in).

── MacBook (Apple Silicon) ───────────────────────────────────────────────────

  Recommended: use --mlx for 4-bit quantized inference (~4 GB RAM for model).
  No pre-converted MLX repo exists for CodeLlama-Instruct; convert it locally
  first (requires HuggingFace login + CodeLlama license accepted):

    pip install -r requirements.txt

    # Step 1: convert to 4-bit MLX (downloads ~13 GB, saves ~4 GB locally)
    python -m mlx_lm.convert \
        --hf-path codellama/CodeLlama-7b-Instruct-hf \
        -q \
        --mlx-path ./mlx_codellama_4bit

    # Step 2: run the runner
    python codellama_runner.py --mlx --model ./mlx_codellama_4bit

  Without --mlx, the runner uses the transformers backend with float16 (~14 GB).
  This requires at least 24 GB of unified memory.

    python codellama_runner.py          # auto → mps, float16

  NOTE: --load-in-4bit is not available on MPS (bitsandbytes is CUDA-only).
  Use --mlx instead.

── GPU / Colab ───────────────────────────────────────────────────────────────

  # A100 (40 GB): full float16
  python codellama_runner.py --device cuda

  # T4 (16 GB): 4-bit quantized via bitsandbytes
  python codellama_runner.py --device cuda --load-in-4bit

  Colab setup:
    !pip install torch transformers accelerate bitsandbytes
    !python codellama_runner.py --device cuda --load-in-4bit
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_ROOT = Path(__file__).parent.parent / "IR-Plag-Dataset"
OUT_DIR      = Path(__file__).parent / "out"
OUTPUT_CSV   = OUT_DIR / "codellama_results.csv"

DEFAULT_MODEL       = "codellama/CodeLlama-7b-Instruct-hf"
MLX_DEFAULT_MODEL   = "./mlx_codellama_4bit"  # local path after conversion (see docs)
DEFAULT_THRESHOLD   = 0.5
DEFAULT_MAX_CONTEXT = 4096

PROMPT_OVERHEAD = 120  # approximate token count for the prompt template itself

PROMPT_TEMPLATE = """\
<s>[INST] You are a source code plagiarism detector. Analyze the two Java programs below.
Does the Submission appear to be plagiarized from the Original?

### Original:
{original_code}

### Submission:
{submission_code}

Answer with YES if the submission is plagiarized from the original, NO otherwise. [/INST]"""


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
    parts = []
    for f in files:
        parts.append(f"// --- FILE: {f.name} ---")
        parts.append(read_source(f))
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Transformers backend — model loading
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
    model_name: str,
    device: str,
    load_in_4bit: bool,
    cache_dir: str | None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    kwargs: dict = {"cache_dir": cache_dir} if cache_dir else {}

    print(f"  Loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

    print(f"  Loading model: {model_name}", flush=True)
    if load_in_4bit:
        if device != "cuda":
            sys.exit(
                "ERROR: --load-in-4bit requires --device cuda. "
                "bitsandbytes does not support MPS or CPU. Use --mlx on Mac."
            )
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            **kwargs,
        )
    elif device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            **kwargs,
        )
    else:
        # MPS (Apple Silicon) or CPU: load in float16 to halve memory (~14 GB
        # for 7B vs ~28 GB in float32). device_map="auto" is CUDA-only.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            **kwargs,
        )
        model.to(device)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    actual_device = next(model.parameters()).device
    print(f"  Model on {actual_device} ({n_params:.0f}M parameters)")
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# MLX backend — model loading  (Apple Silicon only)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_mlx(model_name: str, cache_dir: str | None):
    """Load a quantized model via mlx-lm. Returns (tokenizer, model)."""
    try:
        from mlx_lm import load as mlx_load
    except ImportError:
        sys.exit(
            "ERROR: mlx-lm is not installed. Run: pip install mlx-lm\n"
            "(mlx-lm requires Apple Silicon.)"
        )
    kwargs: dict = {}
    if cache_dir:
        # mlx_lm.load accepts tokenizer_config for cache, but the simplest
        # approach is to set the env var that HuggingFace respects.
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)

    # mlx_lm.load rejects relative paths — resolve to absolute if it's a local dir.
    # If the path looks local but doesn't exist, the model hasn't been converted yet.
    resolved = Path(model_name).resolve()
    looks_local = model_name.startswith(".") or model_name.startswith("/")
    if looks_local and not resolved.exists():
        sys.exit(
            f"ERROR: local model path '{model_name}' does not exist.\n"
            "Convert the model first with:\n\n"
            "  python -m mlx_lm.convert \\\n"
            "      --hf-path codellama/CodeLlama-7b-Instruct-hf \\\n"
            "      -q \\\n"
            "      --mlx-path ./mlx_codellama_4bit\n"
        )
    load_path = str(resolved) if resolved.exists() else model_name
    print(f"  Loading model via mlx-lm: {load_path}", flush=True)
    model, tokenizer = mlx_load(load_path, **kwargs)
    print(f"  MLX model loaded (4-bit quantized)")
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# YES/NO token ids
# ─────────────────────────────────────────────────────────────────────────────

def get_yes_no_ids(tokenizer) -> tuple[int, int]:
    """Return the single-token ids for 'YES' and 'NO'.

    Works with both HuggingFace AutoTokenizer and mlx-lm TokenizerWrapper
    (which exposes the underlying HF tokenizer via ._tokenizer).
    """
    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)

    def first_id(word: str) -> int:
        ids = hf_tok.encode(word, add_special_tokens=False)
        if not ids:
            ids = hf_tok.encode(" " + word, add_special_tokens=False)
        return ids[0]

    return first_id("YES"), first_id("NO")


# ─────────────────────────────────────────────────────────────────────────────
# Transformers backend — truncation + scoring
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_to_budget(text: str, token_budget: int, tokenizer) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= token_budget:
        return text
    return tokenizer.decode(ids[:token_budget], skip_special_tokens=True)


def prepare_texts(
    original: str,
    submission: str,
    max_context: int,
    tokenizer,
    source_label: str,
) -> tuple[str, str]:
    """Symmetrically truncate if the combined prompt would exceed max_context."""
    available = max_context - PROMPT_OVERHEAD
    half = available // 2

    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    orig_ids = hf_tok.encode(original, add_special_tokens=False)
    sub_ids  = hf_tok.encode(submission, add_special_tokens=False)

    if len(orig_ids) + len(sub_ids) <= available:
        return original, submission

    print(
        f"  TRUNCATED {source_label}: orig={len(orig_ids)} sub={len(sub_ids)} "
        f"tokens → each capped at {half}",
        file=sys.stderr,
    )
    if len(orig_ids) > half:
        original = hf_tok.decode(orig_ids[:half], skip_special_tokens=True)
    if len(sub_ids) > half:
        submission = hf_tok.decode(sub_ids[:half], skip_special_tokens=True)
    return original, submission


def score_pair(
    original: str,
    submission: str,
    tokenizer,
    model,
    device: str,
    yes_id: int,
    no_id: int,
    max_context: int,
) -> float:
    """Transformers backend: return P(YES) in [0, 1]."""
    prompt = PROMPT_TEMPLATE.format(
        original_code=original,
        submission_code=submission,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_context,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    yes_logit = logits[yes_id].float()
    no_logit  = logits[no_id].float()
    probs = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)
    return float(probs[0].item())


def score_submission(
    orig_files: list[Path],
    sub_files: list[Path],
    tokenizer,
    model,
    device: str,
    yes_id: int,
    no_id: int,
    max_context: int,
    sub_label: str,
) -> float:
    """Transformers backend: MAX score across individual file pairs."""
    orig_text = concat_sources(orig_files)
    best = 0.0
    for sub_file in sub_files:
        sub_text = read_source(sub_file)
        orig_trunc, sub_trunc = prepare_texts(
            orig_text, sub_text, max_context, tokenizer,
            source_label=f"{sub_label}/{sub_file.name}",
        )
        sim = score_pair(orig_trunc, sub_trunc, tokenizer, model, device,
                         yes_id, no_id, max_context)
        best = max(best, sim)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# MLX backend — scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_pair_mlx(
    original: str,
    submission: str,
    tokenizer,
    model,
    yes_id: int,
    no_id: int,
    max_context: int,
    source_label: str,
) -> float:
    """MLX backend: return P(YES) in [0, 1]."""
    import mlx.core as mx

    prompt = PROMPT_TEMPLATE.format(
        original_code=original,
        submission_code=submission,
    )

    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    input_ids = hf_tok.encode(prompt)

    if len(input_ids) > max_context:
        print(
            f"  TRUNCATED {source_label}: {len(input_ids)} tokens → {max_context}",
            file=sys.stderr,
        )
        input_ids = input_ids[:max_context]

    x = mx.array([input_ids])  # [1, seq_len]
    logits = model(x)          # [1, seq_len, vocab_size]

    last = logits[0, -1, :]    # [vocab_size]
    yes_logit = float(last[yes_id].item())
    no_logit  = float(last[no_id].item())

    probs = mx.softmax(mx.array([yes_logit, no_logit]), axis=-1)
    mx.eval(probs)
    return float(probs[0].item())


def score_submission_mlx(
    orig_files: list[Path],
    sub_files: list[Path],
    tokenizer,
    model,
    yes_id: int,
    no_id: int,
    max_context: int,
    sub_label: str,
) -> float:
    """MLX backend: MAX score across individual file pairs."""
    orig_text = concat_sources(orig_files)
    best = 0.0
    for sub_file in sub_files:
        sub_text = read_source(sub_file)
        sim = score_pair_mlx(
            orig_text, sub_text, tokenizer, model,
            yes_id, no_id, max_context,
            source_label=f"{sub_label}/{sub_file.name}",
        )
        best = max(best, sim)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CodeLlama-Instruct over IR-Plag-Dataset and write results CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT,
                        help="Path to IR-Plag-Dataset")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV,
                        help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="P(YES) threshold for predicted_plag (default: 0.5)")
    parser.add_argument("--cases", nargs="+", default=None, metavar="CASE",
                        help="Run only these cases, e.g. --cases case-01 case-03")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model id. "
                             f"Default (transformers): {DEFAULT_MODEL}. "
                             f"Default (--mlx): {MLX_DEFAULT_MODEL}.")
    parser.add_argument("--mlx", action="store_true",
                        help="Use mlx-lm backend with a 4-bit quantized model. "
                             "Apple Silicon only. Recommended for MacBook testing. "
                             "Ignores --device and --load-in-4bit.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device for the transformers backend "
                             "(default: auto → cuda > mps > cpu). Ignored with --mlx.")
    parser.add_argument("--load-in-4bit", action="store_true", dest="load_in_4bit",
                        help="4-bit quantization via bitsandbytes (CUDA only). "
                             "Use --mlx instead on Mac.")
    parser.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT,
                        dest="max_context",
                        help=f"Token budget for the full prompt (default: {DEFAULT_MAX_CONTEXT})")
    parser.add_argument("--model-cache", type=str, default=None, dest="model_cache",
                        help="HuggingFace cache directory (sets TRANSFORMERS_CACHE)")
    args = parser.parse_args()

    if not args.dataset.exists():
        sys.exit(f"ERROR: Dataset not found at {args.dataset}")

    if args.model_cache:
        os.environ["TRANSFORMERS_CACHE"] = args.model_cache

    cases = _get_cases(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("\nLoading model…")

    if args.mlx:
        model_name = args.model or MLX_DEFAULT_MODEL
        tokenizer, model = load_model_mlx(model_name, args.model_cache)
        yes_id, no_id = get_yes_no_ids(tokenizer)
        print(f"  YES token id: {yes_id}  NO token id: {no_id}")
        score_fn = lambda orig_files, sub_files, sub_label: score_submission_mlx(
            orig_files, sub_files, tokenizer, model,
            yes_id, no_id, args.max_context, sub_label,
        )
    else:
        model_name = args.model or DEFAULT_MODEL
        device = _resolve_device(args.device)
        print(f"Using device: {device}")
        tokenizer, model = load_model(model_name, device, args.load_in_4bit, args.model_cache)
        yes_id, no_id = get_yes_no_ids(tokenizer)
        print(f"  YES token id: {yes_id}  NO token id: {no_id}")
        score_fn = lambda orig_files, sub_files, sub_label: score_submission(
            orig_files, sub_files, tokenizer, model, device,
            yes_id, no_id, args.max_context, sub_label,
        )

    rows: list[dict] = []

    for case_dir in cases:
        case_name = case_dir.name
        print(f"\n{'='*50}\n{case_name}\n{'='*50}", flush=True)

        orig_files = find_java_files(case_dir / "original")
        if not orig_files:
            print(f"  WARNING: no .java files in {case_dir / 'original'} — skipping",
                  file=sys.stderr)
            continue

        subs = collect_case_files(case_dir)
        print(f"  {len(subs)} submissions | original: {len(orig_files)} file(s)")

        for key, (level, sub_id, is_plag, sub_files) in sorted(subs.items()):
            sim = score_fn(orig_files, sub_files, key)
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
            print(f"  [{flag}] {key:<30} P(YES)={sim:.4f}  pred={'Y' if predicted else 'N'}")

    if not rows:
        print("\nNo results produced.", file=sys.stderr)
        sys.exit(1)

    _write_csv(args.output, rows)
    print(f"\nDone. {len(rows)} rows → {args.output}")


if __name__ == "__main__":
    main()
