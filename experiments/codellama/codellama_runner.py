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

── Run-based architecture ───────────────────────────────────────────────────────

Each invocation is a named run. Results are written to:
  out/<run_name>_results.csv     — per-submission predictions
  out/codellama_runs.csv         — one summary row per run (params + metrics)

Score caching: P(YES) scores are cached per (case, model, max_context, quantization)
in out/case-XX-model-<model_short>-ctx-<N>-quant-<q>_scores.csv.
Runs that share the same model/max_context/quantization but differ only in
threshold reuse the cache automatically. Pass --force to re-run inference.

── Backends ─────────────────────────────────────────────────────────────────────

  Two backends are supported:

  transformers (default)
    Uses HuggingFace transformers + PyTorch. Works on CUDA, MPS, and CPU.
    --quantization fp16  loads in float16 (~14 GB for 7B).  [default]
    --quantization fp32  loads in float32 (~28 GB for 7B).
    --quantization int4  uses bitsandbytes 4-bit (CUDA only, ~4 GB for 7B).

  mlx  (--mlx flag)
    Uses Apple's mlx-lm framework. Apple Silicon ONLY. Loads a pre-quantized
    4-bit model (~4 GB). Quantization is always int4; --quantization is ignored.
    Recommended for MacBook testing.

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

  Without --mlx, the runner uses the transformers backend.

── GPU / Colab ───────────────────────────────────────────────────────────────

  # A100 (40 GB): full float16
  python codellama_runner.py --device cuda

  # T4 (16 GB): 4-bit quantized via bitsandbytes
  python codellama_runner.py --device cuda --quantization int4

  Colab setup:
    !pip install torch transformers accelerate bitsandbytes
    !python codellama_runner.py --device cuda --quantization int4

── Examples ──────────────────────────────────────────────────────────────────

  python codellama_runner.py --mlx --model ./mlx_codellama_4bit --threshold 0.5
  python codellama_runner.py --threshold 0.6    # reuses cached scores if same model/ctx/quant
  python codellama_runner.py --device cuda --quantization int4 --threshold 0.5
  python codellama_runner.py --force            # re-run inference ignoring cache
"""

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_ROOT = Path(__file__).parent.parent / "IR-Plag-Dataset"
OUT_DIR      = Path(__file__).parent / "out"

DEFAULT_MODEL       = "codellama/CodeLlama-7b-Instruct-hf"
MLX_DEFAULT_MODEL   = "./mlx_codellama_4bit"
DEFAULT_THRESHOLD   = 0.5
DEFAULT_MAX_CONTEXT = 4096
DEFAULT_QUANT       = "fp16"

PROMPT_OVERHEAD = 120  # approximate token count for the prompt template itself

RUNS_CSV = OUT_DIR / "codellama_runs.csv"
RUNS_FIELDNAMES = [
    "run_name", "model", "max_context", "quantization", "threshold",
    "tp", "fp", "tn", "fn",
    "precision", "recall", "f1", "accuracy", "auc", "mcc",
    "predictions_csv",
]
PREDICTIONS_FIELDNAMES = [
    "case", "level", "submission_id", "similarity", "is_plagiarized", "predicted_plag",
]
SCORE_CACHE_FIELDNAMES = ["level", "submission_id", "is_plagiarized", "p_yes"]

PROMPT_TEMPLATE = """\
<s>[INST] You are a source code plagiarism detector. Analyze the two Java programs below.
Does the Submission appear to be plagiarized from the Original?

### Original:
{original_code}

### Submission:
{submission_code}

Answer with YES if the submission is plagiarized from the original, NO otherwise. [/INST]"""


# ─────────────────────────────────────────────────────────────────────────────
# Run name / cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def model_short(model_name: str) -> str:
    return Path(model_name).name


def score_cache_path(case_name: str, m_short: str, max_context: int, quant: str) -> Path:
    return OUT_DIR / f"{case_name}-model-{m_short}-ctx-{max_context}-quant-{quant}_scores.csv"


def load_score_cache(
    case_name: str, m_short: str, max_context: int, quant: str
) -> list[dict] | None:
    path = score_cache_path(case_name, m_short, max_context, quant)
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows if rows else None


def save_score_cache(
    case_name: str, m_short: str, max_context: int, quant: str, entries: list[dict]
) -> None:
    path = score_cache_path(case_name, m_short, max_context, quant)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_CACHE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(entries)
    print(f"  Score cache saved → {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

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


def append_run(run_row: dict) -> None:
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
    print(f"  {action} run '{run_row['run_name']}' in {RUNS_CSV.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_java_files(directory: Path) -> list[Path]:
    return list(directory.rglob("*.java"))


def collect_case_files(case_dir: Path) -> list[tuple[str, str, bool, list[Path]]]:
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
    quantization: str,
    cache_dir: str | None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    kwargs: dict = {"cache_dir": cache_dir} if cache_dir else {}

    print(f"  Loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

    print(f"  Loading model: {model_name}  [{quantization}]", flush=True)
    if quantization == "int4":
        if device != "cuda":
            sys.exit(
                "ERROR: --quantization int4 requires --device cuda. "
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
    elif quantization == "fp32":
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                **kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                **kwargs,
            )
            model.to(device)
    else:  # fp16 (default)
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                **kwargs,
            )
        else:
            # MPS or CPU: device_map="auto" is CUDA-only
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
    if cache_dir:
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)

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
    model, tokenizer = mlx_load(load_path)
    print("  MLX model loaded (int4 quantized)")
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_ids(tok, text: str, add_special_tokens: bool = False) -> list[int]:
    """Encode text to a plain list of ints.

    HuggingFace fast tokenizers return a tokenizers.Encoding object whose
    token ids live in .ids; slow tokenizers and AutoTokenizer return a plain
    list directly.
    """
    enc = tok.encode(text, add_special_tokens=add_special_tokens)
    return enc.ids if hasattr(enc, "ids") else list(enc)


def _decode_ids(tok, ids: list[int], skip_special_tokens: bool = True) -> str:
    return tok.decode(ids, skip_special_tokens=skip_special_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# YES/NO token ids
# ─────────────────────────────────────────────────────────────────────────────

def get_yes_no_ids(tokenizer) -> tuple[int, int]:
    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)

    def first_id(word: str) -> int:
        ids = _encode_ids(hf_tok, word)
        if not ids:
            ids = _encode_ids(hf_tok, " " + word)
        return ids[0]

    return first_id("YES"), first_id("NO")


# ─────────────────────────────────────────────────────────────────────────────
# Transformers backend — truncation + scoring
# ─────────────────────────────────────────────────────────────────────────────

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
    orig_ids = _encode_ids(hf_tok, original)
    sub_ids  = _encode_ids(hf_tok, submission)

    if len(orig_ids) + len(sub_ids) <= available:
        return original, submission

    print(
        f"  TRUNCATED {source_label}: orig={len(orig_ids)} sub={len(sub_ids)} "
        f"tokens → each capped at {half}",
        file=sys.stderr,
    )
    if len(orig_ids) > half:
        original = _decode_ids(hf_tok, orig_ids[:half])
    if len(sub_ids) > half:
        submission = _decode_ids(hf_tok, sub_ids[:half])
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
    input_ids = _encode_ids(hf_tok, prompt, add_special_tokens=True)

    if len(input_ids) > max_context:
        print(
            f"  TRUNCATED {source_label}: {len(input_ids)} tokens → {max_context}",
            file=sys.stderr,
        )
        input_ids = input_ids[:max_context]

    x = mx.array([input_ids])
    logits = model(x)

    last = logits[0, -1, :]
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
        description="Run CodeLlama-Instruct over IR-Plag-Dataset (run-based architecture).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT,
                        help="Path to IR-Plag-Dataset")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="P(YES) threshold for predicted_plag (default: 0.5)")
    parser.add_argument("--cases", nargs="+", default=None, metavar="CASE",
                        help="Run only these cases, e.g. --cases case-01 case-03")
    parser.add_argument("--model", default=None,
                        help=f"HuggingFace model id or local path. "
                             f"Default (transformers): {DEFAULT_MODEL}. "
                             f"Default (--mlx): {MLX_DEFAULT_MODEL}.")
    parser.add_argument("--mlx", action="store_true",
                        help="Use mlx-lm backend (Apple Silicon only). "
                             "Quantization is always int4; --quantization is ignored.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device for the transformers backend "
                             "(default: auto → cuda > mps > cpu). Ignored with --mlx.")
    parser.add_argument("--quantization", default=DEFAULT_QUANT,
                        choices=["fp16", "fp32", "int4"],
                        help="Model precision for the transformers backend "
                             "(default: fp16). int4 requires CUDA + bitsandbytes. "
                             "Ignored with --mlx (always int4).")
    parser.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT,
                        dest="max_context",
                        help=f"Token budget for the full prompt (default: {DEFAULT_MAX_CONTEXT})")
    parser.add_argument("--model-cache", type=str, default=None, dest="model_cache",
                        help="HuggingFace cache directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-run inference even if a score cache already exists")
    args = parser.parse_args()

    if not args.dataset.exists():
        sys.exit(f"ERROR: Dataset not found at {args.dataset}")

    if args.model_cache:
        os.environ["TRANSFORMERS_CACHE"] = args.model_cache

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = _get_cases(args)

    # Resolve model name and effective quantization label
    if args.mlx:
        model_name = args.model or MLX_DEFAULT_MODEL
        quant_label = "int4"  # MLX is always int4
    else:
        model_name = args.model or DEFAULT_MODEL
        quant_label = args.quantization

    m_short = model_short(model_name)

    run_name = (
        f"CodeLlama"
        f"-Model-{m_short}"
        f"-Ctx-{args.max_context}"
        f"-Quant-{quant_label}"
        f"-Threshold-{args.threshold:.2f}"
    )
    predictions_csv = OUT_DIR / f"{run_name}_results.csv"

    print("=" * 60)
    print(f"Run: {run_name}")
    print(f"  model         = {model_name}")
    print(f"  max_context   = {args.max_context}")
    print(f"  quantization  = {quant_label}")
    print(f"  threshold     = {args.threshold}")
    print(f"  backend       = {'mlx' if args.mlx else 'transformers'}")
    print(f"  output        = {predictions_csv.name}")
    print("=" * 60)

    # Determine which cases need inference
    cases_needing_inference = [
        c for c in cases
        if args.force or load_score_cache(c.name, m_short, args.max_context, quant_label) is None
    ]

    tokenizer = model = None
    if cases_needing_inference:
        print(f"\nInference needed for {len(cases_needing_inference)} case(s). Loading model…")
        if args.mlx:
            tokenizer, model = load_model_mlx(model_name, args.model_cache)
        else:
            device = _resolve_device(args.device)
            print(f"Using device: {device}")
            tokenizer, model = load_model(model_name, device, args.quantization, args.model_cache)

        yes_id, no_id = get_yes_no_ids(tokenizer)
        print(f"  YES token id: {yes_id}  NO token id: {no_id}")

        if args.mlx:
            score_fn = lambda orig_files, sub_files, sub_label: score_submission_mlx(
                orig_files, sub_files, tokenizer, model,
                yes_id, no_id, args.max_context, sub_label,
            )
        else:
            score_fn = lambda orig_files, sub_files, sub_label: score_submission(
                orig_files, sub_files, tokenizer, model, device,
                yes_id, no_id, args.max_context, sub_label,
            )
    else:
        print("\nAll score caches found — skipping inference.")
        score_fn = None

    all_rows: list[dict] = []

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

        # Load from cache or run inference
        cached = load_score_cache(case_name, m_short, args.max_context, quant_label)
        if cached is not None and not args.force:
            print(f"  Using score cache ({len(cached)} rows)")
            cache_map = {(r["level"], r["submission_id"]): float(r["p_yes"]) for r in cached}
        else:
            cache_entries: list[dict] = []
            cache_map: dict[tuple[str, str], float] = {}
            for level, sub_id, is_plag, sub_files in subs:
                key = f"{'plag' if is_plag else 'nonplag'}_{level}_{sub_id}" if is_plag else f"nonplag_{sub_id}"
                p_yes = score_fn(orig_files, sub_files, key)
                cache_entries.append({
                    "level":          level,
                    "submission_id":  sub_id,
                    "is_plagiarized": is_plag,
                    "p_yes":          round(p_yes, 4),
                })
                cache_map[(level, sub_id)] = p_yes
            save_score_cache(case_name, m_short, args.max_context, quant_label, cache_entries)

        for level, sub_id, is_plag, _ in subs:
            p_yes = cache_map.get((level, sub_id), 0.0)
            predicted = p_yes >= args.threshold
            all_rows.append({
                "case":           case_name,
                "level":          level,
                "submission_id":  sub_id,
                "similarity":     round(p_yes, 4),
                "is_plagiarized": is_plag,
                "predicted_plag": predicted,
            })
            flag = "PLAG" if is_plag else "    "
            print(f"  [{flag}] {level}/{sub_id:<4}  P(YES)={p_yes:.4f}  pred={'Y' if predicted else 'N'}")

    if not all_rows:
        print("\nNo results produced.", file=sys.stderr)
        sys.exit(1)

    # Write predictions CSV
    with open(predictions_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTIONS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nPredictions written → {predictions_csv.name}  ({len(all_rows)} rows)")

    # Compute metrics and append to runs CSV
    y_true  = np.array([r["is_plagiarized"] for r in all_rows], dtype=bool)
    y_score = np.array([r["similarity"]      for r in all_rows], dtype=float)
    m = compute_metrics(y_true, y_score, args.threshold)

    run_row = {
        "run_name":     run_name,
        "model":        model_name,
        "max_context":  args.max_context,
        "quantization": quant_label,
        "threshold":    args.threshold,
        **m,
        "predictions_csv": predictions_csv.name,
    }
    append_run(run_row)

    print(f"\nMetrics @ threshold={args.threshold:.2f}:")
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  Accuracy={m['accuracy']:.4f}  "
          f"AUC={m['auc']:.4f}  MCC={m['mcc']:.4f}")
    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
