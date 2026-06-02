#!/usr/bin/env python3
"""
finetune.py — Fine-tune CodeBERT as a bi-encoder for code clone/plagiarism detection.

Uses CosineEmbeddingLoss (contrastive loss) to:
  - Pull clone pairs together (cosine sim → 1)
  - Push non-clone pairs apart (cosine sim → ≤ margin)

The fine-tuned model is saved in HuggingFace format and can be loaded directly by
codebert_runner.py via --model finetune/model/.

── Prerequisites ────────────────────────────────────────────────────────────────

  Run prepare_dataset.py first to generate train.jsonl and valid.jsonl.
  GPU strongly recommended (see --device).

── Memory note ──────────────────────────────────────────────────────────────────

  Training uses pair-by-pair gradient accumulation: for each pair (code_a, code_b),
  the forward passes, loss, and backward are computed immediately so only 2 CLS
  embeddings live on the computation graph at a time. This keeps memory constant
  regardless of batch size (which here controls only how often the optimizer steps).

  LoRA (Low-Rank Adaptation) is used by default: only ~300K parameters are
  trainable instead of 125M. This reduces optimizer state memory (AdamW first
  and second moments) from ~1 GB to ~2 MB — the key fix for Apple MPS where
  macOS consumes most of the unified memory.

  After training, LoRA weights are merged back into the base model and saved in
  standard HuggingFace format so codebert_runner.py needs no changes.

  On MPS, torch.mps.empty_cache() is called after every backward to work around
  the MPS backend's lazy memory release.

── Usage ────────────────────────────────────────────────────────────────────────

  # Full fine-tuning (GPU auto-detected)
  python finetune.py

  # Quick smoke test (1 epoch, 50 pairs)
  python finetune.py --epochs 1 --max-samples 50 --device cpu

  # Explicit GPU
  python finetune.py --device cuda

  # Custom output
  python finetune.py --model-out model/
"""

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Suppress HuggingFace's warning about sequences longer than 512 — we handle
# long files explicitly via the sliding window in get_cls_embedding.
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

HERE = Path(__file__).parent

DEFAULT_MODEL    = "microsoft/codebert-base"
DEFAULT_EPOCHS   = 5
DEFAULT_BATCH    = 8     # gradient-accumulation steps before optimizer.step()
DEFAULT_LR       = 2e-4  # higher LR is standard for LoRA (only adapter weights update)
DEFAULT_MARGIN   = 0.0   # CosineEmbeddingLoss margin for negative pairs
DEFAULT_MAX_TOK  = 512
DEFAULT_STRIDE   = 256

# LoRA config — adapts query and value projection matrices in every attention layer
LORA_R      = 8    # rank of the low-rank matrices
LORA_ALPHA  = 16   # scaling factor (effective lr scale = lora_alpha / lora_r = 2)
LORA_DROP   = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def configure_mps() -> None:
    """
    Disable the MPS high-watermark limit before the first allocation.

    By default PyTorch caps MPS at 80% of unified RAM. During long training runs
    the MPS backend accumulates memory in internal caches (a known leak) and hits
    this limit. Setting the ratio to 0.0 removes the cap so PyTorch can use all
    available RAM. This must be set before any MPS tensor is created.
    """
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    print("  MPS watermark disabled (PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0)")


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window embedding (mirrors codebert_runner.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_cls_embedding(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_tokens: int,
    stride: int,
    no_grad: bool,
) -> torch.Tensor:
    """
    Returns a CLS embedding as a 1-D tensor on `device` (training) or CPU (eval).

    Files exceeding max_tokens are split into overlapping windows; the result is
    the mean of all window CLS vectors.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    inner = max_tokens - 2

    def _forward(chunk_ids: list[int]) -> torch.Tensor:
        input_ids = torch.tensor([[cls_id] + chunk_ids + [sep_id]], device=device)
        if no_grad:
            with torch.no_grad():
                out = model(input_ids=input_ids)
        else:
            out = model(input_ids=input_ids)
        return out.last_hidden_state[:, 0, :].squeeze(0)

    if len(ids) <= inner:
        vec = _forward(ids)
        return vec.cpu() if no_grad else vec

    vecs = []
    start = 0
    while start < len(ids):
        vecs.append(_forward(ids[start: start + inner]))
        if start + inner >= len(ids):
            break
        start += stride

    stacked = torch.stack(vecs).mean(dim=0)
    return stacked.cpu() if no_grad else stacked


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ClonePairDataset(Dataset):
    def __init__(self, jsonl_path: Path, max_samples: int | None = None):
        self.pairs: list[tuple[str, str, int]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                # CosineEmbeddingLoss expects target ∈ {+1, -1}
                target = 1 if p["label"] == 1 else -1
                self.pairs.append((p["code_a"], p["code_b"], target))
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    loader: DataLoader,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_tokens: int,
    stride: int,
    loss_fn: nn.CosineEmbeddingLoss,
) -> tuple[float, float]:
    """Returns (val_loss, val_f1) at decision threshold = 0.0 in cosine ∈ [-1, 1]."""
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0

    for code_a, code_b, target in tqdm(loader, desc="  Val", leave=False):
        code_a = code_a[0] if isinstance(code_a, (list, tuple)) else code_a
        code_b = code_b[0] if isinstance(code_b, (list, tuple)) else code_b
        target_val = target.item() if hasattr(target, "item") else int(target[0])
        emb_a = get_cls_embedding(code_a, tokenizer, model, device, max_tokens, stride, no_grad=True)
        emb_b = get_cls_embedding(code_b, tokenizer, model, device, max_tokens, stride, no_grad=True)
        t = torch.tensor([target_val], dtype=torch.float, device=device)
        loss = loss_fn(emb_a.to(device).unsqueeze(0), emb_b.to(device).unsqueeze(0), t)
        total_loss += loss.item()

        sim = torch.nn.functional.cosine_similarity(
            emb_a.to(device).unsqueeze(0), emb_b.to(device).unsqueeze(0)
        ).item()
        pred = 1 if sim >= 0.0 else -1
        if pred == 1 and target_val == 1:
            tp += 1
        elif pred == 1 and target_val == -1:
            fp += 1
        elif pred == -1 and target_val == 1:
            fn += 1
        else:
            tn += 1

    avg_loss = total_loss / len(loader)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return avg_loss, f1


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    print(f"Device: {device}")
    if device == "mps":
        configure_mps()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModel.from_pretrained(args.model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["query", "value"],  # adapt Q and V projections in every attention layer
        lora_dropout=LORA_DROP,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()      # shows ~300K / 125M trainable

    model.to(device)
    model.train()

    train_path = HERE / "train.jsonl"
    valid_path = HERE / "valid.jsonl"
    if not train_path.exists():
        sys.exit(f"ERROR: {train_path} not found. Run prepare_dataset.py first.")
    if not valid_path.exists():
        sys.exit(f"ERROR: {valid_path} not found. Run prepare_dataset.py first.")

    train_ds = ClonePairDataset(train_path, max_samples=args.max_samples)
    valid_ds  = ClonePairDataset(valid_path, max_samples=args.max_samples)
    # batch_size=1 here because we accumulate over args.batch_size pairs manually
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)
    print(f"Train: {len(train_ds):,} pairs  |  Val: {len(valid_ds):,} pairs")
    print(f"Accumulating gradients over {args.batch_size} pairs per optimizer step")

    loss_fn   = nn.CosineEmbeddingLoss(margin=args.margin)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    steps_per_epoch = len(train_ds) // args.batch_size
    total_steps  = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1   = 0.0
    model_out = args.model_out
    ckpt_dir  = HERE / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    start_epoch = 1

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            sys.exit(f"ERROR: checkpoint not found: {ckpt_path}")
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_f1     = ckpt.get("best_f1", 0.0)
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resuming from epoch {start_epoch}  (best_f1 so far: {best_f1:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        accum_loss = torch.tensor(0.0, device=device)
        step = 0

        for i, (code_a, code_b, target) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        ):
            # code_a / code_b arrive as 1-element tuples from DataLoader
            code_a = code_a[0] if isinstance(code_a, (list, tuple)) else code_a
            code_b = code_b[0] if isinstance(code_b, (list, tuple)) else code_b
            target_val = target.item() if hasattr(target, "item") else int(target[0])

            emb_a = get_cls_embedding(code_a, tokenizer, model, device,
                                       args.max_tokens, args.stride, no_grad=False)
            emb_b = get_cls_embedding(code_b, tokenizer, model, device,
                                       args.max_tokens, args.stride, no_grad=False)
            t = torch.tensor([target_val], dtype=torch.float, device=device)
            loss = loss_fn(emb_a.unsqueeze(0), emb_b.unsqueeze(0), t)

            # Divide by batch_size so the effective gradient matches a batch average
            (loss / args.batch_size).backward()
            epoch_loss += loss.item()

            # Aggressively free MPS memory after every backward.
            # synchronize() ensures the MPS command queue is drained before
            # empty_cache() releases the allocator pool; gc.collect() drops
            # any Python-side tensor references that MPS hasn't freed yet.
            if device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
                gc.collect()

            # Optimizer step every batch_size pairs (or at the end of the epoch)
            if (i + 1) % args.batch_size == 0 or (i + 1) == len(train_ds):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if device == "mps":
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                    gc.collect()
                step += 1

        avg_train = epoch_loss / len(train_ds)
        val_loss, val_f1 = evaluate(valid_loader, tokenizer, model, device,
                                     args.max_tokens, args.stride, loss_fn)
        print(f"Epoch {epoch}: train_loss={avg_train:.4f}  val_loss={val_loss:.4f}"
              f"  val_f1={val_f1:.4f}")

        # ── Checkpoint (always) ───────────────────────────────────────────────
        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_f1":              best_f1,
            "val_f1":               val_f1,
        }, ckpt_path)
        print(f"  Checkpoint saved → {ckpt_path}")

        # ── Best model (merged, HuggingFace format) ───────────────────────────
        if val_f1 > best_f1:
            best_f1 = val_f1
            model_out.mkdir(parents=True, exist_ok=True)
            # Merge LoRA adapters back into the base weights so the saved model
            # is a plain HuggingFace model — codebert_runner.py loads it unchanged.
            merged = model.merge_and_unload()
            merged.save_pretrained(model_out)
            tokenizer.save_pretrained(model_out)
            print(f"  Saved best model (val_f1={best_f1:.4f}) → {model_out}")

    print(f"\nDone. Best val F1: {best_f1:.4f}  Model saved to: {model_out}")
    print(f"\nRun inference with:")
    print(f"  python ../codebert_runner.py --model {model_out.resolve()} "
          f"--output ../out/codebert_finetuned_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune CodeBERT bi-encoder on POJ-104 clone pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Base model to fine-tune (default: {DEFAULT_MODEL})")
    parser.add_argument("--model-out", type=Path, default=HERE / "model", dest="model_out",
                        help="Output directory for fine-tuned weights (default: finetune/model/)")
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH,  dest="batch_size",
                        help="Gradient accumulation steps before optimizer.step() (default: 16)")
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--margin",     type=float, default=DEFAULT_MARGIN,
                        help="CosineEmbeddingLoss margin for non-clone pairs (default: 0.0)")
    parser.add_argument("--max-tokens", type=int,   default=DEFAULT_MAX_TOK, dest="max_tokens")
    parser.add_argument("--stride",     type=int,   default=DEFAULT_STRIDE)
    parser.add_argument("--device",     default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples",
                        help="Limit pairs per split — for smoke tests, e.g. --max-samples 50")
    parser.add_argument("--resume-from", type=str, default=None, dest="resume_from",
                        help="Path to a checkpoint .pt file to resume training from "
                             "(e.g. finetune/checkpoints/epoch_01.pt)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
