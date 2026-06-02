#!/usr/bin/env python3
"""
prepare_dataset.py — Build training/validation pairs for CodeBERT fine-tuning.

Source: google/code_x_glue_cc_clone_detection_poj104 (POJ-104 wrapped as CodeXGlue).
Each raw example has a `code` field (C source) and a `label` field (problem class 1–104).

Output:
  finetune/train.jsonl   — balanced clone / non-clone pairs from the train split
  finetune/valid.jsonl   — balanced pairs from the validation split

Each output line:
  {"code_a": "...", "code_b": "...", "label": 1}   ← same problem → clone
  {"code_a": "...", "code_b": "...", "label": 0}   ← different problem → not clone

── Usage ────────────────────────────────────────────────────────────────────────

  python prepare_dataset.py                      # default: 500 pos pairs per class
  python prepare_dataset.py --pairs-per-class 200   # smaller, for quick tests
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def build_pairs(
    examples: list[dict],
    pairs_per_class: int,
    seed: int,
) -> list[dict]:
    """
    Build balanced positive/negative code pairs from a flat list of {code, label} dicts.

    Positive (label=1): two snippets from the SAME problem class.
    Negative (label=0): two snippets from DIFFERENT problem classes.
    Returns a shuffled list; len(positives) == len(negatives).
    """
    rng = random.Random(seed)

    by_class: dict[int, list[str]] = defaultdict(list)
    for ex in examples:
        by_class[ex["label"]].append(ex["code"])
    classes = sorted(by_class.keys())

    # ── Positive pairs ────────────────────────────────────────────────────────
    positives: list[dict] = []
    for cls in classes:
        pool = by_class[cls]
        rng.shuffle(pool)
        # Walk consecutive pairs in the shuffled pool to avoid O(N²) enumeration
        count = 0
        for i in range(0, len(pool) - 1, 2):
            if count >= pairs_per_class:
                break
            positives.append({"code_a": pool[i], "code_b": pool[i + 1], "label": 1})
            count += 1

    # ── Negative pairs (same count as positives) ──────────────────────────────
    negatives: list[dict] = []
    while len(negatives) < len(positives):
        cls_a, cls_b = rng.sample(classes, 2)
        negatives.append({
            "code_a": rng.choice(by_class[cls_a]),
            "code_b": rng.choice(by_class[cls_b]),
            "label": 0,
        })

    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs


def write_jsonl(pairs: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Written {len(pairs):,} pairs → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare POJ-104 clone pairs for CodeBERT fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pairs-per-class", type=int, default=500, dest="pairs_per_class",
                        help="Positive pairs to sample per problem class (default: 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=HERE, dest="output_dir")
    args = parser.parse_args()

    print("Loading google/code_x_glue_cc_clone_detection_poj104 …")
    from datasets import load_dataset
    ds = load_dataset("google/code_x_glue_cc_clone_detection_poj104")

    for split_name, out_name in [("train", "train.jsonl"), ("validation", "valid.jsonl")]:
        split = ds[split_name]
        examples = [{"code": ex["code"], "label": ex["label"]} for ex in split]
        n_classes = len({ex["label"] for ex in examples})
        print(f"\n{split_name}: {len(examples):,} examples, {n_classes} problem classes")
        print(f"  Building pairs (pairs_per_class={args.pairs_per_class}) …")
        pairs = build_pairs(examples, args.pairs_per_class, seed=args.seed)
        n_pos = sum(1 for p in pairs if p["label"] == 1)
        n_neg = len(pairs) - n_pos
        print(f"  Positive: {n_pos:,}  Negative: {n_neg:,}  Total: {len(pairs):,}")
        write_jsonl(pairs, args.output_dir / out_name)

    print("\nDone. Run finetune.py next.")


if __name__ == "__main__":
    main()
