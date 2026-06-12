# CodeLlama — Runner for IR-Plag-Dataset

CodeLlama (Touvron et al., 2023) is a family of code-specialized LLMs built on top of
Llama 2. This runner uses the **instruct** variant (`CodeLlama-7b-Instruct-hf`) for
**zero-shot prompt-based binary classification**: the model is asked directly whether a
submission is plagiarized, and its answer is quantified via next-token logits.

This mirrors the approach of Brach et al. (FLLM 2024) who applied GPT-4o to SCPD, here
extended to an open-source 7B model. Unlike CodeBERT (embedding-based), CodeLlama uses
the model's generative reasoning to judge plagiarism.

## How the analysis works

```
For each case:
  1. Read the original/*.java source.
  2. For each submission:
       build the instruct prompt:
         "[INST] You are a source code plagiarism detector. …
          Does the Submission appear to be plagiarized from the Original?
          Answer with YES or NO. [/INST]"
       run a single forward pass
       extract logit(YES) and logit(NO) from the last token position
       similarity = softmax([logit(YES), logit(NO)])[0] = P(YES) ∈ [0, 1]
```

- **Approach**: zero-shot, no fine-tuning
- **Model**: `codellama/CodeLlama-7b-Instruct-hf` (default) or any instruct model via `--model`
- **Similarity**: P(YES) derived from the YES/NO logit ratio at the final prompt position
- **Multi-file submissions**: MAX P(YES) across all individual files vs. the original
- **Token budget**: if `original + submission` exceeds `--max-context` (default: 4096), both
  sides are truncated symmetrically to `(max_context − overhead) / 2` tokens each

## Backends

Two backends are supported, selectable at runtime:

### `transformers` (default)
Uses HuggingFace Transformers + PyTorch. Works on CUDA, MPS, and CPU.

| Hardware | `--quantization` | Memory | Notes |
|----------|-----------------|--------|-------|
| A100 (40 GB) | `fp16` | ~14 GB | `--device cuda` |
| T4 (16 GB) | `int8` | ~7 GB | `--device cuda --quantization int8` (requires `bitsandbytes`) |
| T4 (16 GB) | `int4` | ~4 GB | `--device cuda --quantization int4` (requires `bitsandbytes`) |
| Apple M-series | `fp16` | ~14 GB | `--device mps` (needs ≥24 GB unified memory) |

`--quantization int8/int4` requires `bitsandbytes`, which is **CUDA-only** — not available on MPS or CPU.
`--quantization fp32` uses full precision (~28 GB for 7B).

### `mlx` (`--mlx` flag)
Uses Apple's `mlx-lm` framework. **Apple Silicon only.** Loads a locally-converted
4-bit quantized model (~4 GB RAM). Recommended for MacBook testing.
Quantization is always `int4`; `--quantization` is ignored.

## Run-based architecture

Each invocation is a named **run**. Results are written to:

```
codellama/out/
  codellama_runs.csv                                  ← one row per run (params + metrics)
  <run_name>_results.csv                              ← per-submission predictions for that run
  case-XX-model-<model>-ctx-<N>-quant-<q>_scores.csv ← P(YES) score cache (reused across runs)
```

**Score caching**: P(YES) scores depend on `model`, `max_context`, and `quantization`. Running
with a different `threshold` only (same model/ctx/quant) reuses the cache without re-running
inference. Pass `--force` to re-run inference regardless.

**Run name** format: `CodeLlama-Model-<model>-Ctx-<N>-Quant-<q>-Threshold-<t>`

## Directory layout

```
codellama/
  codellama_runner.py     ← zero-shot inference runner (transformers + mlx backends)
  suggest_next.py         ← Bayesian optimisation advisor (GP + Expected Improvement)
  requirements.txt        ← pip dependencies
  mlx_codellama_4bit/     ← converted MLX model (generated locally, not in git)
  out/
    codellama_runs.csv
    <run_name>_results.csv
    case-XX-model-..._scores.csv
```

## Setup

```bash
cd experiments/codellama
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Note**: `bitsandbytes` (for `--quantization int4`) and `mlx-lm` (for `--mlx`) are both
optional. Install only what you need for your hardware:

```bash
# CUDA (T4/A100 on Colab)
pip install torch transformers accelerate bitsandbytes scikit-learn

# Apple Silicon
pip install torch transformers accelerate mlx-lm scikit-learn
```

### MLX model conversion (Apple Silicon only)

The converted model lives at `experiments/codellama/mlx_codellama_4bit/` (excluded from git
via `.gitignore` — ~3.5 GB of safetensors weights). It was generated with:

```bash
huggingface-cli login   # requires accepting the CodeLlama license on HuggingFace

python -m mlx_lm.convert \
    --hf-path codellama/CodeLlama-7b-Instruct-hf \
    -q \
    --mlx-path ./mlx_codellama_4bit
```

This downloads ~13 GB from HuggingFace and saves ~4 GB locally as 4-bit MLX weights.
The `mlx_codellama_4bit/` directory is the default `--model` path when using `--mlx`.

## Usage

```bash
# Apple Silicon (4-bit MLX) — recommended for Mac
python codellama_runner.py --mlx --model ./mlx_codellama_4bit --threshold 0.5

# Threshold sweep — reuses score cache (no inference)
python codellama_runner.py --mlx --model ./mlx_codellama_4bit --threshold 0.6

# Force re-run inference even if cache exists
python codellama_runner.py --mlx --model ./mlx_codellama_4bit --force

# Colab A100 (float16, full quality)
python codellama_runner.py --device cuda --threshold 0.5

# Colab T4 (4-bit bitsandbytes, ~4 GB)
python codellama_runner.py --device cuda --quantization int4 --threshold 0.5

# Specific cases only
python codellama_runner.py --mlx --cases case-01 case-02

# Bayesian next-config suggestion (needs ≥3 runs in codellama_runs.csv)
python suggest_next.py
python suggest_next.py --metric mcc
```

### Colab setup

```python
!pip install torch transformers accelerate bitsandbytes scikit-learn
!python codellama_runner.py --device cuda --quantization int4 --threshold 0.5
```

## Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--model` | `codellama/CodeLlama-7b-Instruct-hf` (transformers) / `./mlx_codellama_4bit` (mlx) | Any instruct-tuned causal LM |
| `--mlx` | off | Use mlx-lm backend; Apple Silicon only; quantization always int4 |
| `--device` | `auto` | Resolves: cuda → mps → cpu; ignored with `--mlx` |
| `--quantization` | `fp16` | `fp16` / `fp32` / `int4` (int4 CUDA+bitsandbytes only); ignored with `--mlx` |
| `--max-context` | `4096` | Total token budget for the prompt |
| `--threshold` | `0.5` | P(YES) cutoff for `predicted_plag` |
| `--cases` | all | Run only specific cases, e.g. `--cases case-01 case-03` |
| `--force` | off | Re-run inference even if score cache already exists |
| `--model-cache` | HF default | Override HuggingFace cache directory |

## Output

| File | Description |
|------|-------------|
| `out/codellama_runs.csv` | One row per run: params + TP/FP/TN/FN + precision/recall/F1/AUC/MCC |
| `out/<run_name>_results.csv` | Standard 6-column predictions CSV for `analyze.py` |
| `out/case-XX-model-..._scores.csv` | P(YES) score cache; reused when only threshold changes |

CSV columns (predictions): `case, level, submission_id, similarity, is_plagiarized, predicted_plag`

The `similarity` column stores `P(YES) ∈ [0, 1]`.

## Known limitations

- **YES/NO token ambiguity**: the logit extraction takes the first token id for the
  words `YES` and `NO`. If the tokenizer encodes them as multi-token sequences (rare
  for LLaMA-family), only the first token id is used — a known approximation in
  logit-based classification.
- **Prompt truncation loses context**: when files exceed half the token budget, code
  is cut off mid-function. Long submissions (>2000 tokens) lose structural context.
- **Single-pass scoring**: unlike CodeBERT's sliding window, there is no windowing
  for LLMs — truncation is hard. This is an inherent limitation of the prompt-based
  approach.
- **Zero-shot calibration**: P(YES) may not be well-calibrated as a similarity score.
  The optimal threshold is likely not 0.5; use `suggest_next.py` to find good configurations.
- **No instruction fine-tuning for SCPD**: the model is an instruct model but has not
  been fine-tuned on plagiarism pairs. Performance is bounded by the model's general
  code understanding, not by explicit plagiarism training signal.
- **Float16 on MPS requires ≥24 GB**: the 7B model in float16 occupies ~14 GB; macOS
  kernel and other processes compete for the same unified memory pool.
- **Quantization affects logits**: `fp16` and `int4` produce slightly different P(YES)
  values for the same prompt. They are treated as separate runs with separate score caches.

## References

- Touvron H. et al. — *Llama 2: Open Foundation and Fine-Tuned Chat Models*, 2023.
  [arxiv 2307.09288](https://arxiv.org/abs/2307.09288)
- Rozière B. et al. — *Code Llama: Open Foundation Models for Code*, 2023.
  [arxiv 2308.12950](https://arxiv.org/abs/2308.12950)
- Brach S. et al. — *Large Language Models for Source Code Plagiarism Detection*,
  FLLM 2024. (GPT-4o baseline this runner replicates with an open-source model)
- codellama/CodeLlama-7b-Instruct-hf — https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf
- mlx-lm — https://github.com/ml-explore/mlx-lm
