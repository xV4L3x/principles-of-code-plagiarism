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

| Hardware | Memory | Notes |
|----------|--------|-------|
| A100 (40 GB) | float16 ~14 GB | `--device cuda` |
| T4 (16 GB) | 4-bit ~4 GB | `--device cuda --load-in-4bit` (requires `bitsandbytes`) |
| Apple M-series | float16 ~14 GB | `--device mps` (needs ≥24 GB unified memory) |

`--load-in-4bit` requires `bitsandbytes`, which is **CUDA-only** — not available on MPS or CPU.

### `mlx` (`--mlx` flag)
Uses Apple's `mlx-lm` framework. **Apple Silicon only.** Loads a locally-converted
4-bit quantized model (~4 GB RAM). Recommended for MacBook testing.
Incompatible with `--load-in-4bit` (quantization is built in).

## Directory layout

```
codellama/
  codellama_runner.py     ← zero-shot inference runner (transformers + mlx backends)
  requirements.txt        ← pip dependencies
  mlx_codellama_4bit/     ← converted MLX model (generated locally, not in git)
  out/
    codellama_results.csv ← standard CSV for analyze.py
```

## Setup

```bash
cd experiments/codellama
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Note**: `bitsandbytes` (for `--load-in-4bit`) and `mlx-lm` (for `--mlx`) are both
optional. Install only what you need for your hardware:

```bash
# CUDA (T4/A100 on Colab)
pip install torch transformers accelerate bitsandbytes

# Apple Silicon
pip install torch transformers accelerate mlx-lm
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
python codellama_runner.py --mlx --model ./mlx_codellama_4bit

# Apple Silicon (float16 transformers) — requires ≥24 GB unified memory
python codellama_runner.py --device mps

# Colab A100 (float16, full quality)
python codellama_runner.py --device cuda

# Colab T4 (4-bit bitsandbytes, ~4 GB)
python codellama_runner.py --device cuda --load-in-4bit

# Specific cases only
python codellama_runner.py --mlx --cases case-01 case-02

# Custom threshold (default: 0.5)
python codellama_runner.py --mlx --threshold 0.6

# Custom output path
python codellama_runner.py --mlx --output out/codellama_results.csv
```

### Colab setup

```python
!pip install torch transformers accelerate bitsandbytes
!python codellama_runner.py --device cuda --load-in-4bit
```

## Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--model` | `codellama/CodeLlama-7b-Instruct-hf` (transformers) / `./mlx_codellama_4bit` (mlx) | Any instruct-tuned causal LM |
| `--mlx` | off | Use mlx-lm backend; Apple Silicon only |
| `--device` | `auto` | Resolves: cuda → mps → cpu; ignored with `--mlx` |
| `--load-in-4bit` | off | 4-bit via bitsandbytes; CUDA only |
| `--max-context` | `4096` | Total token budget for the prompt |
| `--threshold` | `0.5` | P(YES) cutoff for `predicted_plag`; use `analyze.py` for optimal F1 |
| `--cases` | all | Run only specific cases, e.g. `--cases case-01 case-03` |
| `--model-cache` | HF default | Override HuggingFace cache directory |

## Output

| File | Description |
|------|-------------|
| `out/codellama_results.csv` | Standard 6-column CSV for `analyze.py` |

CSV columns: `case, level, submission_id, similarity, is_plagiarized, predicted_plag`

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
  The optimal threshold is likely not 0.5; use `analyze.py` to find the F1-optimal value.
- **No instruction fine-tuning for SCPD**: the model is an instruct model but has not
  been fine-tuned on plagiarism pairs. Performance is bounded by the model's general
  code understanding, not by explicit plagiarism training signal.
- **Float16 on MPS requires ≥24 GB**: the 7B model in float16 occupies ~14 GB; macOS
  kernel and other processes compete for the same unified memory pool.

## References

- Touvron H. et al. — *Llama 2: Open Foundation and Fine-Tuned Chat Models*, 2023.
  [arxiv 2307.09288](https://arxiv.org/abs/2307.09288)
- Rozière B. et al. — *Code Llama: Open Foundation Models for Code*, 2023.
  [arxiv 2308.12950](https://arxiv.org/abs/2308.12950)
- Brach S. et al. — *Large Language Models for Source Code Plagiarism Detection*,
  FLLM 2024. (GPT-4o baseline this runner replicates with an open-source model)
- codellama/CodeLlama-7b-Instruct-hf — https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf
- mlx-lm — https://github.com/ml-explore/mlx-lm
