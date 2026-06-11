# CodeBERT — Runner

[CodeBERT](https://arxiv.org/abs/2002.08155) (Feng et al., EMNLP 2020) is a bimodal RoBERTa model pre-trained on code–documentation pairs from GitHub across 6 programming languages including Java. [GraphCodeBERT](https://arxiv.org/abs/2009.08366) extends it with data-flow graph structure.

This runner evaluates both models in **zero-shot** mode: no fine-tuning, no whitening, no anonymization. Each Java source file is embedded as a single dense vector; plagiarism is detected as cosine similarity above a threshold.

---

## How the analysis works

### 1. Tokenisation

Each Java source file is passed to the CodeBERT tokenizer (BPE, max 512 tokens). Files are read as-is — no preprocessing, no identifier renaming.

### 2. Embedding

CodeBERT produces a sequence of 768-dimensional hidden states for every input token. The runner derives a single vector per file using the chosen pooling strategy:

| Pooling | Description |
|---------|-------------|
| `mean` | Mean of all token hidden states. Captures the overall content distribution. |
| `cls` | [CLS] token hidden state only. The standard BERT sentence representation. |

### 3. Sliding window for long files

RoBERTa's position embeddings are hard-capped at 512 tokens. Files exceeding `--max-tokens` are split into overlapping chunks of size `max_tokens` with step `stride`. Each chunk is embedded independently and the final vector is the mean of all window vectors.

### 4. Similarity

Cosine similarity between the original's embedding and each submission's embedding, clipped to [0, 1]. For multi-file submissions, the **max** over individual files is used.

### 5. Score caching

Per-submission similarity scores are cached per `(case, model, max_tokens, stride, pooling)`:

```
out/case-01-codebert-base-maxlen512-stride256-pooling-mean_scores.csv
```

Runs that share the same model/pooling/max_tokens/stride but differ only in threshold reuse the cache — no model reload. Pass `--force` to recompute.

---

## Setup

```bash
pip install -r requirements.txt
```

The first run downloads the model (~500 MB) to `~/.cache/huggingface/`.

---

## Usage

Each invocation is a **run**: a fixed combination of parameters. The runner writes a per-run predictions CSV and appends one summary row to `out/codebert_runs.csv`.

```bash
# Default run (codebert-base, mean pooling, threshold=0.5)
python codebert_runner.py

# Try CLS pooling — reuses score cache if already computed for same model
python codebert_runner.py --pooling cls

# GraphCodeBERT
python codebert_runner.py --model microsoft/graphcodebert-base

# Threshold sweep (fast — no model reload)
python codebert_runner.py --threshold 0.92
python codebert_runner.py --threshold 0.95

# Restrict to specific cases
python codebert_runner.py --cases case-01 case-02 --device cpu

# Re-run inference from scratch
python codebert_runner.py --force
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | `0.5` | Similarity cutoff for `predicted_plag`. Post-hoc — reuses cache. |
| `--model` | `microsoft/codebert-base` | HuggingFace model ID. Changing this requires re-inference. |
| `--pooling` | `mean` | `mean` or `cls`. Changing this requires re-inference. |
| `--max-tokens` | `512` | Hard cap from RoBERTa's position embeddings. |
| `--stride` | `256` | Sliding-window overlap in tokens. |
| `--device` | `auto` | `auto` selects cuda > mps > cpu. |
| `--model-cache` | system default | HuggingFace cache directory. |
| `--force` | off | Recompute even when cached scores exist. |
| `--cases` | all | Run only named cases, e.g. `--cases case-01 case-03`. |

---

## Output

```
experiments/codebert/out/
  codebert_runs.csv                                                          ← one row per run
  CodeBERT-Threshold-0.90-Model-codebert-base-Pooling-mean_results.csv
  CodeBERT-Threshold-0.95-Model-graphcodebert-base-Pooling-cls_results.csv
  ...
  case-01-codebert-base-maxlen512-stride256-pooling-mean_scores.csv         ← score cache
  case-01-graphcodebert-base-maxlen512-stride256-pooling-cls_scores.csv
  ...
```

### `codebert_runs.csv` schema

| Column | Description |
|--------|-------------|
| `run_name` | Auto-generated identifier encoding all parameters |
| `model` | Full HuggingFace model ID |
| `pooling` | `cls` or `mean` |
| `threshold` | Similarity cutoff used |
| `max_tokens`, `stride` | Tokenisation parameters |
| `tp`, `fp`, `tn`, `fn` | Confusion matrix counts |
| `precision`, `recall`, `f1`, `accuracy` | Standard classification metrics |
| `auc` | ROC-AUC (threshold-independent) |
| `mcc` | Matthews Correlation Coefficient — balanced metric |
| `predictions_csv` | Filename of the corresponding predictions CSV |

### Predictions CSV schema

Follows the [standard format](../README.md#standard-csv-format) shared by all tool runners.

---

## Note on raw cosine similarity

Without whitening, CodeBERT embeddings exhibit **anisotropy**: the raw cosine similarity between any two code embeddings is naturally high (often 0.85–0.99) regardless of actual similarity. This compresses the useful score range to a narrow band near 1.0, so the optimal threshold is typically 0.90–0.98 rather than 0.5. Use `suggest_next.py` to find the optimal threshold efficiently.

---

## Hyperparameter search — `suggest_next.py`

After accumulating several runs, `suggest_next.py` fits a **Gaussian Process** surrogate on the observed results and recommends the next configuration to try via **Expected Improvement** (EI).

```bash
# Suggest next 5 runs optimising F1
../results-analyzer/.venv/bin/python suggest_next.py

# Optimise MCC
../results-analyzer/.venv/bin/python suggest_next.py --metric mcc
```

### `suggest_next.py` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--metric` | `f1` | Metric to maximise: `f1`, `auc`, `accuracy`, or `mcc`. |
| `--top` | `5` | Number of suggestions to display. |
| `--xi` | `0.01` | EI exploration bonus. |
| `--diversity` | `0.4` | Minimum normalised distance between suggestions. |
