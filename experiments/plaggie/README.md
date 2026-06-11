# Plaggie — Runner

[Plaggie](https://www.cs.hut.fi/Software/Plaggie/) (v1.1) is a token-based plagiarism detector that applies **Greedy String Tiling** (GST) to find maximal matching token sequences between every pair of submissions. It was developed at Helsinki University of Technology and is the direct predecessor of JPlag's algorithm.

## How the analysis works

### 1. Tokenisation

Plaggie parses each `.java` file into a normalised token sequence using its own Java lexer. Unlike SIM, identifier names are preserved in the token stream — renaming identifiers does affect the token sequence.

### 2. Greedy String Tiling (GST)

Plaggie applies GST to find the largest non-overlapping matching substrings between every pair of token sequences. The `--min-tokens` parameter sets the minimum tile length to count as a match — tiles shorter than this threshold are ignored. GST is the same core algorithm used in JPlag.

### 3. Batch comparison

For each dataset case, Plaggie receives all submissions in a single invocation (original + all plagiarised + all non-plagiarised). It compares all pairs and reports similarity for every combination.

### 4. Directional similarity

Plaggie reports two values per pair (A, B):
- **simA** — fraction of A's tokens covered by tiles found in B
- **simB** — fraction of B's tokens covered by tiles found in A

The runner filters to pairs involving `original` and derives a single similarity score per submission.

### 5. Similarity metrics

| Metric | Formula | Behaviour |
|--------|---------|-----------|
| `MAX` | `max(orig_in_sub, sub_in_orig)` | Aggressive; captures short plagiarised files nearly fully covered |
| `AVG` | `(orig_in_sub + sub_in_orig) / 2` | Balanced; penalises one-sided matches |
| `ORIG_IN_SUB` | fraction of original covered by submission | High when submission reproduces the original |
| `SUB_IN_ORIG` | fraction of submission covered by original | High when submission is a subset of the original |
| `PRODUCT` | `orig_in_sub × sub_in_orig` | Symmetric; penalises one-sided matches very harshly |

### 6. Score caching

Per-submission directional scores are cached per `(case, min_tokens)`:

```
out/case-01-mintokens-3_scores.csv
```

Runs that share the same `min_tokens` but differ only in `metric` or `threshold` reuse the cached scores — Plaggie is not re-invoked. Pass `--force` to overwrite an existing cache.

---

## Setup

**Requirements:** Java 8+ (`java`, `javac`, `jar` on PATH), Python 3.10+, numpy, scikit-learn

```bash
# Download, patch, compile, and jar Plaggie from SourceForge
cd experiments/plaggie
python plaggie_runner.py --build
```

Plaggie v1.1 from SourceForge contains a hardcoded path override in `main()` added by a contributor. The `--build` step removes this line automatically before compiling.

---

## Usage

Each invocation of `plaggie_runner.py` is a **run**: a fixed combination of parameters. The runner writes a per-run predictions CSV and appends one summary row to `out/plaggie_runs.csv`.

```bash
# Default run (threshold=0.5, min_tokens=3, metric=MAX)
python plaggie_runner.py

# Change metric only — reuses cached scores for min_tokens=3
python plaggie_runner.py --metric ORIG_IN_SUB --threshold 0.6

# Change min_tokens — triggers new Plaggie execution and new score cache
python plaggie_runner.py --min-tokens 7 --threshold 0.5

# Re-run Plaggie even if a cached score file exists
python plaggie_runner.py --min-tokens 3 --force

# Restrict to specific cases
python plaggie_runner.py --metric AVG --threshold 0.35 --cases case-01 case-02
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | `0.5` | Similarity cutoff for `predicted_plag`. |
| `--min-tokens` | `3` | Minimum GST tile length. Changing this requires re-running Plaggie. |
| `--metric` | `MAX` | Which score to use: `MAX`, `AVG`, `ORIG_IN_SUB`, `SUB_IN_ORIG`, `PRODUCT`. Changing this reuses the cached scores. |
| `--force` | off | Re-run Plaggie even when cached scores exist for the current `min_tokens`. |
| `--cases` | all | Run only named cases, e.g. `--cases case-01 case-03`. |
| `--dataset` | auto | Path to IR-Plag-Dataset directory. |
| `--build` | off | Download, compile, and create `plaggie.jar`, then exit. |

---

## Output

```
experiments/plaggie/out/
  plaggie_runs.csv                                                      ← one row per run (params + metrics)
  Plaggie-Threshold-0.50-MinTokens-3-Metric-MAX_results.csv
  Plaggie-Threshold-0.60-MinTokens-7-Metric-ORIG_IN_SUB_results.csv
  ...
  case-01-mintokens-3_scores.csv    ← cached (orig_in_sub, sub_in_orig) per submission
  case-01-mintokens-7_scores.csv
  ...
```

### `plaggie_runs.csv` schema

| Column | Description |
|--------|-------------|
| `run_name` | Auto-generated identifier encoding all parameters, e.g. `Plaggie-Threshold-0.50-MinTokens-3-Metric-MAX` |
| `min_tokens` | Value of `--min-tokens` for this run |
| `threshold` | Value of `--threshold` for this run |
| `metric` | Similarity metric used |
| `tp`, `fp`, `tn`, `fn` | Confusion matrix counts |
| `precision`, `recall`, `f1`, `accuracy` | Standard classification metrics |
| `auc` | ROC-AUC (threshold-independent discriminative power) |
| `mcc` | Matthews Correlation Coefficient — balanced metric, robust to class imbalance |
| `predictions_csv` | Filename of the corresponding predictions CSV in `out/` |

If a run with the same `run_name` is re-executed, its row is **overwritten** in place.

### Predictions CSV schema

Follows the [standard format](../README.md#standard-csv-format) shared by all tool runners.

---

## Hyperparameter search — `suggest_next.py`

After accumulating several runs, `suggest_next.py` fits a **Gaussian Process** surrogate on the observed results and recommends the next configuration to try via **Expected Improvement** (EI).

```bash
# Suggest next 5 runs (optimise F1, default diversity filter)
../results-analyzer/.venv/bin/python suggest_next.py

# Optimise MCC instead
../results-analyzer/.venv/bin/python suggest_next.py --metric mcc

# More exploration, wider diversity between suggestions
../results-analyzer/.venv/bin/python suggest_next.py --xi 0.05 --diversity 0.6

# Restrict the min_tokens search range
../results-analyzer/.venv/bin/python suggest_next.py --min-tokens-range 3 12
```

The script also prints:
- A **gradient estimate** at the current best (finite differences on the GP mean), showing which direction each parameter should move and the effect of switching to each alternative metric.
- A **degenerate run warning** if any run has TN=0 or FN=0 (threshold too low — all submissions predicted as plagiarised, metrics are inflated).
- A **landscape summary** showing best and average metric grouped by metric choice and by `min_tokens`.

### `suggest_next.py` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--metric` | `f1` | Metric to maximise: `f1`, `auc`, `accuracy`, or `mcc`. |
| `--top` | `5` | Number of suggestions to display. |
| `--xi` | `0.01` | EI exploration bonus. `0` = pure exploitation, `0.1` = strong exploration. |
| `--diversity` | `0.4` | Minimum normalised distance between suggestions. Prevents clustering around one region. |
| `--min-tokens-range` | `2 15` | `min_tokens` search range (inclusive). |
