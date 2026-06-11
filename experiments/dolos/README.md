# Dolos — Runner

[Dolos](https://dolos.ugent.be/) is a token-based plagiarism detector developed at Ghent University. It fingerprints source files using **k-gram hashing** (Rabin fingerprints), applies the **winnowing** algorithm to select a representative subset of fingerprints, and then computes pairwise similarity between all submissions.

## How the analysis works

### 1. Fingerprinting

Each source file is parsed into a flat token stream (using tree-sitter for language-specific tokenisation) and then sliced into all overlapping k-grams of length `--kgram`. Each k-gram is hashed with a Rabin fingerprint.

### 2. Winnowing

From the sequence of hashes, the **winnowing** algorithm selects a representative subset using a sliding window of size `--window` kgrams, keeping the minimum hash in each window. This produces a compact, position-invariant set of fingerprints per file.

### 3. Pairwise comparison

Dolos computes the overlap between fingerprint sets for every pair of files. For each pair it reports the number of matching fingerprints and the total fingerprint count per file — from these the runner derives directional coverage fractions.

### 4. Submission layout

For each dataset case, all Java files (original + all plagiarised + all non-plagiarised) are passed to Dolos in a single invocation. Dolos compares all pairs and writes a `pairs.csv` with per-pair similarity data.

The runner then filters the report to extract only `(original, submission)` pairs and applies the chosen metric to produce a single similarity score per submission.

### 5. Similarity metrics

Dolos reports directional coverage for each pair: how much of the left file is covered by matches and how much of the right. From these the runner computes:

| Metric | Formula | Behaviour |
|--------|---------|-----------|
| `COMBINED` | Dolos internal combined score | Default; symmetric-ish blend |
| `MAX` | `max(orig_frac, sub_frac)` | More aggressive; captures cases where a short plagiarised file is nearly fully covered |
| `AVG` | `(orig_frac + sub_frac) / 2` | Symmetric; penalises one-sided matches |
| `ORIG_IN_SUB` | `orig_covered / orig_total` | Fraction of the original covered by the submission |
| `SUB_IN_ORIG` | `sub_covered / sub_total` | Fraction of the submission covered by the original |

### 6. Report caching

Report directories are keyed by `(kgram, window)`:

```
out/case-01-kgram-23-window-17_report/
```

Multiple runs that differ only in `metric` or `threshold` reuse the same Dolos execution automatically. Pass `--force` to overwrite an existing cached report.

---

## Setup

**Requirements:** Node.js 22 (required for tree-sitter native module), Python 3.10+, numpy, scikit-learn

```bash
# Install Node 22 via nvm
nvm install 22
nvm use 22    # or: cd experiments/dolos && nvm use  (reads .nvmrc)

# Install Dolos
cd experiments/dolos
npm install @dodona/dolos
```

---

## Usage

Each invocation of `dolos_runner.py` is a **run**: a fixed combination of parameters. The runner logs the combination to the console, writes a per-run predictions CSV, and appends one summary row to `out/dolos_runs.csv`.

```bash
# Default run (threshold=0.5, kgram=23, window=17, metric=COMBINED)
python dolos_runner.py

# Try a different metric and threshold (reuses cached Dolos report)
python dolos_runner.py --metric ORIG_IN_SUB --threshold 0.25

# Change kgram (triggers a new Dolos execution)
python dolos_runner.py --kgram 10 --threshold 0.4

# Re-run Dolos even if a cached report exists
python dolos_runner.py --kgram 23 --force

# Restrict to specific cases
python dolos_runner.py --metric AVG --threshold 0.30 --cases case-01 case-02
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | `0.5` | Similarity cutoff for `predicted_plag`. |
| `--kgram` | `23` | k-gram length for fingerprinting. Changing this requires re-running Dolos. |
| `--window` | `17` | Winnowing window size in kgrams. Changing this requires re-running Dolos. |
| `--metric` | `COMBINED` | Which aggregate score to use: `COMBINED`, `MAX`, `AVG`, `SUB_IN_ORIG`, `ORIG_IN_SUB`. Changing this reuses the cached Dolos report. |
| `--force` | off | Re-run Dolos even when a cached report exists for the current kgram+window. |
| `--cases` | all | Run only named cases, e.g. `--cases case-01 case-03`. |
| `--dataset` | auto | Path to IR-Plag-Dataset directory. |

---

## Output

```
experiments/dolos/out/
  dolos_runs.csv                                                     ← one row per run (params + metrics)
  Dolos-Threshold-0.25-KGram-23-Window-17-Metric-ORIG_IN_SUB_results.csv
  Dolos-Threshold-0.50-KGram-23-Window-17-Metric-COMBINED_results.csv
  ...
  case-01-kgram-23-window-17_report/    ← raw Dolos CSV report, one dir per (case, kgram, window)
  case-02-kgram-23-window-17_report/
  ...
```

### `dolos_runs.csv` schema

| Column | Description |
|--------|-------------|
| `run_name` | Auto-generated identifier encoding all parameters, e.g. `Dolos-Threshold-0.25-KGram-23-Window-17-Metric-ORIG_IN_SUB` |
| `kgram` | Value of `--kgram` for this run |
| `window` | Value of `--window` for this run |
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

After accumulating several runs, `suggest_next.py` fits a **Gaussian Process** surrogate on the observed results and recommends the next configuration to try via **Expected Improvement** (EI). EI balances exploitation (predicted improvement over current best) and exploration (high uncertainty regions).

```bash
# Suggest next 5 runs (optimise F1, default diversity filter)
../results-analyzer/.venv/bin/python suggest_next.py

# Optimise AUC instead
../results-analyzer/.venv/bin/python suggest_next.py --metric auc

# More exploration, wider diversity between suggestions
../results-analyzer/.venv/bin/python suggest_next.py --xi 0.05 --diversity 0.6

# Restrict the kgram search range
../results-analyzer/.venv/bin/python suggest_next.py --kgram-range 10 30
```

The script also prints:
- A **gradient estimate** at the current best (finite differences on the GP mean), showing which direction each parameter should move and the effect of switching to each alternative metric.
- A **degenerate run warning** if any run has TN=0 or FN=0 (threshold too low — all submissions predicted as plagiarised, metrics are inflated).
- A **landscape summary** showing best and average metric grouped by metric choice and by kgram.

### `suggest_next.py` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--metric` | `f1` | Metric to maximise: `f1`, `auc`, or `accuracy`. |
| `--top` | `5` | Number of suggestions to display. |
| `--xi` | `0.01` | EI exploration bonus. `0` = pure exploitation, `0.1` = strong exploration. |
| `--diversity` | `0.4` | Minimum normalised distance between suggestions. Prevents clustering around one region. |
| `--kgram-range` | `5 30` | `kgram` search range (inclusive). |
| `--window` | `17` | Window value included in the suggested run command (for reproducibility). |
