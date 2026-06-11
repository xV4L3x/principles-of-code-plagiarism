# JPlag — Runner

[JPlag](https://github.com/jplag/JPlag) is a token-based plagiarism detector developed at KIT. It tokenises source files using a language-specific frontend and then applies the **Greedy String Tiling (GST)** algorithm to find the longest common token sequences between every pair of submissions.

## How the analysis works

### 1. Tokenisation

JPlag parses each `.java` file into a flat sequence of language tokens (keywords, operators, identifiers, etc.). The `--min-tokens` parameter sets the minimum length of a matching token run that counts as a match — lower values increase sensitivity, which is necessary because the IR-Plag files are short (tens of lines).

### 2. Pairwise comparison

JPlag compares every submission against every other submission in the input directory. For each pair it computes two directional similarity scores:

- **first_similarity**: fraction of submission A's tokens covered by matches
- **second_similarity**: fraction of submission B's tokens covered by matches

From these it derives two aggregate metrics:

| Metric | Formula | Behaviour |
|--------|---------|-----------|
| `MAX` | `max(first, second)` | Captures cases where a small plagiarised file is 100% covered even if the original is larger. More aggressive. |
| `AVG` | `(first + second) / 2` | Symmetric; penalises one-sided matches. Tends to reduce false positives on structurally similar non-plagiarised submissions. |

Both can be selected via `--similarity-metric`.

### 3. Submission layout

For each dataset case the runner creates a temporary directory with one subfolder per submission:

```
submissions/
  original/           ← the reference file
  plag_L1_01/         ← plagiarised submission (level L1, id 01)
  plag_L1_02/
  ...
  plag_L6_09/
  nonplag_01/         ← independently written submission
  ...
  nonplag_15/
```

JPlag is invoked once per case and compares all pairs. The runner then filters the report to extract only `(original, submission)` pairs — that similarity score is what goes into the CSV.

### 4. Report format

JPlag writes a ZIP file (`case-XX_report.zip`) containing:

| File | Contents |
|------|----------|
| `overview.json` | Summary: all pairs, similarity scores, clustering |
| `<subA>-<subB>.json` | Detailed match data for each pair (token ranges, line numbers) |
| `files/<sub>/` | Source files as submitted |
| `options.json` | Exact CLI options used |

The raw ZIPs are preserved in `out/` so you can open them in the [JPlag web viewer](https://jplag.github.io/JPlag/) or inspect them manually.

---

## Setup

**Requirements:** Java 21, Python 3.10+, numpy, scikit-learn

Download the JAR (Java 21 compatible):

```bash
curl -L -o experiments/jplag/jplag.jar \
  https://github.com/jplag/JPlag/releases/download/v5.1.0/jplag-5.1.0-jar-with-dependencies.jar
```

---

## Usage

Each invocation of `jplag_runner.py` is a **run**: a fixed combination of parameters. The runner logs the combination to the console, writes a per-run predictions CSV, and appends one summary row to `out/jplag_runs.csv`.

```bash
# Default run (threshold=0.5, min-tokens=5, metric=MAX)
python jplag_runner.py

# Tune min-tokens
python jplag_runner.py --min-tokens 3

# Change similarity metric
python jplag_runner.py --similarity-metric AVG --threshold 0.20

# Restrict to specific cases
python jplag_runner.py --min-tokens 5 --threshold 0.30 --cases case-01 case-02
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | `0.5` | Similarity cutoff for `predicted_plag`. |
| `--min-tokens` | `5` | Minimum token-run length for a match to count. |
| `--similarity-metric` | `MAX` | Which aggregate score to extract: `MAX` or `AVG`. |
| `--jar` | `jplag.jar` | Path to the JPlag fat JAR. |
| `--cases` | all | Run only named cases, e.g. `--cases case-01 case-03`. |

---

## Output

```
experiments/jplag/out/
  jplag_runs.csv                                             ← one row per run (params + metrics)
  JPlag-Threshold-0.30-MinTokens-5-Metric-AVG_results.csv   ← predictions for that run
  JPlag-Threshold-0.50-MinTokens-5-Metric-MAX_results.csv
  ...
  case-01_report.zip    ← raw JPlag report, one per case
  case-02_report.zip
  ...
```

### `jplag_runs.csv` schema

| Column | Description |
|--------|-------------|
| `run_name` | Auto-generated identifier, e.g. `JPlag-Threshold-0.30-MinTokens-5-Metric-AVG` |
| `min_tokens` | Value of `--min-tokens` for this run |
| `threshold` | Value of `--threshold` for this run |
| `similarity_metric` | `MAX` or `AVG` |
| `tp`, `fp`, `tn`, `fn` | Confusion matrix counts |
| `precision`, `recall`, `f1`, `accuracy` | Standard classification metrics |
| `auc` | ROC-AUC (threshold-independent) |
| `mcc` | Matthews Correlation Coefficient — balanced metric, unaffected by class imbalance |
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

# Restrict the min-tokens search range
../results-analyzer/.venv/bin/python suggest_next.py --mt-range 2 8
```

The script also prints:
- A **gradient estimate** at the current best (finite differences on the GP mean), showing which direction each parameter should move.
- A **degenerate run warning** if any run has TN=0 or FN=0 (threshold too low — tool predicts everything as plagiarised, metrics are inflated).
- A **landscape summary** showing best and average metric grouped by similarity metric and by min-tokens.

### `suggest_next.py` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--metric` | `f1` | Metric to maximise: `f1`, `auc`, or `accuracy`. |
| `--top` | `5` | Number of suggestions to display. |
| `--xi` | `0.01` | EI exploration bonus. `0` = pure exploitation, `0.1` = strong exploration. |
| `--diversity` | `0.4` | Minimum normalised distance between suggestions. Prevents the list from clustering around a single region. |
| `--mt-range` | `1 10` | `min_tokens` search range (inclusive). |
