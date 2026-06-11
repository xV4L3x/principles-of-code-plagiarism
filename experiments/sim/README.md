# SIM — Runner

[SIM](https://github.com/sauloq/sim) (Software Similarity Tester) is a string-based plagiarism detector by Dick Grune. It tokenises source files using a language-specific lexer and finds the longest common runs of tokens between every pair of files. Unlike JPlag (which uses Greedy String Tiling), SIM uses a direct run-matching approach similar to longest common substring.

## How the analysis works

### 1. Tokenisation

SIM parses each `.java` file into a sequence of normalised tokens. Identifiers are all mapped to the same `IDF` token, string literals to `STR`, numbers to their numeric token, etc. This means identifier renaming (plagiarism Level 3) does **not** fool SIM — the token sequence remains identical after normalisation.

### 2. Run matching

SIM scans for the longest contiguous runs of matching tokens between the two files. The `-r N` flag sets the minimum run length to count as a match. With the default value of `24`, most files in the IR-Plag-Dataset (which are only tens of lines long) produce no matches at all — so we use small values of `-r` to stay sensitive on short files.

### 3. Directional similarity

SIM reports similarity **asymmetrically** — it computes two values for a pair (A, B):

- **A → B** (`ORIG_IN_SUB`): percentage of A's tokens covered somewhere in B
- **B → A** (`SUB_IN_ORIG`): percentage of B's tokens covered somewhere in A

These differ when files have different lengths.

Output format (with `-p -T` flags):
```
/path/to/original.java consists for 85 % of /path/to/submission.java material
/path/to/submission.java consists for 91 % of /path/to/original.java material
```
Empty output means 0% similarity (no matching runs of length ≥ `-r`).

### 4. Similarity metrics

From the two directional values the runner derives a single score:

| Metric | Formula | Behaviour |
|--------|---------|-----------|
| `MAX` | `max(orig→sub, sub→orig)` | More aggressive; captures short plagiarised files nearly fully covered |
| `AVG` | `(orig→sub + sub→orig) / 2` | Balanced; penalises length asymmetry |
| `SUB_IN_ORIG` | `sub→orig` only | Fraction of the submission's tokens covered by the original |
| `ORIG_IN_SUB` | `orig→sub` only | Fraction of the original's tokens reproduced in the submission |

### 5. Score caching

Per-submission directional scores are cached per `(case, min_run)`:

```
out/case-01-minrun-5_scores.csv
```

Runs that share the same `min_run` but differ only in `metric` or `threshold` reuse the cached scores automatically — SIM is not re-invoked. Pass `--force` to overwrite an existing cache.

---

## Setup

**Requirements:** C compiler (`cc`/`gcc`/`clang`), Python 3.10+, numpy, scikit-learn

```bash
# Clone and compile
git clone https://github.com/sauloq/sim
cd sim
make sim_java

# Place the binary in the sim tool folder
cp sim_java /path/to/experiments/sim/sim_java
```

The runner looks for `sim_java` next to itself (`experiments/sim/sim_java`) by default.

---

## Usage

Each invocation of `sim_runner.py` is a **run**: a fixed combination of parameters. The runner writes a per-run predictions CSV and appends one summary row to `out/sim_runs.csv`.

```bash
# Default run (threshold=0.5, min_run=5, metric=MAX)
python sim_runner.py

# Change metric only — reuses cached scores for min_run=5
python sim_runner.py --metric SUB_IN_ORIG --threshold 0.8

# Change min_run — triggers new SIM execution and new score cache
python sim_runner.py --min-run 10 --threshold 0.6

# Re-run SIM even if a cached score file exists
python sim_runner.py --min-run 5 --force

# Restrict to specific cases
python sim_runner.py --metric AVG --threshold 0.35 --cases case-01 case-02
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | `0.5` | Similarity cutoff for `predicted_plag`. |
| `--min-run` | `5` | Minimum token run length (`-r` flag). Changing this requires re-running SIM. |
| `--metric` | `MAX` | Which score to use: `MAX`, `AVG`, `SUB_IN_ORIG`, `ORIG_IN_SUB`. Changing this reuses the cached scores. |
| `--force` | off | Re-run SIM even when cached scores exist for the current `min_run`. |
| `--cases` | all | Run only named cases, e.g. `--cases case-01 case-03`. |
| `--dataset` | auto | Path to IR-Plag-Dataset directory. |
| `--sim-bin` | auto | Path to `sim_java` binary. |

---

## Output

```
experiments/sim/out/
  sim_runs.csv                                                   ← one row per run (params + metrics)
  SIM-Threshold-0.50-MinRun-5-Metric-MAX_results.csv
  SIM-Threshold-0.80-MinRun-10-Metric-SUB_IN_ORIG_results.csv
  ...
  case-01-minrun-5_scores.csv    ← cached (orig_in_sub, sub_in_orig) per submission
  case-01-minrun-10_scores.csv
  ...
```

### `sim_runs.csv` schema

| Column | Description |
|--------|-------------|
| `run_name` | Auto-generated identifier encoding all parameters, e.g. `SIM-Threshold-0.50-MinRun-5-Metric-MAX` |
| `min_run` | Value of `--min-run` for this run |
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

# Optimise AUC instead
../results-analyzer/.venv/bin/python suggest_next.py --metric auc

# More exploration, wider diversity between suggestions
../results-analyzer/.venv/bin/python suggest_next.py --xi 0.05 --diversity 0.6

# Restrict the min_run search range
../results-analyzer/.venv/bin/python suggest_next.py --min-run-range 5 20
```

The script also prints:
- A **gradient estimate** at the current best (finite differences on the GP mean), showing which direction each parameter should move and the effect of switching to each alternative metric.
- A **degenerate run warning** if any run has TN=0 or FN=0 (threshold too low — all submissions predicted as plagiarised, metrics are inflated).
- A **landscape summary** showing best and average metric grouped by metric choice and by `min_run`.

### `suggest_next.py` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--metric` | `f1` | Metric to maximise: `f1`, `auc`, or `accuracy`. |
| `--top` | `5` | Number of suggestions to display. |
| `--xi` | `0.01` | EI exploration bonus. `0` = pure exploitation, `0.1` = strong exploration. |
| `--diversity` | `0.4` | Minimum normalised distance between suggestions. Prevents clustering around one region. |
| `--min-run-range` | `3 20` | `min_run` search range (inclusive). |

---

## Known limitation — token normalisation on trivial exercises

Because SIM collapses all identifiers and string literals to the same token, two independently written Java programs that solve the same simple exercise (e.g. "print a string 5 times") produce nearly identical token sequences. This causes high false-positive rates on simple cases. Increasing `--min-run` and `--threshold` compensates for this.
