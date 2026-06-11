# Oreo — Runner for IR-Plag-Dataset

Oreo (Saini et al., ESEC/FSE 2018) is a 3-phase hybrid clone detector for Java
that targets **Type 3–4 clones** by combining SourcererCC (IR) with a Siamese neural network.

## Architecture

```
Phase 1  Metric extraction  (local, Java JAR)
  Input : flat dir — one subdir per submission, containing *.java files
  Script: metricCalculationWorkManager.py → java-parser/dist/metricCalculator.jar
  Output: out/work/blocks.file  (24 code metrics per Java method)

Phase 2  Candidate generation  (Docker, SourcererCC)
  Input : out/work/blocks.file
  Output: oreo/results/candidates/<port>/*.txt  (token-similar method pairs)

Phase 3  ML classification  (Docker, Siamese network)
  Input : candidates + ml_model/oreo_model_fse.h5
  Output: oreo/results/predictions/*.txt  (predicted clone pairs)
          format: folder1,file1,start1,end1,folder2,file2,start2,end2
```

Phases 2 and 3 always run inside Docker to faithfully reproduce the original
2018 environment (Java 11, Python 3.6, TensorFlow 1.5, Keras 2.1.3).

Similarity is reported as `matched_original_methods / total_original_methods`,
where a method is "matched" if Oreo predicted it as a clone of a method in the
submission being evaluated.

## Directory layout

```
oreo/
  oreo_runner.py          ← this runner
  suggest_next.py         ← Bayesian threshold search
  Dockerfile              ← Phase 2 + Phase 3 environment
  docker_entrypoint.sh    ← entrypoint script run inside Docker
  oreo-artifact/          ← cloned from github.com/Mondego/oreo-artifact
    oreo/
      java-parser/
        dist/metricCalculator.jar      ← pre-built, no ant needed for Phase 1
      clone-detector/
        src/  build.xml  controller.py ← rebuilt by Docker on each run
      python_scripts/
        metricCalculationWorkManager.py
        Predictor.py
      ml_model/
        oreo_model_fse.h5              ← pre-trained Siamese model
  out/
    oreo_runs.csv          ← one row per run (params + metrics)
    oreo_scores.csv        ← score cache (reused across threshold sweeps)
    Oreo-Threshold-0.50_results.csv   ← per-run predictions CSV
    work/
      flat/                ← 2-level input tree (case_submission/file.java)
      blocks.file          ← Phase 1 output
      predictions/         ← Phase 3 output (copied from Docker)
```

## Setup (one-time)

### 1. Build the Docker image

```bash
cd experiments/oreo
docker build --platform=linux/amd64 -t oreo-runner .
```

This installs the exact original dependencies:
- Python 3.6, TensorFlow 1.5.0, Keras 2.1.3
- Java 11 + Apache Ant (to rebuild the SourcererCC JAR inside the container)

The image takes ~5 minutes to build (downloading TF 1.5 wheel).

### 2. Verify

```bash
python oreo_runner.py --phase1-only   # runs metric extraction only, no Docker
```

## Usage

Each invocation is a **run**: a named combination of parameters. The runner writes a per-run predictions CSV and appends one summary row to `out/oreo_runs.csv`.

```bash
# Full pipeline (Phase 1 local + Phase 2+3 via Docker)
python oreo_runner.py

# Threshold sweep — reuses score cache, no Docker needed
python oreo_runner.py --threshold 0.3
python oreo_runner.py --threshold 0.7

# Skip Phase 1 if blocks.file already exists (Docker still runs)
python oreo_runner.py --skip-phase1

# Phase 1 only (no Docker, inspect blocks.file)
python oreo_runner.py --phase1-only

# Re-run from scratch (ignores score cache)
python oreo_runner.py --force

# Specific cases
python oreo_runner.py --cases case-01 case-02
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | `0.5` | Similarity cutoff for `predicted_plag`. Post-hoc — reuses score cache. |
| `--skip-phase1` | off | Reuse existing `blocks.file` (Docker still runs unless score cache exists). |
| `--phase1-only` | off | Run only Phase 1, then stop. Useful for debugging. |
| `--force` | off | Ignore score cache and re-run the full pipeline. |
| `--cases` | all | Run only named cases, e.g. `--cases case-01 case-03`. |
| `--oreo-dir` | `./oreo-artifact` | Path to the oreo-artifact directory. |
| `--dataset` | auto | Path to IR-Plag-Dataset directory. |

### Score cache

After the first full run, `out/oreo_scores.csv` caches the per-submission similarities for all cases. Subsequent runs with a different `--threshold` load this cache directly — no Docker invocation needed.

To force a re-run of the full pipeline (e.g. after changing the dataset), pass `--force`.

## Output

| File | Description |
|------|-------------|
| `out/oreo_runs.csv` | One row per run with all parameters and metrics |
| `out/oreo_scores.csv` | Score cache: per-submission similarities (reused across threshold sweeps) |
| `out/Oreo-Threshold-0.50_results.csv` | Per-run predictions CSV |
| `out/work/flat/` | 2-level input tree (`<case>_<submission>/`) |
| `out/work/blocks.file` | Phase 1: 24-metric vectors per Java method |
| `out/work/predictions/` | Phase 3: clone pair files from Predictor.py |

### `oreo_runs.csv` schema

| Column | Description |
|--------|-------------|
| `run_name` | Auto-generated identifier, e.g. `Oreo-Threshold-0.50` |
| `threshold` | Similarity cutoff used |
| `tp`, `fp`, `tn`, `fn` | Confusion matrix counts |
| `precision`, `recall`, `f1`, `accuracy` | Standard classification metrics |
| `auc` | ROC-AUC (threshold-independent) |
| `mcc` | Matthews Correlation Coefficient — balanced metric |
| `predictions_csv` | Filename of the corresponding predictions CSV |

If a run with the same `run_name` is re-executed, its row is **overwritten** in place.

### Predictions CSV schema

Follows the [standard format](../README.md#standard-csv-format) shared by all tool runners.

## Hyperparameter search — `suggest_next.py`

After accumulating several runs at different thresholds, `suggest_next.py` fits a **Gaussian Process** surrogate and recommends the next threshold to try via **Expected Improvement** (EI).

```bash
# Suggest next 5 thresholds optimising F1
../results-analyzer/.venv/bin/python suggest_next.py

# Optimise MCC
../results-analyzer/.venv/bin/python suggest_next.py --metric mcc

# More exploration
../results-analyzer/.venv/bin/python suggest_next.py --metric mcc --xi 0.05
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--metric` | `f1` | Metric to maximise: `f1`, `auc`, `accuracy`, or `mcc`. |
| `--top` | `5` | Number of suggestions to display. |
| `--xi` | `0.01` | EI exploration bonus. |

## Source patches applied to oreo-artifact

Two patches were needed to run Oreo on modern Java (11) inside Docker:

**`clone-detector/src/.../SearchManager.java`** — replaced `EProperties`
(broken variable-substitution in version 1.1.5 on Java 11) with a
`TypedProperties` inner class that extends standard `java.util.Properties` and
adds `getInt()`, `getString()`, `getBoolean()` typed accessors.

**`clone-detector/sourcerer-cc.properties`** — replaced `OUTPUT_DIR=${NODE_PREFIX}/output`
with the literal `OUTPUT_DIR=NODE_1/output` to avoid EProperties forward-reference failure.

`docker_entrypoint.sh` also reduces the JVM heap from 10 GB to 2 GB (sufficient for IR-Plag).

## Known limitations

**Binary similarity**: Oreo outputs binary clone pairs (clone / not-clone).
Similarity is proxied as `matched_methods / total_original_methods`, yielding
values in {0.0, 0.5, 1.0} depending on how many original methods a case has.
`evaluate.py` will auto-select an optimal threshold.

**Method-level detection**: IR-Plag files are small (20–100 lines). Cases with
only 1–2 distinct methods produce coarse similarity scores. Some submissions may
score 0.0 because their methods fall below SourcererCC's minimum token threshold.

**False positives**: Oreo targets Type 3–4 near-miss clones and can flag
non-plagiarized submissions when their code is structurally similar to the
original (e.g., common algorithm patterns). This is visible in the results.

## References

- Saini V. et al. — *Oreo: Detection of Clones in the Twilight Zone*,
  ESEC/FSE 2018. [arxiv 1806.05837](https://arxiv.org/abs/1806.05837)
- Mondego/oreo-artifact — https://github.com/Mondego/oreo-artifact
