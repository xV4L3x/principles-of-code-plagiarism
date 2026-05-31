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
    oreo_results.csv       ← standard CSV for evaluate.py
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

```bash
# Full pipeline (Phase 1 local + Phase 2+3 via Docker)
python oreo_runner.py

# Skip Phase 1 if blocks.file already exists
python oreo_runner.py --skip-phase1

# Skip Docker if predictions/ already exists (re-aggregate only)
python oreo_runner.py --skip-phase1 --skip-docker

# Phase 1 only (no Docker, inspect blocks.file)
python oreo_runner.py --phase1-only

# Specific cases
python oreo_runner.py --cases case-01 case-02

# Custom threshold (default: 0.5)
python oreo_runner.py --threshold 0.5
```

## Output

| File | Description |
|------|-------------|
| `out/oreo_results.csv` | Standard CSV for `evaluate.py` |
| `out/work/flat/` | 2-level input tree (`<case>_<submission>/`) |
| `out/work/blocks.file` | Phase 1: 24-metric vectors per Java method |
| `out/work/predictions/` | Phase 3: clone pair files from Predictor.py |

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
