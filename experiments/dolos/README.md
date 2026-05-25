# Dolos Runner

Evaluates [Dolos](https://dolos.ugent.be) (v2.9.3) over the IR-Plag-Dataset.

## How it works

Dolos tokenises Java source files with tree-sitter, builds k-gram fingerprints,
and identifies shared fingerprints between all file pairs via the Rabin-Karp
winnowing algorithm. It runs once per case on all files simultaneously, then
`dolos_runner.py` extracts per-submission similarity vs the reference original.

## Requirements

### Node.js 22 (required)

Dolos depends on `tree-sitter`, a native Node addon. The pre-built binary is
incompatible with Node 26 (V8 API change). **Node 22 LTS is required.**

```bash
# Install nvm if needed
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.nvm/nvm.sh

# Install and activate Node 22
nvm install 22
nvm use 22       # or: cd experiments/dolos && nvm use  (reads .nvmrc)
```

### Install Dolos

```bash
cd experiments/dolos
nvm use 22
npm install
```

## Usage

```bash
# Normal run (all cases, default settings)
python dolos_runner.py

# Custom kgram length and metric
python dolos_runner.py --kgram 10 --metric MAX --threshold 0.7

# Only specific cases
python dolos_runner.py --cases case-01 case-03

# Sweep mode — find the optimal (kgram, metric, threshold) combination by F1
python dolos_runner.py --sweep
python dolos_runner.py --sweep --sweep-kgrams 5 10 15 23
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--kgram` | 23 | k-gram length for fingerprinting |
| `--window` | 17 | Winnowing window size (in kgrams) |
| `--metric` | COMBINED | Similarity metric (see below) |
| `--threshold` | 0.5 | Decision threshold for `predicted_plag` |

### Similarity metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| `COMBINED` | `(origCovered + subCovered) / (origTotal + subTotal)` | Dolos default, symmetric |
| `ORIG_IN_SUB` | `origCovered / origTotal` | Fraction of original appearing in submission |
| `SUB_IN_ORIG` | `subCovered / subTotal` | Fraction of submission appearing in original |
| `MAX` | `max(ORIG_IN_SUB, SUB_IN_ORIG)` | Favours the smaller, plagiarising file |
| `AVG` | `(ORIG_IN_SUB + SUB_IN_ORIG) / 2` | Balanced directional average |

`COMBINED` is the raw similarity Dolos reports. `MAX` is analogous to the
metric used in JPlag/SIM evaluations and often gives better recall for
partial-copy plagiarism.

## Output

```
out/
  dolos_results.csv       standard CSV for evaluate.py
  case-01_report/         raw Dolos CSV report (files.csv, pairs.csv, ...)
  case-02_report/
  ...
  sweep_results.csv       sweep mode: all (kgram, metric, threshold) combos
  sweep_best.txt          sweep mode: top-20 + best config summary
```

### CSV format

```
case, level, submission_id, similarity, is_plagiarized, predicted_plag
```

See `experiments/README.md` for column definitions.

## Notes

- IR-Plag files are very small (tens of lines). A smaller `--kgram` (5–15)
  usually outperforms the default 23 — use `--sweep` to find the optimum.
- Dolos filters fingerprints that appear in > 90% of files by default
  (`-M 0.9`). This is fine for IR-Plag's ~20 submissions per case.
- Raw Dolos reports are preserved in `out/<case>_report/` for manual inspection
  via `dolos serve out/case-01_report`.
