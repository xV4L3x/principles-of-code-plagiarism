# Plaggie Runner

Evaluates [Plaggie](https://www.cs.hut.fi/Software/Plaggie/) (v1.1) over the IR-Plag-Dataset.

## How it works

Plaggie tokenises Java source files and applies Greedy String Tiling (GST) to find
maximal matching token sequences between every pair of submissions. It runs once per
case on all files simultaneously, then `plaggie_runner.py` extracts per-submission
similarity vs the reference original.

## Requirements

### Build Plaggie

Plaggie is distributed as source only. The runner auto-downloads from SourceForge,
patches a hardcoded-path bug, compiles, and creates `plaggie.jar`:

```bash
cd experiments/plaggie
python plaggie_runner.py --build
```

**Requires:** Java 8+ (`java`, `javac`, `jar` on PATH).

## Usage

```bash
# Normal run (all cases, default settings)
python plaggie_runner.py

# Custom token length and metric
python plaggie_runner.py --min-tokens 5 --metric MAX --threshold 0.6

# Only specific cases
python plaggie_runner.py --cases case-01 case-03
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-tokens` | 3 | Minimum matching token sequence length |
| `--metric` | MAX | Similarity metric (see below) |
| `--threshold` | 0.5 | Decision threshold for `predicted_plag` |

### Similarity metrics

Plaggie reports two directional similarity values per pair:
- **simA** = fraction of submission A's tokens found in submission B
- **simB** = fraction of submission B's tokens found in submission A

When comparing a submission against the original:
- `ORIG_IN_SUB` = fraction of original tokens found in submission
- `SUB_IN_ORIG` = fraction of submission tokens found in original
- `MAX` = `max(ORIG_IN_SUB, SUB_IN_ORIG)` — favours the smaller, plagiarising file
- `AVG` = `(ORIG_IN_SUB + SUB_IN_ORIG) / 2`
- `PRODUCT` = `ORIG_IN_SUB × SUB_IN_ORIG` — Plaggie's own "submission similarity" metric

`MAX` is the default and is analogous to the metric used in JPlag/SIM evaluations.

## Output

```
out/
  plaggie_results.csv     standard CSV for evaluate.py
```

### CSV format

```
case, level, submission_id, similarity, is_plagiarized, predicted_plag
```

See `experiments/README.md` for column definitions.

## Notes

- IR-Plag files are very small (tens of lines). The default `--min-tokens 3`
  is conservative; the original Plaggie default is 11. Use a smaller value for
  very short files to avoid missing matches.
- Plaggie reads `plaggie.properties` from its current working directory. The runner
  writes a per-run properties file to the temp directory for each case, so no manual
  configuration is needed.
- Plaggie v1.1 from SourceForge contains a hardcoded path override in `main()` added
  by a contributor. The `--build` step removes this line automatically before compiling.
