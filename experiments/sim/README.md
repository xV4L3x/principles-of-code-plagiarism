# SIM — Runner

[SIM](https://github.com/sauloq/sim) (Software Similarity Tester) is a string-based plagiarism detector by Dick Grune. It tokenises source files using a language-specific lexer and finds the longest common runs of tokens between every pair of files. Unlike JPlag (which uses Greedy String Tiling), SIM uses a direct run-matching approach similar to longest common substring.

## How the analysis works

### 1. Tokenisation

SIM parses each `.java` file into a sequence of normalised tokens. Identifiers are all mapped to the same `IDF` token, string literals to `STR`, numbers to their numeric token, etc. This means identifier renaming (plagiarism Level 3) does **not** fool SIM — the token sequence remains identical after normalisation.

### 2. Run matching

SIM scans for the longest contiguous runs of matching tokens between the two files. The `-r N` parameter sets the minimum run length to count as a match. With the default of `24`, most files in the IR-Plag-Dataset (which are only tens of lines long) would produce no matches at all. We use small values of `-r` to stay sensitive on short files.

### 3. Directional similarity

SIM reports similarity **asymmetrically** — it computes two values for a pair (A, B):

- **A → B** (`ORIG_IN_SUB`): percentage of A's tokens found somewhere in B
- **B → A** (`SUB_IN_ORIG`): percentage of B's tokens found somewhere in A

These differ when files have different lengths.

Output format (with `-p -T` flags):
```
/path/to/original.java consists for 85 % of /path/to/submission.java material
/path/to/submission.java consists for 91 % of /path/to/original.java material
```
Empty output means 0% similarity (no matching runs of length ≥ `-r`).

From these two values the runner derives four **metrics**:

| Metric | Formula | When useful |
|--------|---------|-------------|
| `MAX` | max(A→B, B→A) | Short plagiarised file fully covered by original |
| `AVG` | (A→B + B→A) / 2 | Balanced; penalises length asymmetry |
| `SUB_IN_ORIG` | B→A only | Focus on how much of the submission comes from the original |
| `ORIG_IN_SUB` | A→B only | Focus on how much of the original is reproduced |

### 4. Per-submission runs

The runner invokes SIM **once per submission** (pairwise against the original). With IR-Plag's small files each invocation takes milliseconds.

### 5. Known limitation — token normalisation on trivial exercises

Because SIM collapses all identifiers and string literals to the same token, two independently written Java programs that solve the same trivial exercise (e.g. "print a string 5 times") produce nearly identical token sequences. This causes high false-positive rates on simple cases. The sweep mode exists specifically to find the `(min_run, metric, threshold)` triple that best compensates for this.

## Setup

**Requirements:** C compiler (`cc`/`gcc`/`clang`), Python 3.10+

```bash
# Clone and compile
git clone https://github.com/sauloq/sim
cd sim
make sim_java

# Place the binary in the sim tool folder
cp sim_java /path/to/experiments/sim/sim_java
```

The runner looks for `sim_java` next to itself (`experiments/sim/sim_java`) by default.

## Usage

### Normal mode

Runs SIM with fixed parameters and writes the standard CSV.

```bash
# Default parameters (min_run=5, metric=MAX, threshold=0.5)
python experiments/sim/sim_runner.py

# Specific cases only
python experiments/sim/sim_runner.py --cases case-01 case-02

# Override any parameter
python experiments/sim/sim_runner.py --min-run 10 --metric SUB_IN_ORIG --threshold 0.8

# Custom binary path
python experiments/sim/sim_runner.py --sim-bin /usr/local/bin/sim_java
```

### Sweep mode

Tries every combination of `(min_run × metric × threshold)` and ranks them by F1. Use this to find the best parameter set before committing to a final run.

```bash
# Full sweep (default min_run values: 3 5 8 10 15 20)
python experiments/sim/sim_runner.py --sweep

# Sweep over a custom set of min_run values
python experiments/sim/sim_runner.py --sweep --sweep-runs 5 10 15

# Sweep on specific cases only (faster for a quick check)
python experiments/sim/sim_runner.py --sweep --cases case-04 case-05
```

The sweep evaluates **6 × 4 × 19 = 456 combinations** (default settings) and prints a ranked top-20 table:

```
------------------------------------------------------------------------
 min_run  metric          threshold      F1     Acc    Prec     Rec
------------------------------------------------------------------------
      10  SUB_IN_ORIG          0.80  0.9123  0.9200  0.8900  0.9400
       8  AVG                  0.75  0.9050  0.9100  0.8800  0.9350
     ...
------------------------------------------------------------------------

BEST  →  min_run=10  metric=SUB_IN_ORIG  threshold=0.80  F1=0.9123  Accuracy=0.9200
```

It then prints the exact command to re-run with the winning parameters:

```bash
python sim_runner.py --min-run 10 --metric SUB_IN_ORIG --threshold 0.80
```

## Output

```
experiments/sim/out/
  sim_results.csv       ← standard CSV from normal mode (input to evaluate.py)
  case-01_raw.txt       ← raw SIM stdout per case (normal mode)
  ...
  sweep_results.csv     ← all 456 (min_run, metric, threshold) rows, sorted by F1
  sweep_best.txt        ← human-readable top-20 table and best combo
```

The CSV follows the [standard format](../README.md#standard-csv-format) shared by all tool runners.

Each `case-XX_raw.txt` contains the raw SIM output for every `(original, submission)` pair, labelled by filename. Open it to verify that the `consists for N %` lines are parsed correctly.

## Key parameters

| Parameter | Default | Reason |
|-----------|---------|--------|
| `-r` (`--min-run`) | 5 | IR-Plag files are short; default 24 misses almost everything |
| `-p` | always on | Percentage output mode; produces `X consists for N% of Y material` |
| `-T` | always on | Suppress per-file token-count headers for cleaner parsing |
| `--metric` | `MAX` | Starting point; use `--sweep` to find the optimal metric |
| `--threshold` | 0.5 | Starting point; sweep finds the optimal value automatically |
