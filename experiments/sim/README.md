# SIM — Runner

[SIM](https://github.com/sauloq/sim) (Software Similarity Tester) is a string-based plagiarism detector by Dick Grune. It tokenises source files using a language-specific lexer and finds the longest common runs of tokens between every pair of files. Unlike JPlag (which uses Greedy String Tiling), SIM uses a direct run-matching approach similar to longest common substring.

## How the analysis works

### 1. Tokenisation

SIM parses each `.java` file into a sequence of normalised tokens. Identifiers are all mapped to the same `IDF` token, string literals to `STR`, numbers to their numeric token, etc. This means identifier renaming (plagiarism Level 3) does **not** fool SIM — the token sequence remains identical after normalisation.

### 2. Run matching

SIM scans for the longest contiguous runs of matching tokens between the two files. The `-r N` parameter sets the minimum run length to count as a match. With the default of `24`, most files in the IR-Plag-Dataset (which are only tens of lines long) would produce no matches at all. We use `-r 5` to stay sensitive on small files.

### 3. Directional similarity

SIM reports similarity **asymmetrically** — it computes two values for a pair (A, B):

- **A → B**: percentage of A's tokens found somewhere in B
- **B → A**: percentage of B's tokens found somewhere in A

These differ when files have different lengths. For plagiarism detection we take the **MAX** of both directions: if a short plagiarised file is 100% covered by the original, that is a strong signal even if the original is much larger (and thus scores lower in the other direction).

Output format (with `-p -T` flags):
```
/path/to/original.java consists for 85 % of /path/to/submission.java material
/path/to/submission.java consists for 91 % of /path/to/original.java material
```
Empty output means 0% similarity (no matching runs of length ≥ `-r`).

### 4. Per-submission runs

The runner invokes SIM **once per submission** (pairwise against the original). This is simpler than the all-at-once approach used by JPlag, and avoids any ambiguity when parsing which pair a result belongs to. With IR-Plag's small files, each invocation takes milliseconds.

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

```bash
# Run all cases (default threshold 0.5)
python experiments/sim/sim_runner.py

# Run specific cases
python experiments/sim/sim_runner.py --cases case-01 case-02

# Use a fixed similarity threshold
python experiments/sim/sim_runner.py --threshold 0.4

# Custom binary path or output path
python experiments/sim/sim_runner.py --sim-bin /usr/local/bin/sim_java
```

## Output

```
experiments/sim/out/
  sim_results.csv       ← standard CSV (input to evaluate.py)
  case-01_raw.txt       ← raw SIM stdout for every pairwise run in that case
  case-02_raw.txt
  ...
```

The CSV follows the [standard format](../README.md#standard-csv-format) shared by all tool runners.

Each `case-XX_raw.txt` contains the raw SIM output for every `(original, submission)` pair in that case, labelled by filename. Open it to verify the exact similarity lines and debug the parser if needed.

## Key parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `-r` | 5 | IR-Plag files are short; default 24 misses almost everything |
| `-p` | — | Percentage output mode; gives `X consists for N% of Y material` |
| `-T` | — | Suppress per-file token-count headers; cleaner output for parsing |
| similarity metric | MAX | Captures the case where a small plagiarised file is fully covered by the original |
