# JPlag — Runner

[JPlag](https://github.com/jplag/JPlag) is a token-based plagiarism detector developed at KIT. It tokenises source files using a language-specific frontend and then applies the **Greedy String Tiling (GST)** algorithm to find the longest common token sequences between every pair of submissions.

## How the analysis works

### 1. Tokenisation

JPlag parses each `.java` file into a flat sequence of language tokens (keywords, operators, identifiers, etc.). The `--min-tokens` parameter (`5` here) sets the minimum length of a matching token run that counts as a match — lower values increase sensitivity, which is necessary because the IR-Plag files are short (tens of lines).

### 2. Pairwise comparison

JPlag compares every submission against every other submission in the input directory. For each pair it computes two directional similarity scores:

- **first_similarity**: fraction of submission A's tokens covered by matches
- **second_similarity**: fraction of submission B's tokens covered by matches

From these it derives:
- `AVG = (first_similarity + second_similarity) / 2`
- `MAX = max(first_similarity, second_similarity)`

The runner extracts **MAX**, which captures the case where a small plagiarised file is 100% covered by matches even if the original is much larger.

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

## Setup

**Requirements:** Java 21, Python 3.10+

Download the JAR (Java 21 compatible):

```bash
curl -L -o experiments/jplag/jplag.jar \
  https://github.com/jplag/JPlag/releases/download/v5.1.0/jplag-5.1.0-jar-with-dependencies.jar
```

## Usage

```bash
# Run all cases (default threshold 0.5)
python experiments/jplag/jplag_runner.py

# Run specific cases
python experiments/jplag/jplag_runner.py --cases case-01 case-02

# Use a fixed similarity threshold
python experiments/jplag/jplag_runner.py --threshold 0.7

# Custom JAR or output path
python experiments/jplag/jplag_runner.py --jar /path/to/jplag.jar --output /path/to/results.csv
```

## Output

```
experiments/jplag/out/
  jplag_results.csv       ← standard CSV (input to evaluate.py)
  case-01_report.zip      ← raw JPlag report, one per case
  case-02_report.zip
  ...
```

The CSV follows the [standard format](../README.md#standard-csv-format) shared by all tool runners.

## Key parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--min-tokens` | 5 | IR-Plag files are short; default of 9 misses valid matches |
| `--shown-comparisons` | -1 | Store all pairs, not just the top-N |
| `--mode` | RUN | Run analysis only, do not open the browser viewer |
| similarity metric | MAX | Robust when plagiarised file is smaller than the original |
