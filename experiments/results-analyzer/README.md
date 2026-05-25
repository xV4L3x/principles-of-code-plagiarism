# Results Analyzer

Compares the four SCPD tool runners (JPlag, Dolos, SIM, Plaggie) on the IR-Plag-Dataset
and produces publication-quality charts and summary tables.

## Setup

```bash
cd experiments/results-analyzer
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# Full run — auto-threshold per tool (maximises F1)
.venv/bin/python analyze.py

# Fixed threshold for all tools
.venv/bin/python analyze.py --threshold 0.5

# Subset of tools
.venv/bin/python analyze.py --tools JPlag Dolos
```

## Output

```
out/
  figures/
    01_similarity_distributions.png   box plots per tool, grouped by level + threshold line
    02_roc_curves.png                 ROC curves + AUC for all tools on one plot
    03_metrics_bar.png                Precision / Recall / F1 / Accuracy grouped bar chart
    04_f1_by_level_heatmap.png        heatmap: tools × plagiarism levels (L1–L6, non-plag)
    05_f1_by_case_heatmap.png         heatmap: tools × exercise cases
    06_similarity_plag_vs_nonplag.png violin: plagiarised vs non-plagiarised distributions

  tables/
    global_metrics.csv    threshold, TP/FP/TN/FN, precision, recall, F1, accuracy, AUC
    f1_by_level.csv       F1 per plagiarism level × tool
    f1_by_case.csv        F1 per exercise case × tool
```

## Reading the results

- **AUC** is the most threshold-independent metric — higher means the tool's scores are
  more separable between plagiarised and non-plagiarised submissions.
- **F1 by level** shows which obfuscation levels each tool handles well. All token-based
  tools score high on L1–L4; L5–L6 (structural refactoring) reveals differences.
- **non-plag F1 = 0** is expected: F1 measures detection of plagiarism, and the non-plag
  group contains no positives; use the false-positive count (FP) in `global_metrics.csv`
  and the box plot (figure 01) to assess specificity on that group.
- **Optimal threshold** is chosen per tool by sweeping [0, 1] at 200 steps to maximise F1
  over the whole dataset. A threshold of 0.00 means predict-all-positive wins F1 at that
  class balance (355 plag / 105 non-plag = 77 % positive).
