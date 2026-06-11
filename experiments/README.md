# Experiments — SCPD Evaluation

Empirical evaluation of Source Code Plagiarism Detection (SCPD) tools over the **IR-Plag-Dataset** (Karnalim 2019).

## Dataset — IR-Plag-Dataset

```
IR-Plag-Dataset/
  case-01/ .. case-07/          7 independent programming exercises
    original/                   single reference .java file
    plagiarized/
      L1/ .. L6/                6 plagiarism levels (Faidhi & Robinson taxonomy)
        01/  02/  ...           one submission per subfolder
    non-plagiarized/
      01/ .. 15/                15 independently written solutions
```

| Level | Type | Description |
|-------|------|-------------|
| L1 | Type 1 | Near-verbatim copy |
| L2 | Type 1–2 | Minor superficial edits (whitespace, comments) |
| L3 | Type 2 | Identifier renaming |
| L4 | Type 2–3 | Light structural changes |
| L5 | Type 3 | Significant structural modifications |
| L6 | Type 3–4 | Advanced refactoring |

Ground truth is encoded in the folder structure: everything under `plagiarized/` is positive, everything under `non-plagiarized/` is negative.

---

## Tools

Each tool lives in its own subdirectory and is self-contained:

| Folder | Tool | Technique | Status |
|--------|------|-----------|--------|
| `jplag/` | JPlag v5.1.0 | Token-based, Greedy String Tiling | Done |
| `dolos/` | Dolos | Token-based + k-gram fingerprinting + winnowing | Done |
| `sim/` | SIM | String-based, run matching | Done |
| `plaggie/` | Plaggie | Token-based, GST | Done |
| `oreo/` | Oreo | Hybrid ML+IR (SourcererCC + Siamese network) | Done |
| `codebert/` | CodeBERT | Learning-based, Transformer CLS embeddings | In progress |

---

## Run-based architecture

Each tool runner supports multiple **runs** — independent executions with different parameter combinations. Every run produces:

1. A **predictions CSV** named after the run (e.g. `JPlag-Threshold-0.30-MinTokens-5-Metric-AVG_results.csv`) following the standard format below.
2. A row appended to the tool's **runs CSV** (e.g. `jplag/out/jplag_runs.csv`) recording the parameters and all computed metrics for that run.

This makes it straightforward to sweep hyperparameters and compare configurations without overwriting previous results.

---

## Standard CSV format

Every predictions CSV produced by a tool runner has exactly these columns:

| Column | Type | Description |
|--------|------|-------------|
| `case` | string | Exercise identifier, e.g. `case-01` |
| `level` | string | `L1`–`L6` for plagiarised submissions, `non-plag` for negatives |
| `submission_id` | string | Submission number within its level, e.g. `03` |
| `similarity` | float [0–1] | Raw similarity score reported by the tool |
| `is_plagiarized` | bool | Ground truth (`True` = plagiarised) |
| `predicted_plag` | bool | `similarity >= threshold` |

**`similarity`** is the tool's raw score — do not compare absolute values across tools since each tool uses a different internal scale.

---

## Runs CSV format

Each tool's runs CSV (e.g. `jplag/out/jplag_runs.csv`) tracks every executed parameter configuration:

| Column | Description |
|--------|-------------|
| `run_name` | Auto-generated identifier encoding the parameter combination |
| *(tool-specific params)* | e.g. `min_tokens`, `threshold`, `similarity_metric` |
| `tp`, `fp`, `tn`, `fn` | Confusion matrix counts |
| `precision`, `recall`, `f1`, `accuracy` | Standard classification metrics |
| `auc` | ROC-AUC (threshold-independent discriminative power) |
| `mcc` | Matthews Correlation Coefficient (balanced metric, robust to class imbalance) |
| `predictions_csv` | Filename of the corresponding predictions CSV |

---

## Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| Precision | TP / (TP + FP) | Fraction of flagged pairs that are true plagiarism |
| Recall | TP / (TP + FN) | Fraction of plagiarised pairs that are detected |
| F1 | 2·P·R / (P + R) | Harmonic mean of precision and recall |
| Accuracy | (TP + TN) / N | Overall correctness; can be misleading when classes are imbalanced |
| AUC | Area under ROC curve | Threshold-independent; measures raw discriminative power |
| MCC | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Balanced single-number summary; preferred when positives outnumber negatives |

IR-Plag has ~355 plagiarised and ~105 non-plagiarised submissions per dataset pass, so **MCC and AUC** are more informative than accuracy and F1 alone.

---

## Results analyzer

`results-analyzer/analyze.py` reads all tools' runs CSVs, loads the corresponding predictions files, and produces comparative tables and figures.

```bash
cd experiments/results-analyzer

# Analyse all tools with available runs
.venv/bin/python analyze.py

# Restrict to specific tools or run names
.venv/bin/python analyze.py --tools JPlag
.venv/bin/python analyze.py --tools "JPlag-Threshold-0.30-MinTokens-5-Metric-AVG"
```

### Output

```
results-analyzer/out/
  tables/
    global_metrics.csv          ← precision, recall, F1, accuracy, AUC, MCC per run
    f1_by_level.csv             ← F1 per plagiarism level × run
    f1_by_case.csv              ← F1 per exercise case × run
  figures/
    01_similarity_distributions.png   ← box plots per run, grouped by level
    02_roc_curves.png                 ← ROC curves for all runs
    03_metrics_bar.png                ← precision / recall / F1 / accuracy / MCC bars
    04_f1_by_level_heatmap.png        ← heatmap: runs × levels
    05_f1_by_case_heatmap.png         ← heatmap: runs × cases
    06_similarity_plag_vs_nonplag.png ← violin: plag vs non-plag per run
```

The analyzer uses the **threshold stored by each runner** — it never re-optimises thresholds itself. Tools not yet migrated to the runs-CSV architecture simply do not appear in the output.
