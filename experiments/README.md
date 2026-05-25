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

## Tools

Each tool lives in its own subdirectory and is self-contained:

| Folder | Tool | Technique | Status |
|--------|------|-----------|--------|
| `jplag/` | JPlag v5.1.0 | Token-based, Greedy String Tiling | Done |
| `dolos/` | Dolos | Token-based + winnowing + tree-sitter | Done |
| `sim/` | SIM | String-based, run matching | Done |
| `plaggie/` | Plaggie | Token-based, GST | Done |

## Standard CSV format

Every tool runner produces a CSV with exactly these columns:

| Column | Type | Description |
|--------|------|-------------|
| `case` | string | Exercise identifier, e.g. `case-01` |
| `level` | string | `L1`–`L6` for plagiarised submissions, `non-plag` for negatives |
| `submission_id` | string | Submission number within its level, e.g. `03` |
| `similarity` | float [0–1] | Raw similarity score reported by the tool |
| `is_plagiarized` | bool | Ground truth (`True` = plagiarised) |
| `predicted_plag` | bool | `similarity >= threshold` |

### How to read the results

- **`similarity`** is the tool's raw score — compare it across tools, but do not compare absolute values between different tools since each tool uses a different scale internally.
- **`is_plagiarized`** is the objective label derived from the dataset folder structure.
- **`predicted_plag`** depends on the chosen threshold. The default threshold in all runners is `0.5`; use `evaluate.py` to find the optimal threshold per tool.
- A row with `level = non-plag` and `is_plagiarized = False` is a **true negative** when `predicted_plag = False`, or a **false positive** when `predicted_plag = True`.
- A row with `is_plagiarized = True` is a **true positive** when `predicted_plag = True`, or a **false negative** when `predicted_plag = False`.

## Evaluation

```bash
# Single tool, auto-threshold
python evaluate.py --input jplag/out/jplag_results.csv

# Single tool, fixed threshold
python evaluate.py --input jplag/out/jplag_results.csv --threshold 0.7

# Multi-tool comparison
python evaluate.py --input jplag/out/jplag_results.csv dolos/out/dolos_results.csv
```

`evaluate.py` reports Precision, Recall, F1, and Accuracy — globally, per plagiarism level, and per case.
