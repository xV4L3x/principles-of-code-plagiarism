# CLAUDE.md — Progetto SCPD Evaluation

## Contesto del progetto

Sto scrivendo un paper di rassegna (survey) sullo **stato dell'arte degli strumenti e delle tecniche di rilevazione del plagio del codice sorgente (Source Code Plagiarism Detection, SCPD)**.

Il paper è strutturato in:
1. Introduzione
2. Tassonomia del plagio e tecniche di offuscamento
3. Tecniche e algoritmi di rilevamento (string-based, token-based, tree-based, graph-based, learning-based, LLM-based)
4. Rassegna degli strumenti (JPlag, SIM, Plaggie, Dolos, Oreo, CodeBERT, CodeLlama, MOSS, CodeMatch, Codequiry, GPT-4o)
5. **Metodologie di valutazione e dataset** ← sezione su cui stiamo lavorando
6. Nuove sfide: plagio da AI generativa
7. Conclusioni

---

## Capitolo 5 — Strategia sperimentale decisa

### Dataset scelto: IR-Plag-Dataset (Karnalim 2019)

- **Repository**: `github.com/oscarkarnalim/sourcecodeplagiarismdataset`
- **Lingua**: Java
- **Struttura**:
  ```
  IR-Plag-Dataset/
      case-XX/          (7 case totali: case-01 .. case-07)
          original/     ← un singolo file .java (il codice originale)
          plagiarized/
              L1/ .. L6/    ← 6 livelli di plagio (tassonomia Faidhi & Robinson)
                  01/ 02/ .. ← una submission per sottocartella
          non-plagiarized/
              01/ .. 15/    ← submission scritte indipendentemente
  ```
- **Ground truth**: binaria (plagiarized = True / False), esplicita nella struttura delle cartelle
- **Livelli di plagio** (corrispondenza con tassonomia del paper):
  - L1 → Type 1 (copia quasi verbatim)
  - L2 → Type 1-2 (piccole modifiche superficiali)
  - L3 → Type 2 (rinomina identificatori)
  - L4 → Type 2-3 (modifiche strutturali leggere)
  - L5 → Type 3 (modifiche strutturali significative)
  - L6 → Type 3-4 (refactoring avanzato)

### Perché NON BigCloneBench
Un paper del maggio 2025 (Krinke & Ragkhitwetsagul, arxiv 2505.04311) ha dimostrato che il 93% delle coppie WT3/T4 di BigCloneBench sono etichettate erroneamente. BigCloneBench rimane valido solo per Type 1-3 sintattico, ma per un confronto omogeneo tra tutti i tool abbiamo scelto IR-Plag.

### Perché NON SOCO 2014
Il dataset originale PAN@FIRE 2014 non è più scaricabile (link rotto sul sito PAN). AI-SOCO 2020 è un task completamente diverso (authorship identification, C++, nessuna ground truth di plagio).

---

## Tool da testare (in ordine di priorità)

### Fase 1 — Tool deterministici ✅ COMPLETATI
1. **JPlag** ✅ — token-based, RK-GST, open source (KIT). JAR da GitHub releases.
2. **Dolos** ✅ — token-based + winnowing + tree-sitter, npm package.
3. **SIM** ✅ — string-based, LCS, compilabile da sorgente C.
4. **Plaggie** ✅ — token-based, GST, Java-only, JAR standalone.

### Fase 2 — Tool learning-based
5. **Oreo** ← PROSSIMO — ibrido ML+IR+metriche, Java-only, GitHub: `Mondego/oreo`
6. **CodeBERT/GraphCodeBERT** ✅ — Transformer zero-shot, embedding cosine similarity
7. **CodeLlama** — LLM open source, approccio zero-shot con prompt

---

## Architettura run-based (tutti i tool da Dolos in poi)

Ogni tool seguente lo stesso pattern:

```
experiments/<tool>/
  <tool>_runner.py       ← runner principale
  suggest_next.py        ← advisor Bayesiano (GP + Expected Improvement)
  out/
    <tool>_runs.csv      ← una riga per run (params + metriche)
    <RunName>_results.csv         ← predictions CSV standard per run
    case-XX-<params>_scores.csv   ← cache punteggi grezzi (riusa tra run con stessa config)
```

### Colonne di `<tool>_runs.csv`
Ogni tool ha colonne specifiche per i suoi hyperparameter, ma condivide sempre:
`run_name, threshold, tp, fp, tn, fn, precision, recall, f1, accuracy, auc, mcc, predictions_csv`

### Score caching
- I punteggi grezzi sono costosi da calcolare (inference, Plaggie, SIM) e vengono cachati per evitare ricalcoli
- Cambiare solo `threshold` o `metric` riusa la cache → nessun ricalcolo
- Cambiare i parametri "costosi" (es. modello, pooling, min_tokens) invalida la cache

### `suggest_next.py`
- Ogni tool ha il suo `suggest_next.py` con spazio parametri specifico
- Usa GP (RBF kernel) + Expected Improvement per suggerire la prossima configurazione
- `--metric f1|auc|accuracy|mcc` — metrica da ottimizzare
- Legge `<tool>_runs.csv` e filtra le run degeneri (TN=0 o FN=0)

---

## Formato CSV standard (output di ogni runner)

Tutti i runner devono produrre un CSV con esattamente queste colonne:

```
case, level, submission_id, similarity, is_plagiarized, predicted_plag
```

- `case`: es. `case-01`
- `level`: `L1` .. `L6` oppure `non-plag`
- `submission_id`: es. `01`, `02`, ...
- `similarity`: float [0.0 - 1.0], punteggio grezzo dello strumento
- `is_plagiarized`: `True` / `False` (ground truth dal dataset)
- `predicted_plag`: `True` / `False` (similarity >= threshold)

---

## Note implementative per tool

### JPlag
- Usare `--mode RUN_AND_EXIT` per non aprire il browser
- `-t 5` (min tokens bassa) perché i file del dataset sono piccoli
- La similarity viene estratta come `max()` tra tutte le coppie trovate nel report
- Il report `.jplag` è uno ZIP: dentro ci sono `overview.json` e `comparisons/`

### SIM
- Score caching per `(case, min_run)`
- Cache: `out/case-XX-minrun-R_scores.csv` con colonne: `level, sub_id, is_plag, orig_in_sub, sub_in_orig`
- 4 metriche: MAX, AVG, SUB_IN_ORIG, ORIG_IN_SUB
- Sweet spot trovato: `min_run=9, threshold=0.55, metric=MAX` → MCC≈0.32, AUC≈0.70
- AUC ceiling ~0.70: la normalizzazione token di SIM identifica gli identificatori → L3+ ancora rilevabili, ma limite strutturale

### Plaggie
- Score caching per `(case, min_tokens)`
- Cache: `out/case-XX-mintokens-T_scores.csv`
- 5 metriche: MAX, AVG, ORIG_IN_SUB, SUB_IN_ORIG, PRODUCT
- AUC ceiling ~0.62, MCC max ~0.16: i nomi degli identificatori vengono preservati nel token stream → L3-L6 scarsamente rilevabili per design
- Usare `--build` la prima volta per scaricare, patchare e compilare Plaggie da SourceForge

### Dolos
- Usa tree-sitter per il parsing → produce AST fingerprints
- Score caching integrato nel tool stesso (Dolos non riesegue se non necessario)

### CodeBERT / GraphCodeBERT
- Runner zero-shot: embedding cosine similarity, nessun fine-tuning né whitening né anonimizzazione
- Score caching per `(case, model_short, max_tokens, stride, pooling)`
- **Problema anisotropia**: i modelli base (codebert-base, graphcodebert-base) hanno AUC < 0.5 (mean) o ~0.54 (cls) — i loro embedding raw sono tutti compressi nella stessa direzione nello spazio 768-dim, rendendo la cosine similarity non discriminativa
- **Soluzione**: usare `YoussefHassan/graphcodebert-plagiarism-detector` — fine-tuned per clone detection, spazio embedding genuinamente discriminativo
- **Migliore configurazione trovata**: `graphcodebert-plagiarism-detector + cls + threshold=0.38` → F1=0.9373, MCC=0.7364, AUC=0.8742
- CLS pooling è sempre meglio di mean pooling per questo task
- **Pair mode**: abbiamo sperimentato la modalità `[CLS] orig [SEP] sub [SEP]` ma produce AUC~0.5 (degenere) — rimosso

---

## Metriche e soglia

- **Precision**: TP / (TP + FP) — quante delle coppie segnalate sono davvero plagiate
- **Recall**: TP / (TP + FN) — quante delle coppie plagiate vengono trovate
- **F1**: media armonica di Precision e Recall
- **MCC** (Matthews Correlation Coefficient): metrica bilanciata preferita, robusta allo sbilanciamento delle classi
- **AUC**: potere discriminativo indipendente dalla soglia; run degeneri ottengono MCC=0.0
- **Soglia**: `suggest_next.py` cerca automaticamente la soglia ottimale via GP+EI

La soglia ottimale varia per tool: JPlag tende a valori alti (0.6-0.8), Dolos simile, SIM ~0.55, graphcodebert-plagiarism-detector ~0.38.

---

## Note importanti

- Java 21+ richiesto per JPlag recente (v5+). `java -version` per verificare.
- JPlag v6+ apre automaticamente il browser: usare `--mode RUN_AND_EXIT` per evitarlo.
- Il dataset IR-Plag ha file molto piccoli (decine di righe): abbassare `-t` (min tokens) a 5 per JPlag.
- SIM e Plaggie sono più robusti su file piccoli rispetto a JPlag per via della soglia minima token.
- Oreo, CodeBERT e CodeLlama richiedono GPU (Google Colab consigliato per Colab Pro con A100).
- Per CodeBERT su MacBook: usare `--device mps`

---

## Riferimenti bibliografici rilevanti (già nel paper)

- [1] Prechelt, Malpohl, Philippsen — JPlag (2002)
- [17] Sağlam et al. — Obfuscation-Resilient JPlag (ICSE 2024)
- [21] Maertens et al. — Dolos 2.0 (2023)
- [22] Maertens et al. — Dolos language-agnostic (2022)
- [24] Saini et al. — Oreo: Twilight Zone (2018)
- [16] Brach et al. — LLM for SCPD, GPT-4o 78.7% accuracy (FLLM 2024)
- Krinke & Ragkhitwetsagul — BigCloneBench misuse (arxiv 2505.04311, maggio 2025)
- Karnalim — IR-Plag-Dataset (Informatics in Education, 2019)
