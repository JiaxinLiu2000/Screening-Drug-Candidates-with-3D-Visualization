# Phase 4 proposal: what the external comparison shows, and what to test next

Our binding numbers had never been compared with anything outside this repository - a gap `VERIFICATION.md` lists explicitly. This document closes it, reports the two anomalies the comparison exposed, and proposes follow-up tests in priority order. Measured values come from `src/26_external_comparison.py`; published values are quoted from the papers named beside them and are **not** our measurements.

## 1. Where our binding baseline actually sits

Concordance index and MSE are what the DAVIS/KIBA literature reports, so ours are recomputed in that space from the committed per-fit table.

| KIBA, random split | CI | MSE | Pearson | source |
| --- | --- | --- | --- | --- |
| **ours - random forest, Morgan + AAC/dipeptide** | **0.8241** | **0.2721** | **0.7841** | measured, 5 seeds |
| KronRLS | 0.782 | 0.411 | - | DeepDTA, Bioinformatics 34(17):i821 (2018), Table 4 |
| SimBoost (gradient-boosted trees, non-neural) | 0.836 | 0.222 | - | DeepDTA, Bioinformatics 34(17):i821 (2018), Table 4 |
| DeepDTA (CNN + CNN) | 0.863 | 0.194 | - | DeepDTA, Bioinformatics 34(17):i821 (2018), Table 4 |
| GraphDTA (GIN) | 0.891 | 0.139 | - | GraphDTA, bioRxiv 684662v2 |
| frozen ESM-1b + Morgan-2048, MLP head | - | - | 0.852 | Otter-Knowledge, arXiv 2306.12802v3, Table 3 (TDC random split) |

| DAVIS, random split | CI | MSE | Pearson | source |
| --- | --- | --- | --- | --- |
| **ours - random forest, Morgan + AAC/dipeptide** | **0.8429** | **0.3821** | **0.6773** | measured, 5 seeds |
| KronRLS | 0.871 | 0.379 | - | DeepDTA, Bioinformatics 34(17):i821 (2018), Table 3 |
| SimBoost (gradient-boosted trees, non-neural) | 0.872 | 0.282 | - | DeepDTA, Bioinformatics 34(17):i821 (2018), Table 3 |
| DeepDTA (CNN + CNN) | 0.878 | 0.261 | - | DeepDTA, Bioinformatics 34(17):i821 (2018), Table 3 |
| GraphDTA (GIN) | 0.893 | 0.229 | - | GraphDTA, bioRxiv 684662v2 |
| frozen ESM-1b + Morgan-2048, MLP head | - | - | 0.805 | Otter-Knowledge, arXiv 2306.12802v3, Table 3 (TDC random split) |

**Reading.** On KIBA our forest (CI 0.824, MSE 0.272) lands between KronRLS (0.782 / 0.411) and SimBoost (0.836 / 0.222), below DeepDTA (0.863 / 0.194) and well below GraphDTA (0.891 / 0.139). On DAVIS it is nearer the field (0.843 vs 0.871-0.878), consistent with DAVIS being 71% censored and therefore easy to score on a rank metric. Protocols differ - DeepDTA uses a 1-of-6 held-out test with 5-fold CV for hyperparameters, we use 80/10/10 over pairs with 5 seeds - so these are reference points, not a leaderboard we have entered.

**This narrows one of our claims.** We reported that the random-forest floor beats our neural configurations on affinity. True *inside this repository*, but a published frozen-ESM-1b + Morgan MLP head reaches Pearson 0.852 on KIBA-random (Otter-Knowledge, arXiv 2306.12802v3, Table 3) where our best neural model reaches 0.7563 and our forest 0.7841. The defensible statement is: *our* neural models did not beat *our* forest. Whether neural models beat forests on this task is not something our experiment establishes, and the published number suggests the opposite.

## 2. Two anomalies the comparison exposed

### 2.1 Our KIBA cold-drug result looked too good - it is not leakage

Published cold-drug Pearson on KIBA is 0.576 against 0.852 random: the cold setting costs 0.28. Ours costs 0.05 (0.7841 -> 0.7324). Two measurements explain it, and neither is a defect in the split:

- **The split is genuinely cold.** 0 InChIKeys are shared between train and test across 203 held-out drugs. No molecule leaks.
- **The chemotypes are not held out.** Median nearest-training-set Tanimoto for a held-out drug is 0.672; 18.2% have a training analogue above 0.8 and 1.5% above 0.95. KIBA is dense analogue series, so our cold-drug split is *cold molecule, warm chemotype*.

### 2.2 Pooled Pearson flatters every model, ours included

Correlating over all test pairs at once rewards a model for knowing which protein is easy to bind. Computing the correlation **inside each protein** and averaging (proteins with at least 10 test pairs) removes that:

| KIBA split | pooled Pearson | within-protein Pearson | within-protein Spearman | proteins scored |
| --- | --- | --- | --- | --- |
| random | 0.7869 | 0.6976 | 0.5985 | 177 |
| cold_drug | 0.6983 | 0.5996 | 0.5214 | 180 |

Within-protein, cold-drug falls to 0.5996 - close to the published 0.576 - and that is the number the screening use case actually depends on, since screening ranks molecules against one target. **Recommendation: report within-protein correlation beside the pooled figure from now on.** Single seed (0), because the committed tables store metrics rather than predictions.

## 3. Dataset version reconciliation

| source | KIBA drugs | KIBA proteins | KIBA pairs |
| --- | --- | --- | --- |
| ours (PyTDC + our standardizer) | 2030 | 229 | 117193 |
| DeepDTA paper | 2111 | 229 | 118254 |
| Otter paper | 2068 | 299 | 118036 |

Ours is smaller because the shared standardizer drops molecules outside the 10-60 heavy-atom window and the allowed element set, then merges duplicates by InChIKey. Defensible, but it means our numbers are not strictly on the same data as the published ones; the protein-count difference against Otter (229 vs 299) is a TDC version difference we did not control.

## 4. What the five surveyed repositories are worth

Surveyed: `hani-rahmoune/-EGFR-L858R-mutation-aware-discovery-pipeline` (the URL without the leading hyphen 404s), `Drugrepurposing/drug_repurposing`, `youssefMohamed989/protein-drug-ai`, `premn-2025/MADDIS`, `kexinhuang12345/DeepPurpose`.

**Two of them must not be cited or imitated.** In `Drugrepurposing/drug_repurposing` the GNN is a sine function of the node id, `docking_engine.py` says in its own docstring that it *simulates* Vina and returns a hand-placed synthetic PDB, and the reported metrics (`auroc 0.982`) are hard-coded constants in `dataset.py`. In `premn-2025/MADDIS` the docking score is `12 - pIC50` from a QSAR model, the headline "-9.2 kcal/mol" sits inside an "Expected Output" block, and `molecular_docking.py` falls back to a heuristic scorer on any Vina failure while still labelling the result `vina_score`. That last one is the anti-pattern our pipeline must never adopt: a silent degradation that keeps the authoritative name.

Worth adopting, in order:

| # | lesson | source | status |
| --- | --- | --- | --- |
| 1 | Report CI and MSE beside Pearson so the numbers are comparable to the DTA literature | DeepPurpose metric set; DeepDTA/GraphDTA tables | **done in section 1** |
| 2 | Compare against published cold-split Pearson on the same TDC splits | Otter-Knowledge Table 3 | **done in sections 1-2** |
| 3 | Multi-seed docking noise study; call a difference real only if it exceeds the seed spread | EGFR pipeline (5 Vina seeds x 15 compounds; it caught a lucky outlier at seed 42) | proposed, task D1 |
| 4 | Vina on Windows needs the pre-built binary (`pip install` fails on the Boost dependency); box from the co-crystal ligand centroid, exhaustiveness 8, fixed seed | EGFR pipeline walkthrough | proposed, task D1 |
| 5 | Applicability domain + confidence tier on every served prediction, with per-target Tanimoto thresholds derived from leave-one-out neighbours | MADDIS `applicability_domain.py`; EGFR API | proposed, task S1 |
| 6 | Tune RF `max_features`: at `sqrt` the forest sees ~45 of 2048 fingerprint bits per split; `0.1` gained +0.03-0.05 scaffold R-squared | MADDIS ablation log | proposed, task B1 |
| 7 | Calibration metrics (ECE, Brier, reliability diagram, selective prediction) for the classifier the demo will expose | protein-drug-ai script `04` | proposed, task T2 |
| 8 | Literature reference compounds per endpoint as a sanity set, excluded if present in training | protein-drug-ai script `08` | proposed, task T1 |
| 9 | Report split-seed spread beside the bootstrap CI, so the interval covers split choice and not only test-set sampling | EGFR pipeline; DeepDTA std over 5 training sets | proposed, task T3 |

Skip: the DeepPurpose encoder zoo (Python 3.6-era, and we have already shown shared encoding does not help here), MD refinement, structure prediction, RL generators (the EGFR repo's own RL run collapsed from 80% to 8.6% uniqueness and they tell you not to use the checkpoint), and LLM agent orchestration.

What none of the five has, and we do: committed split index files with 5 seeds, 59 cross-artefact integrity checks, byte-identical reproduction, a `--no-deps` lock, and a null control for the attribution method. DeepPurpose runs one seed and fixes the validation seed at 1 regardless of the run seed.

## 5. Proposed Phase 4 tasks, in priority order

| id | task | why now | cost |
| --- | --- | --- | --- |
| **B2** | Reproduce one published neural config on our splits (DeepPurpose's DeepDTA recipe CNN+CNN, or a frozen-ESM + Morgan MLP head) | Tests whether our neural head is the weak link rather than neural models as such - the biggest threat to our Task 5 conclusion | 2-3 h GPU |
| **B1** | Re-tune the affinity forest (`max_features` 0.1 vs `sqrt`, scaffold-grouped CV) and re-run the 30-fit grid | Our floor may be under-tuned, and every neural-vs-floor claim rests on it | ~1 h CPU |
| **B3** | Add a scaffold-cold-drug split for KIBA (hold out whole Bemis-Murcko scaffolds, not just molecules) | Section 2.1 shows our cold-drug is cold-molecule only; this is the honest hard setting, and nobody in the field does it | ~1 h CPU |
| **D1** | Run the self-docking benchmark: Vina on all 20 cognate pairs, 5 seeds each, RMSD to the crystal pose, stratified by conformational state | The benchmark is assembled but unrun; the seed-noise protocol prevents a lucky-outlier conclusion | 3-4 h CPU |
| **D2** | Add DiffDock and compare pose accuracy against Vina on the same 20 pairs | The proposal promises this comparison; D1 is its prerequisite | GPU, install risk |
| **T1** | Literature reference-compound sanity set for Tox21 | Cheapest external check for the toxicity half, which currently has none | ~1 h |
| **T2** | Calibration of the toxicity classifier: ECE, Brier, reliability diagram, selective prediction | The demo shows probabilities, and uncalibrated probabilities shown to a user are a real risk | ~1 h CPU |
| **T3** | Combine split-seed spread with the bootstrap CI | Closes the admitted gap that the CI covers test-set sampling only | ~30 min |
| **S1** | Applicability domain + model card + limitations page for the service | A forest returns the training mean for out-of-domain input and looks confident doing it | ~2 h |

**Answer to "do we need further testing?": yes, and B2 is the one that matters.** Our headline negative result - shared encoding does not help, and the forest floor is not beaten - is safe as a statement about our own models, but a published frozen-embedding MLP beats everything we built, on the same dataset. Until B2 is run we cannot distinguish "multi-task learning does not help here" from "our encoder and head are under-built". Everything else on the list is additive; B2 is the one that can overturn a conclusion we have already written down.

