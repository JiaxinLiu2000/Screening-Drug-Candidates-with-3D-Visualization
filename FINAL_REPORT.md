# Learning molecular property models under distribution shift

*A machine-learning study. The data is chemical; the questions are not.*

68/68 integrity checks pass · 57/57 artefacts reproduce byte-identically · 76 neural training runs (8.7 GPU-hours) plus 60 learning-curve fits and 100 docking runs

## Abstract

We train and evaluate molecular property models on two tasks - 12-label toxicity classification (6143 molecules) and drug-target affinity regression (117193 measured pairs) - and ask four machine-learning questions: does a learned representation beat a fixed one, does that answer depend on training-set size, what happens under distribution shift, and can the model say when not to trust it. The headline results are four negative or counter-intuitive findings, each with a control: a graph network ties a random forest at every training-set size tested; roughly half of a pretrained model's apparent advantage was an untuned baseline; a 650M-parameter protein language model behaves mostly as an identity lookup except under target shift, where removing it entirely *improves* accuracy; and the two textbook confidence signals for selective prediction are worse than random under class imbalance, while a distance-to-training-set signal works. All artefacts reproduce byte-identically and every reported number is checked against the artefact it derives from.

## 1. Data and splits

| | Tox21 | hERG-large | KIBA | Davis |
| --- | --- | --- | --- | --- |
| task | 12-label classification | binary classification | pair regression | pair regression |
| size | 6143 molecules | 11052 molecules | 117193 pairs | 25772 pairs |
| entities | 12 endpoints, ~25% labels missing | 361 shared with Tox21 | 2030 drugs x 229 proteins | 68 drugs x 379 proteins |
| role | primary classification | external check | primary regression | secondary; 71% censored |

Preprocessing is one shared module for every dataset: parse, keep the largest covalent fragment, drop stereochemistry, deduplicate on InChIKey with a majority vote, filter to 10-60 heavy atoms and an organic element set. For Tox21 that takes 7831 merged rows to 6143 molecules (8 unparsable, 243 duplicates merged, 1437 filtered).

Five split protocols, all committed as index files and replayed rather than recomputed, 5 seeds each:

- **random** - the standard, and the weakest test
- **scaffold** - Bemis-Murcko groups never span subsets (2156 scaffolds, largest group 1260)
- **cold-drug / cold-target** - held-out drug or protein identities
- **scaffold-cold-drug** - held-out whole drug scaffolds; stricter than anything the drug-target literature reports

Why this matters more than it sounds: the three nested definitions of 'held out' give very different answers on identical data and models.

| KIBA split | what is held out | tuned RF Pearson | nearest-train Tanimoto |
| --- | --- | --- | --- |
| random | nothing | 0.8219 | 1.000 |
| cold-drug | drug identity | 0.7289 | 0.672 |
| scaffold-cold-drug | drug scaffolds | 0.6031 | 0.471 |

A cold-drug split guarantees unseen *molecules*, not unseen *chemotypes*: in an analogue-dense dataset a held-out drug still has a training neighbour at Tanimoto 0.672. How much generalisation a benchmark demands is a design choice, and it moves results by more than most modelling decisions do.

## 2. Models

| model | representation | parameters | trained |
| --- | --- | --- | --- |
| random forest | Morgan r=2, 2048 bits (+ amino-acid/dipeptide composition for pairs) | 500 trees | per endpoint / per split |
| GIN | learned from molecular graphs | 1,264,400 | from scratch |
| ChemBERTa-77M | pretrained chemical language model | 4,776,061 | fully fine-tuned |
| frozen ESM-2 650M + Morgan -> MLP | pretrained protein LM, frozen | ~3M (head only) | head only |

All neural models share the same heads, data, folds, seeds and loss, so a difference between configurations is a difference in the encoder.

## 3. Results

### 3.1 Classification

| dataset | setting | random split | scaffold split | drop |
| --- | --- | --- | --- | --- |
| Tox21 | standard | 0.8324 ± 0.0181 | 0.7611 ± 0.0227 | +0.0713 |
| hERG | standard | 0.8577 ± 0.0639 | 0.8665 ± 0.0630 | -0.0088 |
| hERG-large | disjoint | 0.8897 ± 0.0133 | 0.8471 ± 0.0160 | +0.0427 |
| hERG-large | full | 0.8946 ± 0.0120 | 0.8420 ± 0.0049 | +0.0526 |

The scaffold-split penalty on Tox21 is 0.0713 macro ROC-AUC. Bootstrapping test molecules gives [0.0510, 0.0919]; adding the spread across the five committed splits widens it to [0.0413, 0.1013], which still excludes zero. The independent hERG-large set reproduces the effect (+0.0526), so it is not a Tox21 artefact.

9 endpoint/split combinations hold fewer than 20 test positives in at least one seed. They are included in the macro average but not interpreted individually.

### 3.2 Regression, against published baselines

| KIBA, random split | concordance index | MSE | Pearson |
| --- | --- | --- | --- |
| **ours, random forest (tuned)** | **0.8420** | 0.2297 | **0.8219** |
| KronRLS | 0.782 | 0.411 | — |
| SimBoost (gradient-boosted trees, non-neural) | 0.836 | 0.222 | — |
| DeepDTA (CNN + CNN) | 0.863 | 0.194 | — |
| GraphDTA (GIN) | 0.891 | 0.139 | — |
| frozen ESM-1b + Morgan-2048, MLP head | — | — | 0.852 |

Protocols differ from ours, so these are reference points rather than a leaderboard we have entered. Placement, computed from the table: our tuned forest (0.8420) is above KronRLS and SimBoost and below DeepDTA and GraphDTA. Note what that ordering means - tuning one hyperparameter moved a plain random forest past a published gradient-boosting method, which is the same lesson as Finding 2. That placement is on concordance index only: on MSE we are behind DeepDTA and GraphDTA and SimBoost (0.2297 against 0.139 at best), so the ranking depends on which metric is quoted. We did not reach the state of the art and do not claim to.

## 4. What we actually learned

### Finding 1 — the graph network is tied, not beaten

| Tox21 training molecules | random forest | GIN | gap |
| --- | --- | --- | --- |
| 246 (5%) | 0.6345 ± 0.0327 | 0.6583 ± 0.0336 | +0.0238 |
| 492 (10%) | 0.6711 ± 0.0397 | 0.6832 ± 0.0281 | +0.0121 |
| 1229 (25%) | 0.7005 ± 0.0302 | 0.7364 ± 0.0332 | +0.0358 |
| 2458 (50%) | 0.7422 ± 0.0438 | 0.7527 ± 0.0299 | +0.0104 |
| 4915 (100%) | 0.7697 ± 0.0271 | 0.7665 ± 0.0295 | -0.0032 |

Comparing two models at one training-set size cannot tell you whether the loser lacked data. Subsampling the training fold while holding validation and test fixed answers it: the GIN leads by +0.0238 at 5% and the gap is -0.0032 at 100%, inside a seed-to-seed spread of 0.02-0.03. Both curves are still rising, so more data would help both; nothing in this range suggests the ordering changes. The claim is 'a 1.2M-parameter GNN buys nothing over fingerprints at this scale', not 'GNNs are worse'.

### Finding 2 — half the neural advantage was an untuned baseline

| KIBA split | RF `sqrt` | RF tuned | frozen MLP | lead vs untuned | lead vs tuned |
| --- | --- | --- | --- | --- | --- |
| random | 0.7841 | 0.8219 | 0.8510 | +0.0669 | +0.0291 |
| cold_drug | 0.7324 | 0.7289 | 0.5820 | -0.1505 | -0.1469 |
| cold_target | 0.5917 | 0.6438 | 0.5001 | -0.0917 | -0.1437 |

`max_features='sqrt'` lets each split of a 2048-bit sparse fingerprint see ~45 bits. Raising it to 0.3 moves the random-split forest from 0.7841 to 0.8219 - a larger change than most architecture differences measured here - and the frozen MLP's lead falls from +0.0669 to +0.0291. On the learning curve the lead at 5% data drops to +0.0058, i.e. nothing.

The tuning gain is itself regime-dependent, which is not how hyperparameters are usually discussed: +0.0378 on random, +0.0520 on cold-target, but -0.0036 on cold-drug. A search run on a random split would report a gain that does not exist in the regime we care about.

**A margin measured against an untuned baseline measures the tuning as much as the model.** This is the most transferable lesson in the project.

### Finding 3 — the protein language model is mostly an identity lookup

| KIBA split | both | drug only | protein only | shuffled protein | shuffle cost |
| --- | --- | --- | --- | --- | --- |
| random | 0.8514 | 0.6052 | 0.3236 | 0.8047 | +0.0467 |
| cold_drug | 0.5838 | 0.2552 | 0.3296 | 0.5601 | +0.0238 |
| cold_target | 0.4794 | 0.5720 | 0.1302 | 0.2244 | +0.2550 |

Giving every protein a *different* protein's ESM-2 embedding - a one-to-one relabelling that destroys biological similarity while preserving the ability to tell proteins apart - costs +0.0467 on the random split. A 650M-parameter model is earning almost nothing beyond 'which protein is this', because when every protein is seen in training a lookup table suffices.

On cold-target the same shuffle costs +0.2550, five times as much: an arbitrary key cannot transfer to an unseen protein, a biological embedding can. **The value of the pretrained representation is invisible on the split the field reports most often, and appears only under the shift it was meant to handle.**

And on cold-target the drug-only model (0.5720) *beats* the full model (0.4794) by +0.0927. Removing an entire input modality improves the result, because the head leans on protein features that do not transfer. More input information is an advantage only when train and test share it.

### Finding 4 — two textbook confidence signals are worse than random

| signal | PR-AUC, full coverage | PR-AUC, 50% coverage | actives retained | beats random |
| --- | --- | --- | --- | --- |
| distance from half | 0.3806 | 0.2079 | 23.7% | **no** |
| forest agreement | 0.3806 | 0.2024 | 23.5% | **no** |
| random | 0.3806 | 0.3983 | 48.3% | — |
| tanimoto to train | 0.3806 | 0.4904 | 53.8% | yes |

Abstaining where the score is near 0.5, or where the forest's trees disagree, *reduces* PR-AUC and discards three quarters of the actives. The reason is structural: in a 3-18% positive problem the uncertain region is the positive region, so score-based confidence abstains on the minority class and leaves easy negatives behind. Accuracy rises, which is exactly why accuracy is the wrong thing to watch. The signal that works is not an uncertainty estimate at all but a distance - maximum Tanimoto to the training set - reaching 0.4904 while keeping 53.8% of the actives. 'Have I seen chemistry like this?' is answerable when 'how sure am I?' is not.

### Finding 5 — multi-task sharing did not help these encoders

The four-way ablation plus pretrained single-task controls concluded **shared encoding does not help, and the chemical-space measurement explains why**: sharing from scratch costs -0.0869 toxicity and -0.0404 affinity against noise floors of 0.0310 / 0.0658; with the pretrained encoder it is +0.0016 / -0.0054. Two measurements bound the claim. The two input distributions are adjacent rather than overlapping (median cross-set nearest-neighbour Tanimoto 0.2597 against within-set 0.6000 and 0.7143); and the frozen MLP, with no learnable molecular encoder, beats every encoder we trained (0.8510 vs 0.6936). So this is a result about our encoders, not about multi-task learning.

## 5. Can the output be used?

**Not as probabilities.** The forests over-predict by ~1.7x on the scaffold split (0.157 predicted against 0.093 observed, ECE 0.0645). This is `class_weight='balanced'` behaving as designed - re-weighting the minority class to help ranking at the cost of the score no longer estimating P(positive). Isotonic regression on the validation fold brings per-fit ECE to 0.0253.

**Not outside the training chemistry.** Of 14 literature reference compounds, 5 are genuinely external and the model reproduces 40% of those expectations - n=5, too small to conclude from, reported because the classification half otherwise has no external check at all. The informative failure is cyclohexanol: 7 heavy atoms, below the pipeline's own 10-heavy-atom training filter, and the forest confidently predicts 0.936 for mitochondrial toxicity. An applicability-domain gate is not optional.

**Docking works, modestly.** AutoDock Vina AutoDock Vina v1.2.7 redocked all 20 cognate ligands into their own receptors, 100 runs over 5 seeds. Top-pose success at the standard 2.0 Å cutoff is 41.0%, rising to 76.0% if any of the nine returned poses counts - so the engine usually finds the right pose and often mis-ranks it. Median top-pose RMSD 2.152 Å. Stratified by conformation the active set scores 48.0% against 34.0%, but the active entries are systematically lower resolution (2.22 vs 1.44 Å) with higher pocket B-factors, so that difference cannot be attributed to the conformation. Redocking is the easiest docking task there is; this is a sanity check, not a validation.

## 6. Reproducibility

- **57/57 artefacts byte-identical** after re-running 6 scripts from a clean environment, including all 120 Tox21 forest fits.
- **68/68 integrity checks** that recompute each quantity from the artefact it derives from rather than trusting any script's own assertions. A check matching nothing fails rather than passing vacuously.
- Every split committed as an index file; every seed explicit; dependencies pinned as a complete `--no-deps` lock; every prompt logged.
- Two sources of run-to-run drift were found and removed rather than tolerated: thread-order nondeterminism in forest probabilities (metric tables are written at 10 decimal places) and a script-ordering bug where re-running the Tox21 grid silently dropped appended hERG rows.

Known bugs we found in our own work and fixed, listed because they are the part most likely to recur: ligand SDFs written with every bond as order 1; pocket geometry averaged across mutually exclusive alternate conformers; a leakage statistic computed for only one of the two splits it compared; and a PyTorch-Geometric batching rule that silently corrupted any attribute whose name contained `index`.

## 7. Limits

- **Not state of the art.** On KIBA-random our tuned forest reaches a concordance index of 0.8420 against DeepDTA's 0.863 and GraphDTA's 0.891 under their protocols.
- **Our encoders are not strong encoders.** An off-the-shelf frozen embedding beats everything we trained, so no finding here generalises to neural methods as a class.
- **Bootstrap scope.** Endpoint intervals resample test molecules; the combined interval adds split choice; neither covers retraining variance.
- **Budget asymmetry.** The ChemBERTa configurations hit their epoch cap in every run, so their numbers are a lower bound.
- **Sweep resolution.** 2 seeds per loss-weight cell; for the pretrained encoder the seed spread exceeds the weight spread, so the grid bounds the effect rather than locating an optimum.
- **No prospective validation.** Nothing here was tested on a molecule whose label was unknown at prediction time.
- **Stereochemistry dropped** in standardisation, so no model here can express stereo-dependent activity.

## 8. Where the detail lives

| document | contents |
| --- | --- |
| [`ML_REPORT.md`](ML_REPORT.md) | the modelling story in full, with every ablation table |
| [`VERIFICATION.md`](VERIFICATION.md) | integrity and reproduction evidence, and what it does not cover |
| [`RESULTS.md`](RESULTS.md) | every Phase 1-3 measurement with its artefact index |
| [`results/PHASE2_DECISIONS.md`](results/PHASE2_DECISIONS.md) | five data-foundation decisions |
| [`results/PHASE3_DECISIONS.md`](results/PHASE3_DECISIONS.md) | six modelling decisions |
| [`results/PHASE4_PROPOSAL.md`](results/PHASE4_PROPOSAL.md) | the external literature comparison |
| [`results/DATA_AUDIT.md`](results/DATA_AUDIT.md) | all integrity checks, one row each |
| [`PROMPTS.md`](PROMPTS.md) | every prompt used, with date and purpose |

Reproduce from scratch: `pip install --no-deps -r requirements.txt`, then `src/01` through `src/37` in order. `src/15_data_audit.py` checks the artefacts and `src/08_verify_repro.py --with-rf` checks that they rebuild byte-identically.

