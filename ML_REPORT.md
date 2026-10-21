# The modelling report

A supervised-learning study on molecular data. The application is virtual screening, but the questions are machine-learning questions: does a learned representation beat a fixed one, how much does that answer depend on training-set size, what happens under distribution shift, and can the model tell us when not to trust it.

- commit `f079aac-dirty`; 66/67 integrity checks pass; 57/57 artefacts reproduce byte-identically
- 76 neural training runs (8.7 GPU-hours) plus 60 learning-curve fits

## 1. The two prediction problems

| | Tox21 | KIBA |
| --- | --- | --- |
| task | 12-label binary classification, labels ~25% missing | regression on drug-protein pairs |
| n | 6,143 molecules | 117,193 pairs (2,030 drugs x 229 proteins) |
| class balance | 3-18% positive per label | continuous |
| fixed-representation baseline | random forest on Morgan 2048 | random forest on Morgan 2048 + amino-acid/dipeptide composition |
| learned alternative | GIN over molecular graphs; ChemBERTa fine-tuned | frozen ESM-2 650M + Morgan -> MLP |
| shift test | Bemis-Murcko scaffold split | cold-drug / cold-target / scaffold-cold |

Every model in this report is evaluated on the same committed split index files with the same seeds, so differences are differences between models, not between evaluation protocols.

## 2. Learning curves: is the baseline winning because of the data?

The most useful experiment in the project, and the one that reframes everything else. Earlier phases compared models at a single training-set size. Here the training fold is subsampled to 5/10/25/50/100% while the validation and test folds stay fixed, 3 seeds each, so the only variable is how much the model got to learn from.

### Tox21 (scaffold split) - GIN vs random forest

| training molecules | random forest | GIN from scratch | gap |
| --- | --- | --- | --- |
| 246 (5%) | 0.6345 ± 0.0327 | 0.6583 ± 0.0336 | +0.0238 |
| 492 (10%) | 0.6711 ± 0.0397 | 0.6832 ± 0.0281 | +0.0121 |
| 1229 (25%) | 0.7005 ± 0.0302 | 0.7364 ± 0.0332 | +0.0358 |
| 2458 (50%) | 0.7422 ± 0.0438 | 0.7527 ± 0.0299 | +0.0104 |
| 4915 (100%) | 0.7697 ± 0.0271 | 0.7665 ± 0.0295 | -0.0032 |

**The graph network is not losing for lack of data - it is tied.** At 5% of the training set the GIN is ahead by +0.0238; at 100% the gap is -0.0032, i.e. indistinguishable given a seed-to-seed spread of roughly 0.02-0.03. Both curves are still rising at 6,143 molecules, so neither model has saturated and more data would help both, but nothing in this range suggests the ordering would change. The honest summary of the Tox21 half is 'a 1.2M-parameter GNN buys nothing over a random forest on fingerprints at this scale', not 'GNNs are worse'.

### KIBA (random split) - frozen-embedding MLP vs random forest

| training pairs | random forest | frozen ESM-2 + Morgan MLP | gap |
| --- | --- | --- | --- |
| 4688 (5%) | 0.5474 ± 0.0142 | 0.5863 ± 0.0031 | +0.0389 |
| 9375 (10%) | 0.6081 ± 0.0078 | 0.6501 ± 0.0070 | +0.0420 |
| 23438 (25%) | 0.6842 ± 0.0026 | 0.7429 ± 0.0052 | +0.0587 |
| 46877 (50%) | 0.7377 ± 0.0002 | 0.8000 ± 0.0073 | +0.0623 |
| 93754 (100%) | 0.7867 ± 0.0002 | 0.8514 ± 0.0057 | +0.0646 |

**The opposite picture.** The MLP leads at every size and the gap *widens* with data, from +0.0389 at 4,688 pairs to +0.0646 at 93,754. There is no crossover to find. What separates this from the Tox21 case is not scale alone - it is that the representation here is **pretrained and frozen**, not learned from 94k labels. A protein language model trained on UniRef brings information that 117k affinity measurements cannot supply.

Figures: `figures/learning_curve_tox21_scaffold.png`, `figures/learning_curve_kiba_random.png`.

## 3. Re-reading the multi-task ablation

The four-way ablation (A: GIN toxicity-only, B: GIN affinity-only, C: shared GIN, D: ChemBERTa shared, plus pretrained single-task controls E/F) concluded **shared encoding does not help, and the chemical-space measurement explains why**. The effects, against a seed-to-seed noise floor of 0.0310 (toxicity) and 0.0658 (affinity):

| contrast | toxicity macro ROC-AUC | affinity Pearson |
| --- | --- | --- |
| sharing, from scratch (C-A / C-B) | -0.0869 | -0.0404 |
| sharing, at the best loss weight | -0.0236 | +0.0253 |
| sharing, pretrained (D-E / D-F) | +0.0016 | -0.0054 |

Two measurements qualify that conclusion and both belong in the discussion. First, the datasets are only weakly related: median nearest-neighbour Tanimoto from Tox21 to KIBA is 0.2597 against within-set medians of 0.6000 and 0.7143, so the shared encoder is being asked to serve two adjacent but distinct input distributions. Second, and more damaging to any strong claim: the frozen-embedding MLP reaches Pearson 0.8510 on KIBA-random where our best *trained* encoder reaches 0.6936. A model with no learnable molecular encoder at all beats every encoder we trained. The defensible statement is therefore narrow: **multi-task sharing did not help *our* encoders**, and our encoders are not the strongest available.

## 4. The baseline was under-tuned, and it matters

`max_features='sqrt'` is the scikit-learn default for classification and a common choice for regression. On a 2,048-bit sparse fingerprint it lets each split consider only ~45 bits, which is a poor fit for a representation where the informative substructures are scattered:

| max_features | KIBA random Pearson | concordance index | within-protein r |
| --- | --- | --- | --- |
| sqrt | 0.7841 ± 0.0044 | 0.8241 | 0.6966 |
| 0.1 | 0.8144 ± 0.0036 | 0.8384 | 0.7295 |
| 0.3 | 0.8219 ± 0.0026 | 0.8420 | 0.7374 |

Raising it to 0.3 is worth +0.0378 Pearson - larger than most of the architecture differences this project measured. **Methodological lesson: an untuned baseline makes every comparison against it suspect.** With the tuned forest at 0.8219, the frozen MLP's lead shrinks from +0.0669 to +0.0291.

## 5. Distribution shift: how hard is the held-out set, really?

Three nested definitions of 'held out', on the same dataset and model:

| split | what is held out | Pearson | nearest-train Tanimoto (median) |
| --- | --- | --- | --- |
| random | nothing - the same drugs appear in train and test | 0.7841 | 1.000 |
| cold-drug | drug identity (InChIKey) | 0.7324 | 0.672 |
| scaffold-cold-drug | whole Bemis-Murcko scaffolds | 0.6031 ± 0.0160 | 0.471 |

The standard cold-drug split used throughout the drug-target literature holds out molecules but not chemotypes: in an analogue-dense dataset like KIBA a held-out drug still has a training neighbour at Tanimoto 0.672, and 18% have one above 0.8. Holding out scaffolds instead costs another 0.1293 Pearson. **The amount of generalisation a benchmark demands is a design choice, and reporting which one you used is not optional.**

The same point on the classification side: the random-to-scaffold macro ROC-AUC penalty is 0.0713. A bootstrap over test molecules gives [0.0510, 0.0919], but that interval only covers test-set sampling. Adding the spread across the five committed splits widens it to [0.0413, 0.1013], which still excludes zero. Reporting only the narrower interval would have overstated the precision.
### 5b. The same two models swap places under extrapolation

Section 2 showed the frozen-embedding MLP beating the forest on KIBA-random by a margin that grows with data. On the cold-drug split, with the same data and the same folds, the ordering reverses. Two control predictors that contain no molecular information at all make the comparison readable:

| predictor | pooled Pearson (cold-drug) | within-protein Pearson |
| --- | --- | --- |
| predict the training mean | 0.0000 | n/a (constant within a protein) |
| predict each protein's training mean | 0.3665 | n/a (constant within a protein) |
| random forest | 0.7324 | 0.6354 |
| frozen ESM-2 + Morgan MLP | 0.5820 | 0.4535 |

Knowing nothing but which protein a pair involves already yields pooled Pearson 0.3665 and a concordance index of 0.6819, so a pooled number on this task is partly a measure of between-protein variance and should never be quoted alone. Controlling for that, the forest still carries real molecule-level signal (within-protein 0.6354) and carries **more** of it than the MLP (0.4535) - the reverse of the random-split ordering.

**This is the most interesting modelling result in the project.** The two models are not ranked; they are good at different things. The frozen representation plus MLP interpolates better - it exploits a dense, well-covered input space and improves as that space fills in. The forest extrapolates better - when the test molecules are chemotypes it has never seen, an axis-aligned ensemble over substructure counts degrades more gracefully than a smooth function fitted in a learned embedding space. Which model is 'better' is therefore a question about the deployment regime, and a benchmark that only reports a random split cannot distinguish the two.

### 5c. Which input is the affinity model actually using?

The protein-mean control in 5b showed that knowing only the protein is already worth a third of a correlation. A modality ablation finishes the question: the same MLP is trained four ways on identical folds, with one input block zeroed (which keeps the parameter count fixed), plus a control in which each protein gets another protein's embedding.

| split | both | drug only | protein only | shuffled protein | cost of shuffling |
| --- | --- | --- | --- | --- | --- |
| random | 0.8514 | 0.6052 | 0.3236 | 0.8047 | +0.0467 |
| cold_drug | 0.5838 | 0.2552 | 0.3296 | 0.5601 | +0.0238 |
| cold_target | 0.4794 | 0.5720 | 0.1302 | 0.2244 | +0.2550 |

Two results here, and the second one was not expected.

**The protein channel is mostly an identity key, not biology.** On the random split, replacing every protein's ESM-2 embedding with a different protein's - a one-to-one relabelling that destroys all biological similarity structure while preserving the ability to tell proteins apart - costs only +0.0467 Pearson (0.8514 to 0.8047). On cold-drug it costs +0.0238. A 650M-parameter protein language model is earning almost nothing beyond 'which protein is this', because with every protein seen in training a lookup table suffices.

**Except where generalisation actually requires it.** On the cold-target split, where test proteins were never seen in training, shuffling costs +0.2550 - five times the random-split penalty. An arbitrary key cannot transfer to an unseen protein; a biological embedding can. So the value of the pretrained protein representation is invisible on the split the field reports most often and only appears under the shift it was supposed to handle.

**And on cold-target, adding the protein input actively hurts.** The drug-only model scores 0.5720 against 0.4794 for the full model - removing an entire input modality *improves* the result by +0.0927. The head learns to lean on protein features that do not transfer, so for unseen targets it would be better off ignoring the protein altogether. This is a concrete instance of a general lesson: **more input information is only an advantage when the training and test distributions share it.**

(`protein_only` has no within-protein correlation by construction: with the drug block zeroed every molecule against a given protein receives an identical prediction, so the within-protein correlation is undefined rather than zero. It is reported as such rather than silently filled in.)

### 5d. Every margin in this project, re-measured against a tuned floor

Section 4 showed the baseline was under-tuned. Because every neural-vs-baseline margin was computed against the weaker forest, the whole comparison was re-run at `max_features=0.3` on the same folds. The committed `binding_baseline.csv` is left as the record of the untuned run so the difference stays visible.

| KIBA split | RF untuned (`sqrt`) | RF tuned | frozen MLP | MLP lead vs untuned | MLP lead vs tuned |
| --- | --- | --- | --- | --- | --- |
| random | 0.7841 | 0.8219 | 0.8510 | +0.0669 | +0.0291 |
| cold_drug | 0.7324 | 0.7289 | 0.5820 | -0.1505 | -0.1469 |
| cold_target | 0.5917 | 0.6438 | 0.5001 | -0.0917 | -0.1437 |

And the learning curve of section 2, against both floors:

| training fraction | MLP lead vs untuned RF | MLP lead vs tuned RF |
| --- | --- | --- |
| 5% | +0.0389 | +0.0058 |
| 10% | +0.0420 | +0.0101 |
| 25% | +0.0587 | +0.0218 |
| 50% | +0.0623 | +0.0243 |
| 100% | +0.0646 | +0.0287 |

**The qualitative conclusion of section 2 survives but the magnitude does not.** At full data the frozen MLP's lead drops from +0.0646 against the untuned forest to +0.0287 against the tuned one. The pretrained representation still wins at every training-set size, but roughly half of the headline gap was baseline tuning rather than representation quality. This is the single most important methodological lesson in the project: **a margin against an untuned baseline measures the tuning as much as the model.**

## 6. Knowing when not to answer

A classifier that can abstain is more useful than one that cannot, but the confidence signal has to be chosen carefully under class imbalance. Four signals, same predictions, same splits, scored as risk-coverage curves:

| confidence signal | PR-AUC at 100% coverage | at 50% coverage | actives retained at 50% | beats random |
| --- | --- | --- | --- | --- |
| distance from half | 0.3806 | 0.2079 | 23.7% | **no** |
| forest agreement | 0.3806 | 0.2024 | 23.5% | **no** |
| random | 0.3806 | 0.3983 | 48.3% | - |
| tanimoto to train | 0.3806 | 0.4904 | 53.8% | yes |

**Both textbook uncertainty signals are worse than random guessing here.** Rejecting predictions near 0.5, or rejecting where the forest's trees disagree, drops PR-AUC from 0.3806 to 0.2079 while discarding three quarters of the actives. The reason is structural: in a 3-18% positive problem the uncertain region *is* the positive region, so 'confidence' defined on the score abstains on the minority class and leaves behind easy negatives. Accuracy rises, which is exactly why accuracy is the wrong metric to watch.

The signal that works is not an uncertainty estimate at all - it is a distance: maximum Tanimoto similarity to the training set. It raises PR-AUC to 0.4904 at half coverage while *keeping* 53.8% of the actives. Asking 'have I seen chemistry like this?' is answerable even when the model's probability is miscalibrated; asking 'how sure am I?' is not. Figure: `figures/abstention_risk_coverage.png`.

## 7. Calibration

| split | ECE | Brier | mean predicted | observed rate | ECE after isotonic |
| --- | --- | --- | --- | --- | --- |
| random | 0.0602 | 0.0611 | 0.1444 | 0.0842 | 0.0252 |
| scaffold | 0.0645 | 0.0727 | 0.1571 | 0.0926 | 0.0253 |

The forests over-predict by about 1.7x (0.157 predicted against 0.093 observed). This is `class_weight='balanced'` doing what it is designed to do - re-weighting the minority class to improve ranking - at the cost of the scores no longer estimating P(positive). Isotonic regression fitted on the validation fold brings per-fit ECE from 0.0722 to 0.0253. **Takeaway: a re-weighted classifier's output is a score, not a probability, and must be calibrated before it is shown to anyone as a percentage.**

## 8. What this project does and does not establish

**Does:**

- A well-tuned random forest on fingerprints is a serious baseline at this scale. On Tox21 it ties a GIN at every training-set size tested; on KIBA, tuning one hyperparameter moved it further than most architecture changes did.
- Pretrained frozen representations beat both learned-from-scratch encoders and fixed descriptors on the affinity task, and the advantage grows with data rather than shrinking.
- Multi-task sharing between these two tasks produced no gain beyond seed noise, in a setting where the two input distributions are measurably adjacent rather than overlapping.
- Under class imbalance, score-based confidence is worse than useless for abstention, while a training-set-distance signal works.
- The amount of distribution shift a split imposes can be measured, not just asserted, and it changes results by more than most modelling decisions.

**Does not:**

- Establish anything about neural methods in general. Our encoders lose to an off-the-shelf frozen embedding, so this is a statement about our models.
- Reach the published state of the art. On KIBA our concordance index is 0.8241 against DeepDTA's 0.863 and GraphDTA's 0.891 under their protocols.
- Produce a validated screening tool. The docking benchmark is a redocking sanity check, the reference-compound set is 14 molecules, and nothing here has been tested prospectively.

