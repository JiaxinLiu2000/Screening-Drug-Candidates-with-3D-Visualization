# Data audit

**70 passed, 0 failed, 0 skipped** of 70 checks.

Produced by `src/15_data_audit.py`. This audit does not re-run the pipeline and does not trust any script's own assertions: it recomputes each quantity from the committed artefacts and checks that independent sources agree — summary statistics against per-fit rows, pocket geometry against ligand coordinates, split partitions against the molecule tables, document text against the source tables, and the dependency lock against the installed environment.

## Tox21 molecule table

| check | result | measured |
| --- | --- | --- |
| row count matches the reported final_molecules | PASS | parquet rows 6143, stats final_molecules 6143 |
| InChIKeys unique and sorted (deduplication held) | PASS | duplicates 0, sorted True |
| every canonical SMILES re-parses and is canonical | PASS | unparsable 0, non-canonical 0 |
| InChIKey column matches the SMILES it sits next to | PASS | mismatched 0 of 6143 |
| heavy-atom and element filters actually hold | PASS | violations 0, heavy-atom range (10, 60) |
| labels are only 0, 1 or NaN and endpoints are the 12 expected | PASS | endpoints 12, out-of-domain values none |

## Tox21 splits

| check | result | measured |
| --- | --- | --- |
| index files align row-for-row with the molecule table | PASS | misaligned none across 10 files |
| subsets partition the set exactly, at 80/10/10 | PASS | random0:4914/614/615; random1:4914/614/615; ... (10 files, all within 1% of 80/10/10) |
| scaffold groups never span subsets, and random splits do | PASS | scaffold spanning groups {0}, random 242-260 (leakage the split removes) |
| committed scaffolds reproduce from the SMILES (seeded 250-molecule sample) | PASS | sample mismatches 0/250; 2156 unique scaffolds, largest group 1260 |

## Classification metrics

| check | result | measured |
| --- | --- | --- |
| row counts per dataset are the full grid | PASS | rows per dataset {'Tox21': 120, 'hERG': 10, 'hERG-large': 20}; per (dataset, setting) {('Tox21', 'standard'): 120, ('hERG', 'standard'): 10, ('hERG-large', 'disjoint'): 10, ('hERG-large', 'full'): 10} |
| every AUC is in [0, 1] and fold counts are consistent | PASS | out-of-range AUCs 0, impossible positive counts 0; ROC-AUC range 0.6120-0.9580 |
| summary mean/std recompute exactly from the per-fit rows | PASS | largest absolute discrepancy over 38 summary rows: 4.94e-11 |
| the macro row is the mean over endpoints, not over fits | PASS | random 0.8324 +/- 0.0181; scaffold 0.7611 +/- 0.0227 |

## hERG

| check | result | measured |
| --- | --- | --- |
| split files partition one consistent molecule set | PASS | molecule count in every file {588}, stats final_molecules 588 |
| the reported drop equals the summary rows it is drawn from | PASS | random 0.8577 - scaffold 0.8665 = -0.0088, quoted in md: True |

## Binding

| check | result | measured |
| --- | --- | --- |
| metric table is the full 30-fit grid with in-range values | PASS | rows 30, groups 6, out-of-range rows 0 |
| cold splits leak no entity between train and valid/test | PASS | leaking entities none across 20 cold-split files |
| split files cover every pair exactly once | PASS | files with incomplete pair coverage none of 30 |
| dataset statistics recompute from the split files | PASS | DAVIS 68x379=25772 pairs, density 1.0000; KIBA 2030x229=117193 pairs, density 0.2521 |
| Davis cold-drug test folds really are too small to be stable | PASS | Davis 7 test drugs, Pearson sd 0.1897; KIBA 203 test drugs, Pearson sd 0.0305 |

## Chemical space

| check | result | measured |
| --- | --- | --- |
| median nearest-neighbour Tanimoto recomputes to the reported value | PASS | 68 Davis molecules re-standardized; recomputed median 0.166667 vs stored 0.166667 |
| the verdict band matches the measured median | PASS | median 0.1667 -> Negligible overlap; JSON says Negligible overlap; md heading present |

## EGFR receptors

| check | result | measured |
| --- | --- | --- |
| ligand SDFs carry real bond orders and are erlotinib | PASS | 1M17: 17 aromatic, 1 triple, AAKJLRGGTJKAMG-UHFFFAOYSA-N; 4HJO: 17 aromatic, 1 triple, AAKJLRGGTJKAMG-UHFFFAOYSA-N |
| cleaned receptors contain protein only, one chain | PASS | 1M17: HETATM 0, HOH 0, chains ['A']; 4HJO: HETATM 0, HOH 0, chains ['A'] |
| pocket centre and box recompute from the ligand coordinates | PASS | 1M17: centre delta 0.000 A, box delta 0.000 A; 4HJO: centre delta 0.000 A, box delta 0.000 A |
| conformation label agrees with the measured salt-bridge distance | PASS | 1M17: 3.05 A -> active; 4HJO: 13.3 A -> inactive |

## Docking benchmark

| check | result | measured |
| --- | --- | --- |
| 20 structures, every referenced file present | PASS | records 20, missing files 0 |
| ligand SDFs match their JSON record (atoms, InChIKey, SMILES) | PASS | inconsistencies none across 20 structures |
| selection filters hold: resolution, heavy atoms, distinct chemistry | PASS | resolution 1.07-2.1 A, heavy atoms 28-49, 20 distinct ligands, erlotinib absent |
| benchmark receptors are protein-only, single chain | PASS | receptors with HETATM or extra chains none of 20 |
| pocket boxes recompute from the benchmark ligand coordinates | PASS | largest deviation 0.000 A over 20 structures; offenders none |
| conformation calls use the same 4 A threshold as 1M17 | PASS | 3 active / 17 inactive, matches the report: True |
| pairwise ligand similarity recomputes to the reported median | PASS | 190 pairs, median 0.159 (reported 0.159), max 0.831 (reported 0.831) |

## Documents

| check | result | measured |
| --- | --- | --- |
| headline numbers quoted in the markdown match their source tables | PASS | checked 4 headline values across SUMMARY.md, RESULTS.md, PHASE2_DECISIONS.md; mismatches none |
| reproducibility record reports a complete, all-identical run | PASS | 57/57 identical across 8 groups, scripts 6 |

## Environment

| check | result | measured |
| --- | --- | --- |
| dependency lock matches the installed environment exactly | PASS | 79 pins; installed-but-unpinned none; pinned-but-absent none |

## Repository

| check | result | measured |
| --- | --- | --- |
| no raw or processed data is tracked by git | PASS | tracked data files 0, committed artefacts 299 |
| every source module compiles | PASS | 58 modules under src/; compileall exit 0 |
| ablation checkpoints are newer than the metric table they produced | PASS | 6 checkpoints for configs ['A', 'B', 'C', 'D', 'E', 'F']; missing none; stale none |

## Phase 3 / KIBA overlap

| check | result | measured |
| --- | --- | --- |
| verdict band matches the measured median, calibration present | PASS | median 0.2597 -> Weak overlap; within-Tox21 0.6000, within-KIBA 0.7143; ratio recomputes: True |
| KIBA molecule count agrees with the binding tables | PASS | overlap report 2030 drugs, binding stats 2030 drugs |

## Phase 3 / benchmark v2

| check | result | measured |
| --- | --- | --- |
| self-docking set: files, SDF integrity, distinct chemistry | PASS | 20 entries, 20 distinct ligands; problems none |
| conformation labels and the reported balance agree | PASS | 10 active / 10 inactive; matches the report: True |
| pocket boxes recompute from the cognate ligand | PASS | largest deviation 0.000 A; offenders none |

## Phase 3 / hERG-large

| check | result | measured |
| --- | --- | --- |
| split files partition one consistent molecule set | PASS | full {11052} (= 11099 standardized - 47 unlabeled = 11052), disjoint {10691} (= 11052 - 361 shared with Tox21 = 10691); no duplicate keys, no scaffold group spanning subsets |
| Disjoint really shares no InChIKey with Tox21 | PASS | disjoint shares 0 InChIKeys with Tox21 (full setting shares 361) |
| molecule-count keys are unambiguous and self-consistent | PASS | 11099 standardized - 47 unlabeled = 11052 full; - 361 shared with Tox21 = 10691 disjoint |

## Phase 3 / ESM-2

| check | result | measured |
| --- | --- | --- |
| protein embedding cache matches the KIBA protein count | PASS | 229 proteins x 1280-d from esm2_t33_650M_UR50D; KIBA has 229 proteins; NaN: False |

## Phase 3 / attribution

| check | result | measured |
| --- | --- | --- |
| reported coverage recomputes from the per-molecule table | PASS | 252 alert-bearing molecules, top-k hit rate 0.409, median enrichment 0.961; missing figures 0 |
| the group-masking control agrees with the per-atom result | PASS | whole-alert masking beats the random same-size null in 46.0% of 250 molecules (median percentile 0.50); probes disagree on epoxide n=14, para_quinone n=2; none large enough (n>=50) to indicate a bug: True |

## Phase 3 / endpoint reliability

| check | result | measured |
| --- | --- | --- |
| bootstrap CIs bracket their point estimates | PASS | 24 Tox21 rows carry ROC-AUC CIs (roc_auc_ci_lo/roc_auc_ci_hi); point estimates outside their own interval: 0 |
| flagged endpoints really hold <20 test positives | PASS | 9 endpoint/split combinations below 20 test positives recomputed from the metric rows; absent from the report: none |
| the penalty CI is consistent with the point estimate | PASS | macro random-minus-scaffold drop 0.0713 recomputed from the summary table and quoted in the Phase 3 reliability report: True |
| PR-AUC CIs bracket their point estimates too | PASS | 24 Tox21 rows with PR-AUC CIs; point estimates outside their own interval: 0 |

## Phase 3 / ablation

| check | result | measured |
| --- | --- | --- |
| every configuration ran the full seed grid | PASS | main-comparison runs per config {'A': 5, 'B': 15, 'C': 15, 'D': 15, 'E': 5, 'F': 5} (expected {'A': 5, 'B': 15, 'C': 15, 'D': 15, 'E': 5, 'F': 5}); 76 runs in total |
| reported means recompute from the per-run table | PASS | largest discrepancy between report and per-run means: 0.00e+00 |
| the verdict follows from the measured effects | PASS | effects: scratch -0.0869/-0.0404, best-weight -0.0236/+0.0253, pretrained +0.0016/-0.0054 against noise floors 0.0310/0.0658 -> "shared encoding does not help, and the chemical-space measurement explains why" |
| neural configurations are evaluated on the committed folds | PASS | test-fold sizes identical to the random-forest runs for every KIBA split and seed; mismatches none |

## Phase 4 / learning curves

| check | result | measured |
| --- | --- | --- |
| curve grid is complete and aggregates recompute | PASS | 60 rows (expected 60); largest discrepancy between report and per-run means 1.11e-16 |
| subsampling preserved the fold sizes | PASS | train-fold sizes match fraction x full size for every cell; mismatches none |

## Phase 4 / abstention

| check | result | measured |
| --- | --- | --- |
| the naive confidence signal really is beaten by random | PASS | recompute discrepancy 5.55e-17; beats-random {'distance_from_half': False, 'forest_agreement': False, 'tanimoto_to_train': True}; best signal tanimoto_to_train |

## Phase 4 / binding follow-ups

| check | result | measured |
| --- | --- | --- |
| scaffold-cold split holds out whole scaffolds | PASS | drug leakage none; scaffold-cold Pearson 0.6031 vs identity-cold 0.7324 (harder: True) |

## Phase 4 / self-docking

| check | result | measured |
| --- | --- | --- |
| every docked entry has a pose and an RMSD | PASS | 20 entries x 5 seeds = 100 runs (expected 100); missing RMSDs 0 |

## Phase 4 / cold-drug diagnosis

| check | result | measured |
| --- | --- | --- |
| control predictors behave as controls must | PASS | global-mean control gives r=0 and CI=0.5 on both splits; problems none |

## Phase 4 / modality ablation

| check | result | measured |
| --- | --- | --- |
| aggregates recompute and the control behaves | PASS | 36 runs (expected 36); report-vs-per-run discrepancy 5.55e-17; splits where an ablated model BEAT the full model {'cold_target': 0.0927}; shuffle cost per split {'random': 0.0467, 'cold_drug': 0.0238, 'cold_target': 0.255} |

## Phase 4 / tuned floor

| check | result | measured |
| --- | --- | --- |
| the tuned baseline is recomputed, not overwritten | PASS | 30 tuned fits; report-vs-per-fit discrepancy 0.00e+00; tuning gain per split [0.0378, -0.0036, 0.052]; committed untuned table still has no max_features column: True |

## FINAL_REPORT.md

| check | result | measured |
| --- | --- | --- |
| headline numbers agree with their artefacts | PASS | 176 lines; absent headline values none; hand-written placement claim present: False |

## Models

| check | result | measured |
| --- | --- | --- |
| shipped models load, verify and honour their contract | PASS | 3 models + 229 protein embeddings; problems: none |
| serving paths reproduce the metrics their models were trained to | PASS | 11720 held-out pairs; affinity_frozen_mlp served 0.8487 vs trained 0.8487; multitask_gin served 0.6834 vs trained 0.6834; largest gap 9.14e-08 |

## Verdict

No inconsistency found. Every committed number reproduces from the artefact it is derived from, and no cross-document contradiction remains.

