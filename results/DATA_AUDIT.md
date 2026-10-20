# Data audit

**38 passed, 0 failed, 0 skipped** of 38 checks.

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
| row counts per dataset are the full grid | PASS | rows per dataset {'Tox21': 120, 'hERG': 10} |
| every AUC is in [0, 1] and fold counts are consistent | PASS | out-of-range AUCs 0, impossible positive counts 0; ROC-AUC range 0.6120-0.9580 |
| summary mean/std recompute exactly from the per-fit rows | PASS | largest absolute discrepancy over 30 summary rows: 2.22e-16 |
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
| reproducibility record reports a complete, all-identical run | PASS | 36/36 identical across 7 groups, scripts 5 |

## Environment

| check | result | measured |
| --- | --- | --- |
| dependency lock matches the installed environment exactly | PASS | 49 pins; installed-but-unpinned none; pinned-but-absent none |

## Repository

| check | result | measured |
| --- | --- | --- |
| no raw or processed data is tracked by git | PASS | tracked data files 0, committed artefacts 146 |

## Verdict

No inconsistency found. Every committed number reproduces from the artefact it is derived from, and no cross-document contradiction remains.

