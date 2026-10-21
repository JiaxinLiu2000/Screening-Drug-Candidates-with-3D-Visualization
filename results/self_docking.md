# Phase 4 D1 - Self-docking (redocking) benchmark

Program: **AutoDock Vina v1.2.7** (official pre-built Windows binary, sha256 `e0c4b2715e0c1a74...`). Receptors: Meeko mk_prepare_receptor (meeko 0.8.0, `--read_pdb -p -a --default_altloc A`). Ligands: Meeko MoleculePreparation + PDBQTWriterLegacy from RDKit ETKDGv3 conformer, randomSeed=0, hydrogens added to CCD bond-order SDF (neutral). Seeds 0-4 (Vina `--seed` 1-5: Vina --seed 0 means 'random seed' (verified: seed-0 runs were not reproducible across two runs), so project seed s is passed as --seed s+1), exhaustiveness 8, 9 poses, 1 CPU per run, box = recorded pocket box. RMSD: `rdkit.Chem.rdMolAlign.CalcRMS` (heavy-atom, symmetry-corrected over substructure matches, no re-alignment; RDKit 2026.09.1). `GetBestRMS` was deliberately not used: it superimposes the pose on the crystal ligand first, which would erase placement errors; `CalcRMS` is the same symmetry-aware matching evaluated in place.

Entries attempted: 20; preparation failures: 0. Docking runs: 100.

## Overall (top-ranked pose, success = RMSD <= 2.0 A)

- Success rate per seed: [0.4, 0.45, 0.45, 0.35, 0.4] -> mean 0.410 +/- 0.042
- Entries successful on all 5 seeds: 7/20; on at least one seed: 9/20
- Median top-pose RMSD over all runs: 2.152 A
- Best-of-9-poses success rate (sampling, not scoring): 0.760

## Seed spread

Per-entry seed spread (max - min top-pose RMSD over 5 seeds): median 0.135 A, max 7.955 A. Entries with spread > 2 A: 5HG5, 5XDK, 9BY4. Entries that succeed on some seeds but not others: 5HG5, 9BY4. A single-seed result for these entries is a coin flip; any between-entry or between-group RMSD difference smaller than the seed spread is not distinguishable from search noise.

## Per entry

| PDB | state | lig (HA) | res A | pocket B | RMSD mean +/- std | min-max | spread | success seeds | Vina kcal/mol | pocket residues dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| 3POZ | inactive | 03P (38) | 1.5 | 30.1 | 1.029 +/- 0.012 | 1.019-1.048 | 0.029 | 5/5 | -10.501 +/- 0.173 | - |
| 3W33 | inactive | W19 (35) | 1.7 | 31.7 | 0.590 +/- 0.028 | 0.566-0.637 | 0.071 | 5/5 | -9.660 +/- 0.033 | - |
| 4I22 | inactive | IRE (31) | 1.71 | 25.0 | 3.436 +/- 0.078 | 3.309-3.493 | 0.184 | 0/5 | -8.944 +/- 0.110 | - |
| 5CAS | active | 4ZQ (34) | 2.1 | 51.2 | 2.103 +/- 0.034 | 2.062-2.150 | 0.088 | 0/5 | -8.228 +/- 0.024 | - |
| 5CAU | active | 4ZR (34) | 2.25 | 54.9 | 1.377 +/- 0.077 | 1.303-1.463 | 0.160 | 5/5 | -8.292 +/- 0.013 | - |
| 5HG5 | inactive | 633 (35) | 1.52 | 17.2 | 4.460 +/- 2.518 | 1.693-6.306 | 4.613 | 2/5 | -8.695 +/- 0.093 | - |
| 5HG8 | inactive | 634 (28) | 1.42 | 17.2 | 1.349 +/- 0.140 | 1.162-1.557 | 0.395 | 5/5 | -8.321 +/- 0.087 | - |
| 5UG8 | inactive | 8BP (30) | 1.46 | 19.8 | 2.314 +/- 0.012 | 2.296-2.325 | 0.029 | 0/5 | -9.195 +/- 0.005 | - |
| 5UG9 | inactive | 8AM (32) | 1.33 | 18.1 | 2.392 +/- 0.031 | 2.356-2.439 | 0.083 | 0/5 | -9.411 +/- 0.034 | - |
| 5UGC | inactive | 8BS (30) | 1.58 | 28.6 | 2.143 +/- 0.012 | 2.124-2.154 | 0.030 | 0/5 | -8.913 +/- 0.015 | - |
| 5XDK | active | 8JC (40) | 2.35 | 31.5 | 6.355 +/- 3.676 | 2.322-9.062 | 6.740 | 0/5 | -8.292 +/- 0.123 | - |
| 6JRX | active | 9JO (44) | 2.2 | 51.9 | 2.242 +/- 0.021 | 2.220-2.277 | 0.057 | 0/5 | -9.627 +/- 0.036 | - |
| 6LUB | active | EUX (47) | 2.32 | 58.1 | 1.008 +/- 0.007 | 0.998-1.016 | 0.018 | 5/5 | -9.449 +/- 0.063 | - |
| 6LUD | active | YY3 (37) | 2.05 | 55.2 | 1.158 +/- 0.146 | 0.988-1.384 | 0.396 | 5/5 | -8.132 +/- 0.176 | - |
| 7ZYN | active | 0UN (35) | 2.3 | 62.4 | 8.190 +/- 0.294 | 7.868-8.443 | 0.575 | 0/5 | -7.949 +/- 0.075 | - |
| 8A27 | inactive | KY9 (48) | 1.07 | 18.4 | 2.354 +/- 0.021 | 2.323-2.375 | 0.052 | 0/5 | -11.181 +/- 0.032 | - |
| 8A2D | inactive | KXY (49) | 1.11 | 13.4 | 2.268 +/- 0.052 | 2.194-2.326 | 0.132 | 0/5 | -12.110 +/- 0.032 | - |
| 8F1X | active | R28 (43) | 2.3 | 65.4 | 1.089 +/- 0.076 | 0.983-1.163 | 0.180 | 5/5 | -9.319 +/- 0.204 | - |
| 8SC7 | active | D0D (31) | 1.98 | 52.1 | 6.417 +/- 0.053 | 6.331-6.468 | 0.137 | 0/5 | -8.574 +/- 0.038 | - |
| 9BY4 | active | Q6K (37) | 2.31 | 79.9 | 2.861 +/- 3.503 | 1.168-9.123 | 7.955 | 4/5 | -7.787 +/- 0.207 | - |

## Conformation stratification

| state | n | success rate (mean over seeds) | median top-pose RMSD | mean resolution A | mean pocket B |
|---|---|---|---|---|---|
| active | 10 | 0.480 +/- 0.045 | 2.071 | 2.216 | 56.260 |
| inactive | 10 | 0.340 +/- 0.055 | 2.283 | 1.440 | 21.950 |

**Confound (documented in `results/docking_benchmark_v2.md`):** the active-state entries are systematically lower resolution with higher pocket B-factors than the inactive-state entries (that report: active median resolution 2.27 A / median pocket B 55.0; inactive 1.48 A / 19.1), so any per-state difference above is entangled with site quality (and ligand chemotype) and cannot be attributed to the alphaC conformation. With 10 entries per group, one entry moves a group success rate by 0.1.

## Threshold sensitivity

7 entries have a 5-seed mean top-pose RMSD in (2.0, 2.5] A (5CAS, 5UG8, 5UG9, 5UGC, 6JRX, 8A27, 8A2D), so the 2.0 A success rate is threshold-sensitive: success over all runs is 0.410 at 2.0 A, 0.780 at 2.5 A and 0.780 at 3.0 A.

## What this does and does not establish

- It establishes that this receptor/ligand preparation, box definition and Vina configuration can (or cannot, per entry) reproduce the crystallographic pose when the receptor is the ligand's own holo structure - the easiest possible docking task.
- It does NOT establish cross-docking performance (ligand into a different receptor conformation), virtual-screening enrichment, or that Vina scores rank binding affinity. Vina scores here are reported only as program output; no affinity claim is made.
- The box is derived from the known ligand, so the search space is biased toward the answer; apo or unknown-site docking would be harder.
- Per-entry numbers are 5-seed samples; differences smaller than the reported seed spread are not real.
