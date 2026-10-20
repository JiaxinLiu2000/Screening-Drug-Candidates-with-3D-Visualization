# Prompt log

Course requirement: every prompt given to the AI assistant is logged here with
the date it was issued and what it was used for. The *verbatim* message typed by
the user is quoted first; any file the message referenced is reproduced
underneath it, inside a fence, so the attachment's own `#` headings cannot be
confused with this log's structure.

---

## Prompt 1 — 2026-10-19

**Verbatim message:**

```text
按这个要求，获得需要的数据：C:\Users\Admin\Documents\WeChat Files\wxid_icmox6yaq5k022\FileStorage\File\2026-10\claude-code-prompt-preliminary-results.md
```

("Get the required data according to these requirements: <path>")

**Purpose:** set up the whole repository and produce the four preliminary values
plus the 3D viewer figure for the CS 6140 proposal (Tasks 1–6).

**Referenced file, `claude-code-prompt-preliminary-results.md`, in full:**

````markdown
# Task: Preliminary results for a CS 6140 ML final project proposal

## Context

We are building a multi-objective virtual screening tool (toxicity + binding
affinity prediction, with 3D pose visualization). The proposal is due in 2 days
and needs real preliminary numbers. Set up a clean, reproducible repository and
produce four specific values plus one figure. Work in Python 3.11.

## Repository layout to create

```
.
├── README.md
├── requirements.txt          # pinned versions
├── PROMPTS.md                # log every prompt used, with date + purpose
├── data/
│   ├── raw/                  # gitignored
│   └── processed/            # gitignored
├── splits/                   # COMMITTED - the split index files
├── receptors/                # COMMITTED - cleaned PDB files
├── results/                  # COMMITTED - metric tables + summary
├── figures/                  # COMMITTED - viewer html + screenshots
└── src/
    ├── 01_download_tox21.py
    ├── 02_standardize.py
    ├── 03_splits.py
    ├── 04_baseline_rf.py
    └── 05_egfr_structure.py
```

## Task 1 — Download Tox21 via Therapeutics Data Commons

Use PyTDC. Tox21 is exposed as 12 separate single-prediction datasets, one per
assay endpoint — retrieve the list of label names programmatically rather than
hardcoding it, then download each endpoint.

Merge all 12 into ONE wide table keyed by molecule, with one column per endpoint
and NaN where a compound was not tested on that endpoint. Report the exact shape
and the per-endpoint positive rate.

## Task 2 — Standardize and deduplicate

With RDKit, for every SMILES:

1. Parse; log and drop anything that fails to parse (report the count)
2. Strip salts: keep the largest covalently connected fragment
3. Remove stereochemistry (we are simplifying for v1 — note this in README)
4. Generate canonical SMILES and InChIKey
5. Deduplicate on InChIKey. When duplicates disagree on a label, keep the
   majority value; if tied, set NaN. Report how many duplicates were merged.
6. Filter to 10–60 heavy atoms and to the elements C, N, O, S, P, F, Cl, Br, I,
   H only. Report how many were removed.

Write the result to `data/processed/tox21_clean.parquet`.

## Task 3 — Build both splits ONCE, on the merged molecule set

Both splits must be computed on the deduplicated molecule set before any feature
computation, and both must be saved as index files in `splits/` so results are
reproducible even if RDKit changes version.

- `random_split`: 80/10/10 train/valid/test, seed-controlled
- `scaffold_split`: compute the Bemis-Murcko scaffold SMILES for each molecule,
  group molecules by identical scaffold, sort groups by size descending, then
  greedily fill train until it reaches 80%, then valid to 10%, then test. Whole
  scaffold groups must never be split across sets.

Report the number of unique scaffolds and the size of the largest group.

## Task 4 — Random forest baseline

Features: Morgan fingerprint, radius 2, 2048 bits, from RDKit's current
MorganGenerator API (not the deprecated `GetMorganFingerprintAsBitVect`).

For each of the 12 endpoints, independently:

- Train sklearn `RandomForestClassifier` (500 trees, `class_weight='balanced'`,
  `n_jobs=-1`) on the training rows that have a non-null label for that endpoint
- Evaluate on the test rows that have a non-null label
- Record ROC-AUC and PR-AUC

Do this for BOTH splits, with 5 seeds each (0, 1, 2, 3, 4), where the seed
controls both the split and the forest.

Output `results/baseline_metrics.csv` with one row per
`(split, seed, endpoint, roc_auc, pr_auc)`, and `results/baseline_summary.csv`
with the mean and standard deviation across seeds, both per endpoint and
averaged over the 12 endpoints.

## Task 5 — EGFR structure

Download both 1M17 and 4HJO from RCSB. For each:

- Report resolution, chains present, and every HETATM residue code with its atom
  count
- Identify which HETATM is the drug-like ligand (not water, not a buffer or
  cryoprotectant) and report its residue code
- Write a cleaned receptor to `receptors/`: strip waters, buffers,
  cryoprotectants and extra chains; keep the main protein chain only; save the
  ligand separately as SDF
- Compute the pocket center as the ligand's geometric center and the pocket box
  as the ligand bounding box expanded by 5 Å in each direction; save these to
  `receptors/<pdb_id>_pocket.json`

Then build `figures/viewer.html`: a single self-contained page using 3Dmol.js
loaded from cdnjs at a pinned version, which renders the cleaned receptor as a
cartoon, the ligand as sticks, and highlights residues within 5 Å of the ligand.
It must work by opening the file in a browser — do not depend on a Jupyter kernel
or a local server. Tell me how to open it so I can take a screenshot for the
proposal.

## Task 6 — Final summary

Write `results/SUMMARY.md` and also print to stdout, containing exactly these
four values, clearly labeled:

1. Number of molecules in Tox21 after standardization and deduplication
2. Mean ROC-AUC across the 12 endpoints under the random split
   (mean ± std over 5 seeds)
3. Mean ROC-AUC across the 12 endpoints under the scaffold split
   (mean ± std over 5 seeds)
4. The PDB ID we should use, with your recommendation between 1M17 and 4HJO and
   the reason

## Rules

- If any step fails, STOP and tell me what failed. Never substitute placeholder,
  estimated, or illustrative numbers for real results. These figures go into a
  graded academic submission.
- Pin every dependency version in `requirements.txt`.
- Set every random seed explicitly; the whole pipeline must be re-runnable to
  identical results.
- Keep standardization and featurization in importable modules, not inline in
  scripts — the training code and the web service will both import them later.
- Append every prompt I give you to `PROMPTS.md` with the date and what it was
  for. This is a graded course requirement.
- Do not commit raw or processed data; do commit split index files, cleaned
  receptors, metric tables, and the viewer.
- Report progress after each numbered task rather than at the very end.
````

---

## Prompt 2 — 2026-10-19

**Verbatim message:**

```text
继续
```

("continue")

**Purpose:** resume execution after the Python 3.11 environment had been built;
no change to the specification in Prompt 1.

---

## Prompt 3 — 2026-10-19

**Verbatim message:**

```text
把测好的数据整理好，放到一个md文档里。确保所有任务都完成了
```

("Organize the measured data into a single md document. Make sure all the tasks
are finished.")

The message also carried an aggregated automated code review of the work so far.

**Purpose:** consolidate every measured value into one document (`RESULTS.md`,
generated by `src/07_report.py`) and close out the review findings. Acted on:

1. `scaffold_groups_spanning_subsets` was only computed for the scaffold split,
   so the random split's `0` was meaningless. Now computed for both — the random
   split leaks 243–261 scaffold groups (~3800 molecules), the scaffold split 0 —
   and `03_splits.py` fails if either expectation breaks. Added per-seed
   test-set Jaccard overlap (`results/split_seed_overlap.csv`).
2. The receptor recommendation no longer rests on a 0.15 Å resolution
   difference. It now gates on unobserved residues inside the 5 Å pocket, then
   on conformational state, which is read from the deposited TITLE/JRNL records
   and confirmed from coordinates via the Lys721–Glu738 salt bridge
   (1M17: 3.05 Å, active αC-in; 4HJO: 13.30 Å, inactive αC-out). Pocket
   B-factors and alternate conformers are reported as an explicit caveat.
3. `requirements.txt` became a complete lock installed with `--no-deps`, because
   PyTDC 0.4.1 pulls `rdkit-pypi` into the same `rdkit/` directory as the modern
   `rdkit` wheel. The `dataclasses` 3.6 backport was dropped. Verified by
   building a throwaway venv from the lock and importing `tdc` + `rdkit`.
4. Reproducibility is now verified by hash, not by eyeball:
   `src/08_verify_repro.py` re-runs the pipeline and diffs SHA-256 of every
   artefact, recording the outcome in `results/reproducibility.json`.
5. `results/SUMMARY.md` and `RESULTS.md` now state that `±` is the sample
   standard deviation over the 5 seeds (ddof = 1, n = 5) of the per-seed macro
   average.

---

## Prompt 4 — 2026-10-20

**Verbatim message:**

```text
基于你的结果，我准备了进一步的测试任务，帮我完成：C:\Users\Admin\Documents\WeChat Files\wxid_icmox6yaq5k022\FileStorage\File\2026-10\claude-code-prompt-phase2-validation.md
```

("Based on your results, I have prepared a further set of test tasks; help me
complete them: <path>")

The message also carried a second aggregated automated code review of Phase 1.

**Purpose:** Phase 2 — validate the data foundation before committing to model
development: audit the Murcko scaffold variant, measure Tox21/Davis chemical-space
overlap, add KIBA plus a binding-affinity baseline, add hERG as an external
toxicity check, assemble a 15–20 structure EGFR co-crystal docking benchmark, and
write a decision summary (Tasks 1–6 of the referenced file).

**Referenced file, `claude-code-prompt-phase2-validation.md`, in full:**

````markdown
# Phase 2: Validate the data foundation before we build models

## Context

The Tox21 baseline pipeline is done (6,143 molecules, RF baselines at 0.832
random-split and 0.761 scaffold-split ROC-AUC, 1M17 selected as receptor, all
artefacts reproduce byte-identically). Before we commit six weeks to training
GNNs, a generator, and a docking pipeline, we need to know whether the data
foundation actually supports the project's central claim.

That claim is: training toxicity classification and binding-affinity regression
on one shared molecular encoder beats two single-task models. Three things could
quietly invalidate it, and all three are cheap to check now and expensive to
discover in December.

Work in the existing repository. Keep the same conventions: pinned versions,
explicit seeds, split index files committed, every prompt appended to PROMPTS.md
with date and purpose, byte-reproducible outputs. Report after each numbered
task rather than at the end.

**If any step fails or produces an ambiguous result, stop and tell me. Never
substitute placeholder or illustrative numbers — these feed a graded
submission.**

---

## Task 1 — Audit the scaffold computation (30 minutes, do this first)

Our scaffold split reported 2,156 unique scaffolds with the largest group at
1,260 molecules, all described as a benzene ring. RDKit computes Murcko
scaffolds two ways: one keeps atom identity and bond order, the other replaces
every atom with carbon and every bond with a single bond (the generic
framework). The generic version collapses chemically distinct cores into one
group, which would make our split harsher than it needs to be and inflate the
leakage numbers we quoted in the proposal.

Determine which variant `src/03_splits.py` used. Then recompute scaffold
statistics both ways on the same 6,143 molecules and report, for each variant:

- number of unique scaffolds
- size of the largest group, and the SMILES of its scaffold
- number of singleton scaffolds
- the ten largest groups with their scaffold SMILES and member counts

Write `results/scaffold_audit.md` with both tables side by side and a one-line
recommendation on which variant we should use. If the two differ materially,
re-run the random-forest baseline under the recommended variant so our reported
0.761 is correct, and flag clearly that the proposal number needs updating.

---

## Task 2 — Measure chemical-space overlap between the two tasks

This is the decisive check. Tox21 is mostly industrial chemicals, pesticides and
food additives: small, simple molecules. Kinase binding datasets are drug-like
inhibitors: larger, with fused aromatic heterocycles. If the two sets barely
overlap, a shared encoder is being asked to represent two unrelated chemical
spaces, and multi-task learning will likely show no gain for a reason that has
nothing to do with whether the tasks are related.

Download the Davis dataset through Therapeutics Data Commons and run it through
the same standardization pipeline as Tox21 (import the existing module — do not
reimplement it). Then compute:

1. **Nearest-neighbour similarity.** For every Tox21 molecule, find its maximum
   Tanimoto similarity to any Davis molecule using 2048-bit Morgan fingerprints,
   radius 2. Report the distribution: min, 25th percentile, median, 75th
   percentile, max, and the fraction above 0.4 and above 0.6. Do the same in the
   reverse direction (each Davis molecule against all of Tox21).
2. **Exact overlap.** How many InChIKeys appear in both datasets?
3. **Property distributions.** Compare molecular weight, cLogP, TPSA, number of
   rings, number of aromatic rings, number of rotatable bonds and fraction of
   sp3 carbons between the two sets. Report mean, standard deviation and the
   Kolmogorov-Smirnov statistic per property.
4. **A projection figure.** Fit PCA on the combined fingerprint matrix and plot
   both sets in the first two components, coloured by source. Also produce a
   UMAP version if the dependency installs cleanly; skip it rather than fighting
   the install.

Write `results/chemical_space_overlap.md` and save figures to
`figures/`. End the file with an explicit verdict in one of three forms:

- **Strong overlap** (median nearest-neighbour Tanimoto above 0.4): shared
  encoder is well supported, proceed as planned.
- **Weak overlap** (median between 0.25 and 0.4): shared encoder is defensible
  but we should expect modest gains and say so in the report.
- **Negligible overlap** (median below 0.25): the shared-encoder experiment will
  likely show no gain for distributional reasons, and we need to decide whether
  to add bridging data or reframe the research question.

Do not soften the verdict to match what we hoped for.

---

## Task 3 — Add KIBA and establish a binding-affinity baseline

Davis has only 68 distinct drugs. Our proposal promises a cold-drug evaluation,
which under an 80/10/10 split leaves roughly 7 drugs in the test set — far too
few for a stable correlation coefficient. KIBA covers the same task with 2,111
drugs and 229 proteins across roughly 118,000 measurements.

Download KIBA through Therapeutics Data Commons, standardize it with the same
pipeline, and report for both Davis and KIBA:

- number of distinct drugs, distinct proteins, and measured pairs
- matrix density (measured pairs divided by drugs times proteins)
- the affinity value distribution, and for Davis specifically, how many entries
  sit at the censored ceiling (Kd = 10000 nM, pKd = 5)

Build three split variants for each dataset, saved as index files in `splits/`:

- `random`: random over drug-protein pairs
- `cold_drug`: test drugs unseen in training
- `cold_target`: test proteins unseen in training

Report the resulting test-set sizes so we can see which settings are
statistically usable.

Then train a baseline affinity regressor on both datasets, under all three
splits, five seeds each:

- Drug features: 2048-bit Morgan fingerprints
- Protein features: amino-acid composition plus dipeptide composition (a cheap
  stand-in; ESM-2 comes later)
- Model: random forest regressor
- Metrics: RMSE, Pearson r, Spearman rho, and concordance index

Write `results/binding_baseline.csv` and `results/binding_baseline.md`.
Recommend which dataset should carry the main results and which should be the
secondary comparison, with the reason.

---

## Task 4 — Add hERG as an independent toxicity check

Tox21's twelve endpoints all come from one screening campaign, so a model doing
well across them may be fitting that campaign's idiosyncrasies rather than
general toxicity-relevant chemistry. hERG blockade is a separate, well-studied
cardiotoxicity endpoint with denser labels and better class balance, which makes
it a useful external sanity check.

Download the hERG dataset from Therapeutics Data Commons, standardize it, and
report dataset size, positive rate, and overlap with Tox21 by InChIKey. Train
the same random-forest baseline under both random and scaffold splits, five
seeds. Append the results to `results/baseline_summary.csv` with a dataset
column so Tox21 and hERG sit in one table.

Report whether the random-to-scaffold performance drop on hERG resembles the
0.071 we saw on Tox21. A similar drop suggests the leakage effect is general; a
very different one is worth investigating.

---

## Task 5 — Build a co-crystal reference set for docking evaluation

Our design calls for comparing AutoDock Vina against DiffDock on pose accuracy,
which requires experimentally determined poses to measure RMSD against. Two
structures is not enough to support any conclusion.

Query the RCSB API for EGFR kinase-domain structures that contain a drug-like
bound ligand. Filter to: resolution at or below 2.8 Å, a single ligand with
between 15 and 60 heavy atoms, no unobserved residue within 5 Å of that ligand,
and distinct ligand chemistry across the set. Select 15 to 20 structures.

For each, write to `receptors/benchmark/`:

- the cleaned receptor (waters, buffers, cryoprotectants and extra chains
  removed)
- the ligand as SDF with its crystallographic coordinates
- a JSON record with PDB ID, resolution, ligand code, ligand SMILES, pocket
  centre, pocket box, mean pocket B-factor, and whether the structure is in the
  active or inactive conformation (use the Lys745/Glu762 salt-bridge distance,
  or the equivalent numbering for that entry, with the same threshold we applied
  when choosing 1M17)

Write `results/docking_benchmark_set.md` with a summary table, and report how
many of the selected structures are active versus inactive conformation. Also
report pairwise ligand Tanimoto similarity within the set so we can show the
benchmark is chemically diverse rather than twenty variations of erlotinib.

Do not run any docking yet. This task only assembles and documents the
reference set.

---

## Task 6 — Decision summary

Write `results/PHASE2_DECISIONS.md` answering these five questions directly,
each with the number that supports it:

1. Which scaffold variant should we use, and does the proposal's 0.761 need
   correcting?
2. Does the chemical-space overlap support the shared-encoder experiment, and if
   not, what are our options?
3. Which binding dataset carries the main results, and which split settings have
   usable test-set sizes?
4. Does hERG corroborate the Tox21 leakage finding?
5. Is the docking benchmark set large and diverse enough to support a Vina
   versus DiffDock comparison?

Keep it short. One paragraph per question. This file is what we will read before
starting model development, and parts of it will go into the final report's
discussion section.
````

**Phase 1 review findings also acted on in this round:**

1. `receptors/*_ligand.sdf` had every bond written as order 1 (RDKit's
   `proximityBonding` cannot recover bond orders from a PDB), which destroyed the
   quinazoline aromatics and the terminal alkyne. `structure.write_ligand_sdf`
   now assigns bond orders from the RCSB Chemical Component Dictionary entry via
   `AssignBondOrdersFromTemplate` and fails loudly if the template does not map.
   Both files now carry 17 aromatic bonds, 1 triple bond and InChIKey
   `AAKJLRGGTJKAMG-UHFFFAOYSA-N` (erlotinib).
2. The receptor recommendation no longer claims the depositions state both
   conformations (only 4HJO's title does) and no longer calls 1M17 "more
   complete" (it models a longer construct but has more internal gap residues,
   none in the pocket). Conformational state is now stated as the decisive
   criterion, with resolution a minor tiebreak.
3. `pretty()` in `07_report.py` no longer rewrites a bare " A " (which corrupted
   "IN COMPLEX WITH A 4-ANILINOQUINAZOLINE"); only an "A" following a digit
   becomes "Å".
4. `RESULTS.md` provenance now uses `git describe --always --dirty`, so a
   document generated from a dirty tree says so.
5. `08_verify_repro.py` now requires an expected artefact count per group, so a
   group that matches nothing fails instead of passing vacuously, and keys are
   normalised to POSIX form.
6. The delta column is labelled "delta (scaffold - random)" in both generators.
