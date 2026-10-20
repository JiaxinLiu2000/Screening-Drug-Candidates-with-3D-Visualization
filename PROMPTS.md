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

---

## Prompt 5 — 2026-10-20

**Verbatim message:**

```text
继续新的尝试 "C:\Users\Admin\Documents\WeChat Files\wxid_icmox6yaq5k022\FileStorage\File\2026-10\claude-code-prompt-phase3.md"
```

("Continue with the new attempt: <path>")

The message also carried a third aggregated automated code review.

**Purpose:** Phase 3 — fix what Phase 2 exposed, then build the shared-encoder
experiment: re-measure chemical-space overlap against KIBA with a within-set
calibration reference, rebuild the docking benchmark around self-docking, replace
hERG with the large variant, bootstrap per-endpoint confidence intervals, build
the GIN/ChemBERTa encoder and run the four-way ablation, add atom-level
attribution, and write the decision summary (Tasks 1–7 of the referenced file).

**Phase 2 review findings also acted on in this round:**

1. `HetGroup.n_atoms` counted hydrogens and every alternate conformer, so
   `pick_ligand`'s size threshold and "largest group" ranking used the wrong
   number (8SC7's D0D: 41 records, 31 heavy atoms). `HetGroup` now carries
   `n_heavy_atoms` (primary conformer, no H) and selection uses it. The docking
   benchmark's 15–60 filter was unaffected — it already de-duplicated altlocs by
   atom name — and re-running `13_docking_benchmark.py` reproduced the same 20
   structures.
2. `contact_residues` is fed the same filtered coordinate array as the pocket box,
   because `build_pocket` passes the output of `ligand_coords` to both; the altloc
   restriction therefore applies to the 5 Å residue list as well.
3. `PRIMARY_ALTLOCS` moved to the top of `structure.py` as the single definition
   of "the ligand copy we keep", used by heavy-atom counts, pocket centre, box,
   contact residues and the SDF alike.

**Referenced file, `claude-code-prompt-phase3.md`, in full:**

````markdown
# Phase 3: Fix what Phase 2 exposed, then build the shared-encoder experiment

## Context

Phase 2 validated the data foundation and turned up four things that change the
plan. Read `results/PHASE2_DECISIONS.md` before starting; this document assumes
its contents.

What Phase 2 settled:

- The scaffold split already uses the chemically-aware Murcko variant (2156 vs
  1285 generic scaffolds), so the 0.761 scaffold-split ROC-AUC stands. No
  correction needed.
- KIBA (2030 drugs, 117193 pairs) is the right primary binding dataset. Davis
  (68 drugs, 71% censored) becomes a small-data comparison only.
- 36/36 artefacts reproduce byte-identically and 38 integrity checks pass. Keep
  that standard; every task below must stay reproducible.

What Phase 2 broke:

1. **Tox21 and Davis barely share chemical space** (median nearest-neighbour
   Tanimoto 0.1667). The central claim of this project is that a shared
   molecular encoder helps both tasks. If the two tasks occupy disjoint
   chemical space, a from-scratch shared encoder will probably show no gain, and
   for a reason that says nothing about whether the tasks are related.
2. **The docking benchmark is 17/20 inactive conformation** while the chosen
   receptor 1M17 is active. Cross-docking every ligand into 1M17 would fail for
   conformational reasons unrelated to engine quality.
3. **The hERG dataset is the small variant** (588 molecules, 68.7% positive,
   135 InChIKeys shared with Tox21). Too small and too enriched to work as an
   external check.
4. **Several endpoints have very few test positives** (NR-PPAR-gamma 14,
   NR-AR-LBD 17, SR-ATAD5 21), which is why their per-endpoint PR-AUC standard
   deviations reach ±0.11.

Tasks 1 to 4 fix these. Tasks 5 to 7 build on the fixes. Work through them in
order; several later tasks depend on the measurement in Task 1.

**Standing rules.** Pin versions. Set every seed explicitly. Commit split index
files, not just seeds. Append every prompt to `PROMPTS.md` with date and
purpose. Import the existing standardization module rather than reimplementing
it. Extend `src/15_data_audit.py` with integrity checks for every new artefact.
Report after each numbered task.

**If a step fails or the result is ambiguous, stop and tell me. Never
substitute placeholder, estimated or illustrative numbers. Do not soften a
measurement to match what we hoped for — a negative result reported plainly is
worth more to us than a flattering one.**

---

## Task 1 — Re-measure chemical-space overlap against KIBA

The 0.1667 figure was computed against Davis, which has only 68 distinct drugs.
KIBA has 2030. A set thirty times larger and chemically broader may overlap
Tox21 considerably more, and the whole shared-encoder design hinges on this
number. Measure it before changing anything else.

Run the same procedure as `src/*_chemical_space_overlap.py` used for Davis, now
for Tox21 against KIBA:

- Nearest-neighbour maximum Tanimoto for every Tox21 molecule against all of
  KIBA, 2048-bit Morgan radius 2. Report min, quartiles, median, max, and the
  fraction above 0.4 and above 0.6. Then the reverse direction.
- Exact InChIKey overlap between the two sets.
- Property distributions (molecular weight, cLogP, TPSA, ring count, aromatic
  ring count, rotatable bonds, fraction sp3) with means, standard deviations and
  a Kolmogorov-Smirnov statistic per property.
- A PCA projection of the combined fingerprint matrix coloured by source, with
  Davis shown as a third colour so all three sets appear on one figure.

Also report, as a calibration reference, the median nearest-neighbour Tanimoto
**within** Tox21 and **within** KIBA. Without that, we cannot tell whether 0.1667
is unusually low or simply what Morgan fingerprints give for any two large
diverse sets.

Write `results/chemical_space_overlap_kiba.md`. End with the same three-way
verdict as before (strong above 0.4, weak 0.25 to 0.4, negligible below 0.25),
and state explicitly whether the Davis-based conclusion still holds.

---

## Task 2 — Rebuild the docking benchmark around self-docking

A pose-prediction benchmark compares a predicted pose against the
crystallographic pose of the same ligand in the same receptor. Docking all 20
ligands into 1M17 instead measures how well each ligand tolerates a foreign
receptor conformation, which is a different question.

Restructure `receptors/benchmark/` so each of the 20 entries carries its own
cognate receptor, its own pocket definition and its own crystallographic ligand
pose, ready for self-docking. For each entry record:

- PDB ID, resolution, ligand code, ligand SMILES, ligand heavy-atom count
- conformational state (active or inactive) by the Lys/Glu salt-bridge
  criterion, with the measured distance and the residue numbering used
- pocket centre and box from the cognate ligand
- mean pocket B-factor and count of alternate-conformer atoms in the site

Then check whether 3 active structures is enough to say anything about
conformational stratification. Query RCSB again specifically for **active-state**
EGFR kinase-domain co-crystals meeting the same quality filters, and report how
many exist. If more are available, extend the set toward a more even active /
inactive balance and report the final counts. If the active-state population is
genuinely small, say so and record it as a limitation rather than padding the
set with low-quality entries.

Write `results/docking_benchmark_v2.md` with the per-entry table, the final
active/inactive counts, pairwise ligand Tanimoto statistics, and a short
protocol note distinguishing two uses of docking in this project: self-docking
into cognate receptors for engine evaluation, and docking novel molecules into
1M17 for screening. Still no docking runs in this task.

---

## Task 3 — Replace hERG with the large variant

Download the larger hERG dataset available through Therapeutics Data Commons
(the Karim et al. version, roughly 13,000 molecules, not the ~650-molecule
one currently in use). Standardize it with the existing pipeline and report
size, positive rate, and InChIKey overlap with Tox21.

Build two evaluation settings and keep them separate:

- **Full**: standard scaffold split over the whole hERG set
- **Disjoint**: every molecule whose InChIKey also appears in Tox21 removed
  before splitting, so the result is a genuine external check

Train the random-forest baseline under both settings, random and scaffold
splits, five seeds each. Append to `results/baseline_summary.csv` with a dataset
and setting column.

Report whether the random-to-scaffold ROC-AUC drop on hERG resembles the 0.0713
seen on Tox21. Keep the old small-hERG numbers in the table, clearly labelled,
so the two versions can be compared.

---

## Task 4 — Report test-set positive counts and flag unreliable endpoints

Add a `n_test_positives` column to `results/baseline_metrics.csv` for every fit
(it already exists for one seed; make it universal), and to the per-endpoint
summary table.

Then quantify the instability directly: for each endpoint, bootstrap the test
set 1000 times within each seed and report the 95% confidence interval on
ROC-AUC and PR-AUC. Mark any endpoint whose test set holds fewer than 20
positives.

Write `results/endpoint_reliability.md` with the table and a one-line rule we
will follow in the report, of the form: endpoints below the threshold are
included in the macro average but not interpreted individually. Confirm whether
the macro-average confidence interval is tight enough that the 0.0713
scaffold-split penalty survives as a claim.

---

## Task 5 — Build the molecular encoder and run the four-way ablation

This is the project's central experiment, restructured in light of Task 1.

Implement a GIN encoder over molecular graphs (atoms as nodes, bonds as edges;
standard atom and bond features) with a configurable output dimension, plus two
heads: a 12-way toxicity head and an affinity head that concatenates the
molecular embedding with a frozen ESM-2 protein embedding. Precompute and cache
the ESM-2 embeddings for all KIBA proteins; they do not change during training.

Train and evaluate four configurations on identical splits and five seeds:

| config | encoder | training |
| --- | --- | --- |
| A | GIN from scratch | toxicity only |
| B | GIN from scratch | affinity only |
| C | GIN from scratch | both tasks, shared encoder |
| D | pretrained chemical language model, fine-tuned | both tasks, shared encoder |

For D, use a pretrained molecular encoder (ChemBERTa, MolFormer, or equivalent)
whose pretraining corpus spans both industrial chemicals and drug-like
molecules. Full fine-tuning, not LoRA, unless the checkpoint exceeds what the
GPU allows. The point of D is that its representation already covers both
chemical spaces, so transfer is possible even if Task 1 confirms the two task
datasets are disjoint.

For C and D, sweep the loss weight ratio over {0.1, 0.5, 1, 2, 5} and report the
full grid, not only the best cell.

Metrics: toxicity by ROC-AUC and PR-AUC under scaffold split; affinity by RMSE,
Pearson, Spearman and concordance index under random, cold-drug and cold-target
splits. The random forests remain the floor for both tasks.

Write `results/shared_encoder_ablation.md` with the four-way comparison, the
loss-weight grid, and an explicit statement of which of these the evidence
supports:

- shared encoding helps regardless of pretraining
- shared encoding helps only when the encoder is pretrained
- shared encoding does not help, and the chemical-space measurement from Task 1
  explains why

Any of the three is a publishable finding for this report. Report whichever the
numbers show.

---

## Task 6 — Atom-level attribution for the toxicity model

Implement atom masking attribution: for each heavy atom, zero its features (or
replace them with an unknown-atom token), re-run inference, and record the drop
in predicted probability. Produce one attribution vector per molecule per
endpoint, ordered to match the canonical SMILES atom ordering.

Validate qualitatively rather than numerically, since there is no ground truth.
Select molecules from the test set containing well-known structural alerts
(nitroaromatics, aromatic amines, quinones, Michael acceptors), and check
whether the attribution concentrates on those substructures. Produce a figure
with four to six annotated examples, the alerting substructure circled, for the
report.

Also report the failure cases: molecules where attribution is diffuse or lands
on chemically implausible atoms. Showing where the method breaks is more
convincing than showing only where it works.

Write `results/attribution.md` and save figures to `figures/attribution/`.

---

## Task 7 — Decision summary

Write `results/PHASE3_DECISIONS.md`, one short paragraph per question, each
backed by a number:

1. Does the KIBA overlap measurement change the shared-encoder design, and does
   the within-set calibration reference change how we interpret 0.1667?
2. Is the rebuilt docking benchmark adequate for an engine comparison, and what
   is the final active / inactive balance?
3. Does the large hERG set corroborate the Tox21 scaffold-split penalty, and
   does the disjoint setting change the picture?
4. Which endpoints will we decline to interpret individually, and does the
   0.0713 penalty survive bootstrapping?
5. Which of the three shared-encoder outcomes did we observe, and what does the
   loss-weight grid show?
6. Does atom attribution recover known structural alerts often enough to put in
   the demo, and where does it fail?

This file is what we will write the report's discussion section from.
````
