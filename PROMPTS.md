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
