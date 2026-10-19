# Multi-objective virtual screening — preliminary results (CS 6140)

Toxicity (Tox21, 12 endpoints) + binding-site preparation (EGFR kinase domain)
for a multi-objective virtual screening tool with 3D pose visualization.
This repository contains the **preliminary results** for the project proposal:
a reproducible data pipeline, a random-forest baseline under two split
protocols, and a self-contained 3D pocket viewer.

**All measured numbers live in one place: [`RESULTS.md`](RESULTS.md)** (generated
by `src/07_report.py` from the committed artefacts, so it cannot drift from the
CSV/JSON tables). The four values the proposal asks for are also in
[`results/SUMMARY.md`](results/SUMMARY.md).

| value | result |
| --- | --- |
| Molecules after standardization + deduplication | **6143** |
| Mean ROC-AUC over 12 endpoints, random split (5 seeds) | **0.8324 ± 0.0181** |
| Mean ROC-AUC over 12 endpoints, scaffold split (5 seeds) | **0.7611 ± 0.0227** |
| Recommended receptor | **1M17** — active αC-in state (Lys721–Glu738 = 3.05 Å), 2.60 Å, 312 residues, erlotinib/AQ4 |

This README covers setup and method; it quotes only headline figures and defers
the full tables to `RESULTS.md`.

## Setup

```bash
python -m venv .venv                                            # Python 3.11 (3.11.17 here)
.venv/Scripts/python -m pip install --no-deps -r requirements.txt   # Windows
# source .venv/bin/activate && pip install --no-deps -r requirements.txt   # Linux/macOS
```

**`--no-deps` is required, not optional.** `requirements.txt` is a complete lock
(every direct and transitive dependency pinned). PyTDC 0.4.1 declares a
dependency on `rdkit-pypi`, which unpacks into the same `rdkit/` package
directory as the modern `rdkit` wheel; letting the resolver run installs both and
whichever lands last silently overwrites the other, so `rdkit.__version__`
becomes install-order dependent. Resolving nothing and pinning everything avoids
that. Two related pins: `setuptools==80.10.2` (PyTDC 0.4.1 imports
`pkg_resources`, removed in setuptools 81) and no `dataclasses` backport (a
Python 3.6-only package that PyTDC pulls in and that shadows the stdlib on 3.11).

Verified by building a throwaway venv from this lock and importing `tdc`,
`rdkit`, `sklearn`, `gemmi` — one RDKit, stdlib `dataclasses`, 12 endpoints
resolved.

## Run the pipeline

Scripts are numbered and must run in order; each is a thin driver over the
importable modules in `src/tox21lib/`.

```bash
python src/01_download_tox21.py    # TDC download + merge to one wide table
python src/02_standardize.py       # RDKit standardize, dedup, filter
python src/03_splits.py            # random + scaffold split index files
python src/04_baseline_rf.py       # 120 RF fits (~4 min on 8 cores)
python src/05_egfr_structure.py    # 1M17/4HJO cleaning, pockets, viewer.html
python src/06_summary.py           # results/SUMMARY.md + stdout
python src/07_report.py            # RESULTS.md - every measured number
python src/08_verify_repro.py --with-rf   # re-run + SHA-256 diff of all artefacts
```

## Layout

```
data/raw/            TDC CSVs and downloaded PDBs            (gitignored)
data/processed/      tox21_wide.parquet, tox21_clean.parquet (gitignored)
splits/              random_split_seed{0..4}.csv,
                     scaffold_split_seed{0..4}.csv, scaffolds.csv   (committed)
receptors/           <ID>_receptor.pdb, <ID>_ligand.sdf,
                     <ID>_pocket.json                               (committed)
results/             metric tables, stats JSON, SUMMARY.md          (committed)
figures/             viewer.html + screenshots                      (committed)
src/tox21lib/        importable modules: paths, standardize,
                     featurize, splits, structure
src/0*.py            numbered drivers
```

The training code and the planned web service both import
`tox21lib.standardize` and `tox21lib.featurize`, so featurization can never
drift between training and serving.

## Pipeline details

### 1. Data (`01_download_tox21.py`)

Endpoint names come from `tdc.utils.retrieve_label_name_list("Tox21")` (12
names, not hardcoded). Each endpoint is downloaded as a separate
`tdc.single_pred.Tox` dataset and outer-joined on `(Drug_ID, SMILES)` into one
wide table: **(7831, 14)** = 7831 molecules × (`drug_id`, `smiles`, 12 label
columns), NaN where a compound was not tested. Per-endpoint positive rates are
in `results/tox21_endpoint_stats.csv` (2.9 %–16.2 %).

### 2. Standardization (`02_standardize.py` → `tox21lib.standardize`)

Per SMILES: parse → keep the largest covalently connected fragment (salt strip)
→ `Chem.RemoveStereochemistry` → canonical SMILES → InChIKey.

**v1 simplification — stereochemistry is discarded.** Both atom and bond stereo
are removed before the InChIKey is computed, so enantiomers and cis/trans pairs
collapse onto a single record. Morgan fingerprints at radius 2 are only weakly
stereo-aware anyway, but this does mean the model cannot express
stereo-dependent toxicity (a real effect for e.g. some nuclear-receptor
binders). Restoring stereo is the first planned change after the proposal.

Deduplication is on InChIKey with a majority vote per label column; ties become
NaN rather than an arbitrary pick. Then molecules are filtered to 10–60 heavy
atoms and to the elements {C, N, O, S, P, F, Cl, Br, I, H}.

Measured counts (`results/standardization_stats.json`):

| step | count |
| --- | --- |
| merged input rows | 7831 |
| dropped, failed to parse | 8 |
| unique InChIKeys | 7580 |
| duplicate rows merged | 243 |
| label cells set to NaN by a tied vote | 67 |
| removed by heavy-atom/element filter | 1437 (1263 <10 heavy, 72 >60 heavy, 102 disallowed element) |
| **final molecules** | **6143** |

### 3. Splits (`03_splits.py` → `tox21lib.splits`)

Both splits are computed **once, on the deduplicated molecule set, before any
featurization**, and written as index files keyed by row index + InChIKey. Later
stages replay them and assert the InChIKey order matches
`tox21_clean.parquet`, so results survive an RDKit upgrade.

- `random_split`: 80/10/10 via `numpy.random.default_rng(seed)` → 4914/614/615.
- `scaffold_split`: Bemis-Murcko scaffold SMILES per molecule, grouped by
  identical scaffold, groups sorted by size descending, greedily filling train
  to 80 %, then valid to 10 %, remainder to test → 4915/615/613. Whole groups
  are never split. The seed permutes only the order of *equal-sized* groups (the
  size sort is stable), so the required size ordering is preserved while the 5
  seeds still probe genuinely different assignments — 1065 molecules change
  subset between seed 0 and seed 1, and scaffold-split test sets overlap by only
  Jaccard 0.21–0.27 across seed pairs (`results/split_seed_overlap.csv`).

`03_splits.py` computes `scaffold_groups_spanning_subsets` for **both** splits
and fails the run unless the scaffold split is exactly 0 *and* the random split
is nonzero — a zero there would mean the check itself is broken. Measured: the
random split spreads **243–261 scaffold groups (≈3750–3852 molecules)** across
subsets; the scaffold split, 0.

Scaffold statistics: **2156 unique scaffolds**, largest group **1260 molecules**
(benzene, `c1ccccc1`), 1658 singleton scaffolds, 862 acyclic molecules (empty
scaffold, treated as one group).

### 4. Baseline (`04_baseline_rf.py` → `tox21lib.featurize`)

Morgan fingerprints, radius 2, 2048 bits, via
`rdFingerprintGenerator.GetMorganGenerator` (the deprecated
`GetMorganFingerprintAsBitVect` is not used). Mean 32.7 bits set per molecule.

Per endpoint, per split, per seed: `RandomForestClassifier(n_estimators=500,
class_weight="balanced", n_jobs=-1, random_state=seed)` trained on training
rows with a non-null label, evaluated on test rows with a non-null label →
ROC-AUC and PR-AUC (`average_precision_score`). 2 splits × 5 seeds × 12
endpoints = **120 fits**.

- `results/baseline_metrics.csv` — one row per (split, seed, endpoint) with
  fold sizes, positives, `roc_auc`, `pr_auc`.
- `results/baseline_summary.csv` — mean/std across seeds, both per endpoint
  (`level=endpoint`) and macro-averaged over the 12 endpoints
  (`level=overall`). The overall figure averages the 12 endpoints *within* a
  seed first, then takes mean/std over seeds, so the std is seed-to-seed
  variation, not endpoint spread.

Result: random **0.8324 ± 0.0181**, scaffold **0.7611 ± 0.0227** ROC-AUC
(PR-AUC 0.4926 ± 0.0460 and 0.3806 ± 0.0350). The 0.071 ROC-AUC gap is the
motivation for reporting scaffold-split numbers in the proposal: random splits
overstate generalization to new chemotypes.

### 5. EGFR structures (`05_egfr_structure.py` → `tox21lib.structure`)

| PDB | resolution | chains | HETATM codes (atoms) | ligand | kept chain | residues ≤5 Å |
| --- | --- | --- | --- | --- | --- | --- |
| 1M17 | 2.60 Å | A | AQ4 (29), HOH (20 waters) | **AQ4** = erlotinib | A, 312 res | 20 |
| 4HJO | 2.75 Å | A | HOH (34 waters), AQ4 (29) | **AQ4** = erlotinib | A, 278 res | 20 |

The ligand is chosen as the largest HETATM group that is not in
`structure.NON_LIGAND_HET` (waters, buffers, cryoprotectants, ions, unknowns)
and has ≥10 atoms — not by hardcoding `AQ4`.

Cleaned outputs per structure: `<ID>_receptor.pdb` (main protein chain only,
waters/buffers/ligand/other chains stripped), `<ID>_ligand.sdf` (experimental
coordinates), `<ID>_pocket.json` (ligand centroid as pocket center, ligand
bounding box + 5 Å per side as the docking box, plus the residues within 5 Å
with their minimum distances).

Pocket centers: 1M17 `[22.014, 0.253, 52.794]`, 4HJO `[24.770, 9.194, -0.003]`.

**Recommendation: 1M17**, decided on pocket-local evidence rather than the 0.15 Å
resolution difference, which is too small to settle it:

1. *Hard gate — pocket completeness.* Neither structure has a REMARK 465
   unobserved residue inside the 5 Å pocket, so both pass; internal chain gaps
   sit outside the site (1M17: 965–976; 4HJO: 710–713).
2. *Conformational state.* The depositions say it outright — 1M17 is
   "EGFR tyrosine kinase domain with 4-anilinoquinazoline inhibitor erlotinib"
   (Stamos et al., *J. Biol. Chem.* 277:46265, 2002) and 4HJO is "crystal
   structure of the **inactive** EGFR tyrosine kinase domain with erlotinib"
   (Park et al., *Biochem. J.* 448:417, 2012, "Erlotinib binds both inactive and
   active conformations of the EGFR tyrosine kinase domain"). The coordinates
   confirm it: the β3-Lys721–αC-Glu738 salt bridge is intact at **3.05 Å** in
   1M17 (active, αC-in) and broken at **13.30 Å** in 4HJO (inactive, αC-out).
   We screen ATP-competitive chemotypes against the catalytically relevant
   active site, so the αC-in state is the right target.
3. *Then* resolution (2.60 vs 2.75 Å) and completeness (312 vs 278 modelled
   residues), both favouring 1M17.

**Caveat to state in the proposal:** 4HJO's pocket is the better-ordered one —
mean B-factor 42.9 vs 51.5 Å² over pocket atoms, ligand B 47.5 vs 72.8 Å², and 0
vs 16 alternate-conformer atoms in the site. Since erlotinib demonstrably binds
both states, 4HJO is kept as the inactive-state counter-structure for a
cross-docking sanity check rather than discarded.

### Viewer — `figures/viewer.html`

Single self-contained page: 3Dmol.js pinned at **2.0.4** from cdnjs; receptor
PDB, ligand SDF and pocket JSON are **inlined into the HTML**, so it needs no
local server, no Jupyter kernel and no file-access flags.

**How to open it:**

```bash
start figures\viewer.html      # Windows cmd / PowerShell
# or just double-click figures/viewer.html in Explorer
```

Controls: switch between 1M17 and 4HJO, toggle whole-domain vs pocket zoom,
residue labels, the 5 Å-padded docking box, and a pocket surface. The receptor
renders as a spectrum cartoon, erlotinib as cyan sticks, and the 20 residues
within 5 Å as yellow sticks. Rendering verified in headless Chromium (canvas
present, no page errors). Reference captures: `figures/egfr_1M17_viewer.png`,
`figures/egfr_4HJO_viewer.png` (whole domain) and
`figures/egfr_1M17_pocket.png` (pocket zoom with residue labels — use this one
for the proposal figure).

## Reproducibility

- Every seed is explicit: `tox21lib.paths.SEEDS = (0, 1, 2, 3, 4)` drives both
  the split index files and `random_state` of each forest.
- Standardization, deduplication (`groupby(sort=True)`, InChIKey-sorted output)
  and fragment selection (ties broken on canonical SMILES) are
  order-independent, so `tox21_clean.parquet` is byte-stable across runs.
- Split index files are committed; the baseline replays them and asserts the
  InChIKey order still matches `tox21_clean.parquet` rather than recomputing
  scaffolds.
- Verified by hash, not by eye: `src/08_verify_repro.py --with-rf` snapshots the
  SHA-256 of every artefact (`splits/*`, `receptors/*`, `results/*`,
  `tox21_clean.parquet`), re-runs scripts 02–05 including all 120 forest fits,
  and diffs. The outcome, with the interpreter path and package versions, is
  recorded in `results/reproducibility.json` and surfaced in `RESULTS.md` §7.

## Known limitations (to state in the proposal)

1. Stereochemistry is dropped (see above).
2. Tox21 labels are sparse — each endpoint covers 4385–5658 of the 6143
   molecules; models are trained per endpoint on non-null rows only, with no
   multi-task sharing yet.
3. Classes are heavily imbalanced (3–18 % positives); `class_weight="balanced"`
   is the only mitigation in the baseline, and PR-AUC is the more honest
   metric.
4. Binding affinity is not yet modelled — Task 5 only prepares the EGFR
   receptor, pocket box and viewer that the docking/affinity stage will use.
