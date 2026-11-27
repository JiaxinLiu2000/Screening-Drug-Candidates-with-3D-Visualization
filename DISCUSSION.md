# Discussion

Draft of the final report's Discussion section. Everything in `FINAL_REPORT.md`
is generated from a committed artefact; this document is the interpretive layer
and is written by hand, so it is kept separate rather than mixed into generated
output.

---

## 1. Comparison with existing solutions

### 1.1 Where we sit on the binding task

| method | KIBA-random CI | MSE | source |
| --- | --- | --- | --- |
| KronRLS | 0.782 | 0.411 | DeepDTA, *Bioinformatics* 34(17):i821, Table 4 |
| SimBoost (gradient-boosted trees) | 0.836 | 0.222 | same |
| DeepDTA (CNN + CNN) | 0.863 | 0.194 | same |
| GraphDTA (GIN) | 0.891 | 0.139 | GraphDTA, bioRxiv 684662v2 |
| **ours — tuned random forest** | **0.8420** | **0.2297** | `results/binding_baseline_tuned.csv` |

Two honest readings of the same row, which is why both metrics are given. On the
concordance index our tuned forest passes SimBoost (0.8420 against 0.836); on
mean squared error it does not (0.2297 against 0.222). Quoting only the first
would be cherry-picking. Against the neural methods we are behind on both.

The comparison is also not like-for-like: published numbers come from the
authors' own splits and preprocessing, and our standardisation drops
stereochemistry and deduplicates differently. `results/PHASE4_PROPOSAL.md`
records the differences we could identify. We therefore read this table as "the
same order of magnitude as the pre-neural literature", not as a ranking.

### 1.2 Where we sit on the toxicity task

The relevant comparison here turned out not to be a number but a dataset
problem. The Tox21 labels redistributed through MoleculeNet and TDC — the route
we took — were altered during integration, with labels imputed or manufactured
(Ebner et al., arXiv 2511.14744, which hosts a reproducible leaderboard on the
*original* challenge data). Our 0.8324 random-split macro ROC-AUC therefore
cannot be placed directly against published Tox21 numbers. We report it as an
internal reference point and treat the *penalty* between splits, not the
absolute level, as the result.

That same leaderboard reports that DeepTox (the 2015 challenge winner) and
self-normalising networks from 2017 remain among the top methods. That is
consistent with what we measured independently: see §2.

### 1.3 Where our negative results were independently reproduced

Three findings we had treated as possible mistakes of ours have since been
reproduced by others:

- **Scaffold-split penalty.** We measured 0.0713 macro ROC-AUC, 95% interval
  [0.0413, 0.1013]. ToxBench (*BMC Bioinformatics*, 2026) reports 0.057–0.079
  across four model classes on Tox21, mean 0.070 — inside our interval.
- **Pretrained molecular embeddings do not beat fingerprints.** A 25-model,
  25-dataset benchmark under scaffold splits (arXiv 2508.06199v4) finds almost
  no neural embedding significantly better than count-ECFP, with GIN ranking
  20th of 23 (mean AUROC 75.14% against ECFP's 79.89%). Our GIN matching, never
  beating, the random forest is the same observation at smaller scale.
- **Architecture ranking depends on the shift.** "Beyond Random Splits" (arXiv
  2610.03456, 718,800 pairs) finds rank correlation between chemical and protein
  shifts falling to τ = 0.39, and *reversing* to −0.55 under dual OOD. Our
  modality ablation is an instance: on the cold-target split the drug-only model
  beats the two-modality model by +0.0927 Pearson.

This matters for how the project should be read. The headline is not "our models
were weak"; it is that the weakness is a property of the problem setup that
several independent groups measured at the same time.

---

## 2. What worked

**Borrowing a pretrained protein model.** The strongest affinity result —
Pearson 0.8510 ± 0.0043 over five seeds — came from frozen ESM-2 embeddings fed
to a small MLP. Nothing we trained from scratch came close. The useful lesson is
about allocation: the win came from choosing what *not* to train.

**Treating the baseline as a real competitor.** Retuning the random forest
(`max_features=0.3`) cut the neural network's apparent lead from +0.0669 to
+0.0291, and on a 5%-data subset to +0.0058. Had we not retuned, we would have
reported a three-fold larger effect and believed it.

**Replaying splits instead of recomputing them.** Split index files are
committed, so every model in every phase is evaluated on literally the same
molecules. This is why the Phase 3 ablation and the Phase 1 forest can be
compared at all, and why `src/08_verify_repro.py` can check artefacts rebuild
byte-identically.

**Checking the serving path against the training metric.** Three defects were
found this way and none would have been caught by "does the model load": a
mis-parsed embedding cache, a duplicated model class whose state-dict keys had
drifted, and — the serious one — a predictor feeding raw ESM-2 vectors to a model
trained on standardised ones, scoring 0.4832 instead of 0.8487 while producing
entirely plausible-looking output.

---

## 3. What was hard

**Negative results are harder to report than to obtain.** The temptation at
every step was to find a configuration where the neural model won. The
discipline that prevented it was mechanical: regenerate every reported number
from a committed artefact, and let the script write the report.

**A benchmark that cannot support its own claim.** Our docking set was 20
co-crystals of a single protein (EGFR, P00533) selected from 271 RCSB hits.
`results/PHASE2_DECISIONS.md` recorded at the time that no claim about docking
methods could rest on it, and the instructor independently made the same point.
Fixing it meant adopting PoseBusters (428 complexes) and Astex Diverse (85),
which exposed a further problem: those receptors retain cofactors and modified
residues that Meeko cannot parameterise, so roughly a third of complexes failed
preparation. Our own EGFR receptors had been pre-cleaned, which is why this was
invisible until the benchmark was widened.

**A "scientific" result that was an engineering artefact.** Configuration C (GIN
shared between both tasks) scored 0.6820 macro ROC-AUC against 0.7689 for the
single-task control — a 0.0869 penalty we initially read as task interference.
Reading the training loop showed the two task losses were never summed: the
toxicity loader ran, then the affinity loader, each batch taking its own
optimiser step, with affinity last and contributing ~18× more batches. The
shared encoder therefore ended every epoch updated almost entirely by affinity,
and was then validated on toxicity. The same sharing costs ChemBERTa nothing
(0.7603 → 0.7619), which the mechanism predicts: it is pretrained and runs at a
50× lower encoder learning rate, so it barely moves. Gradient conflict, the
explanation we and the instructor both assumed, was not applicable — the two
gradients never coexisted.

**Infrastructure failures that destroyed work.** A full disk killed one
multi-hour experiment outright and truncated another at 253 of 513 complexes.
The docking script now resumes from existing poses, with a settings stamp so
coordinates produced under different search parameters are discarded rather than
pooled.

---

## 4. What we learned

**Calibration and abstention are separate problems, and confidence solves
neither.** Isotonic regression cut expected calibration error from 0.0653 to
0.0252, so the probabilities became trustworthy *in aggregate*. It did nothing
for knowing which individual prediction is wrong. Abstaining on the molecules
whose predicted probability sat closest to 0.5 scored PR-AUC 0.2079 at 50%
coverage — worse than abstaining at random (0.3983). Distance to the training
set scored 0.4904. A well-calibrated model can still be confidently wrong on the
samples that matter.

**Build the evaluation before the model.** Phase 2 existed only to validate the
data foundation, and it is the reason the later results are interpretable. The
scaffold audit, the chemical-space overlap measurement, and the split-index
commitment all predate any model worth reporting.

**Distribution shift is not one axis.** Random → scaffold → cold-drug →
cold-target is not a single difficulty ladder. On cold-target the protein branch
actively hurts, which no amount of "more training data" addresses.

---

## 5. How the application reflects the findings

The interface was designed around §4's first point rather than around the
predictions. Applicability is a column beside the prediction, not a tooltip;
out-of-domain rows are struck through and sorted last under every sort order;
unparseable molecules keep their row with an error rather than vanishing, so a
user cannot read results against the wrong input. The affinity axis is anchored
to the measured KIBA distribution instead of auto-scaling to the batch, because
75% of real values lie within ±0.6 of the median and a batch stretched across
the full plot width reads as a separation that is not there.

---

## 6. What we promised and did not build

Stated plainly, because the proposal committed to each of these.

| promised | status | why |
| --- | --- | --- |
| Generative SMILES model for candidate molecules | **not built** | Abandoned deliberately. It is weakly coupled to the question the project actually answers (reliability under distribution shift), and evaluating generated molecules credibly needs a validation route we do not have. The time went to the application instead. |
| DiffDock vs Vina comparison | **built, run pending** | Promised and never run in the original phases. `src/42_diffdock_compare.py` runs DiffDock in its official container, since it pins Python 3.9.18 / torch 1.13.1+cu117 against this project's 3.11.17 / 2.14.1+cu126. Scored with the same RMSD function as Vina. |
| PoseBusters physical-validity checks | **built** | Integrated into `src/41_docking_scaleup.py`: 28 checks, reported separately from RMSD so a pose that is roughly in place but chemically impossible is not counted as a success. |
| PDBbind as a large structural database | **partly** | The refined-set affinity table (5,311 complexes, pKd/pKi) is mirrored and inventoried — this project's first binding data with a physical unit and its first external affinity validation set. The official structure archive requires registration, so coordinates for any docked subset come from RCSB. |
| FastAPI + Redis-RQ + PostgreSQL + React stack | **simplified** | FastAPI and React were built. The queue and database were dropped: inference takes 0.16 s for eleven molecules, so neither changes any answer the tool gives, and the course grades functionality rather than architecture. |
| Atom attribution in the 3D view | **2D only** | Per-atom attribution is computed and shown on the 2D depiction. The 3D panel shows the reference co-crystal, not a docked pose of the queried molecule, because docking takes tens of seconds per molecule and runs offline. The interface states this on screen. |

---

## 7. Future work

Ordered by expected value, not by effort.

1. **Rebuild the toxicity results on the original Tox21 challenge data**
   (🤗 `ml-jku/tox21`, with official CV folds). This is the only item that would
   change an existing conclusion rather than add one: it makes our toxicity
   numbers comparable with a decade of published work, and the same group hosts
   GIN and random-forest baselines on that data for direct comparison.
2. **A dual-OOD split** (cold-drug ∧ cold-target). The literature finds
   architecture rankings *reverse* there, and our modality ablation already
   points the same way on cold-target alone. Two hours of work on an existing
   script; the single most informative missing cell.
3. **Count-based fingerprints and the CLAMP embedding.** We used binary Morgan
   bits; the benchmark that reproduced our negative result used count-ECFP and
   found CLAMP to be the one representation that significantly beats it. If a
   stronger baseline widens the gap further, that is a cleaner version of the
   result we already have.
4. **A STORM-style ensemble for affinity** (arXiv 2611.23777): many small
   fusions of diverse protein and molecule representations, reported to match
   Boltz-2 at a fraction of the cost. It is the same architectural family as our
   winning frozen-embedding model and reuses the features we already cache.
5. **Prospective validation.** Nothing here was tested on a molecule whose label
   was unknown at prediction time. Until that happens, every number in this
   report describes retrospective behaviour on curated benchmarks.

---

## 8. A note on the supply chain

`requirements.txt` pins `fair-esm==2.0.0`, whose upstream repository is now
archived (last commit 2024-02-07) and redirects to `Biohub/esm-archive`. This
does not affect anyone cloning the repository — the 229 embeddings are committed
— but regenerating them depends on an archived library and a third-party CDN.
Migrating the embedding step to `transformers`, which is already pinned, would
remove that dependency.
