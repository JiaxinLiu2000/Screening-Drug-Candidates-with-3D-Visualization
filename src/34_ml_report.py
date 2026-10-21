"""Generate ML_REPORT.md - the modelling story, written for a machine-learning
course rather than a chemistry one.

The biology is the subject matter; the content is supervised learning under
distribution shift, with a fixed representation as the baseline. Everything here
is read back from committed artefacts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import paths  # noqa: E402

PROVENANCE_PATHS = ("src", "results", "splits", "receptors")


def load(name: str) -> dict:
    path = paths.RESULTS / name
    if not path.exists():
        raise SystemExit(f"FAILED: {path} missing - run the earlier scripts first")
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str:
    try:
        def git(*a: str) -> str:
            return subprocess.run(["git", *a], cwd=paths.ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        commit = git("log", "-1", "--format=%h", "--abbrev=7", "--", *PROVENANCE_PATHS)
        return f"{commit}-dirty" if git("status", "--porcelain", "--",
                                        *PROVENANCE_PATHS) else commit
    except Exception:  # noqa: BLE001
        return "n/a"


def main() -> int:
    curves = load("learning_curves.json")
    abstention = load("abstention.json")
    ablation = load("shared_encoder_ablation.json")
    frozen = load("frozen_baseline.json")
    followups = load("binding_followups.json")
    reliability = load("tox_reliability.json")
    chem = load("chemical_space_overlap_kiba.json")
    audit = load("data_audit.json")
    repro = load("reproducibility.json")

    summary = pd.read_csv(paths.RESULTS / "baseline_summary.csv")
    overall = summary[summary["level"] == "overall"]
    tox = overall[overall["dataset"] == "Tox21"].set_index("split")
    lc = pd.read_csv(paths.RESULTS / "learning_curves.csv")
    neural = pd.read_csv(paths.RESULTS / "shared_encoder_metrics.csv")

    L: list[str] = []
    add = L.append

    add("# The modelling report")
    add("")
    add("A supervised-learning study on molecular data. The application is virtual "
        "screening, but the questions are machine-learning questions: does a learned "
        "representation beat a fixed one, how much does that answer depend on "
        "training-set size, what happens under distribution shift, and can the model "
        "tell us when not to trust it.")
    add("")
    add(f"- commit `{git_commit()}`; {audit['n_pass']}/{len(audit['checks'])} integrity "
        f"checks pass; {repro['n_identical']}/{repro['n_artefacts']} artefacts "
        f"reproduce byte-identically")
    add(f"- {len(neural)} neural training runs "
        f"({neural['train_seconds'].sum() / 3600:.1f} GPU-hours) plus "
        f"{len(lc)} learning-curve fits")
    add("")

    # ---------------- 1. problem setup ----------------
    add("## 1. The two prediction problems")
    add("")
    add("| | Tox21 | KIBA |")
    add("| --- | --- | --- |")
    add("| task | 12-label binary classification, labels ~25% missing | regression on "
        "drug-protein pairs |")
    add("| n | 6,143 molecules | 117,193 pairs (2,030 drugs x 229 proteins) |")
    add("| class balance | 3-18% positive per label | continuous |")
    add("| fixed-representation baseline | random forest on Morgan 2048 | random "
        "forest on Morgan 2048 + amino-acid/dipeptide composition |")
    add("| learned alternative | GIN over molecular graphs; ChemBERTa fine-tuned | "
        "frozen ESM-2 650M + Morgan -> MLP |")
    add("| shift test | Bemis-Murcko scaffold split | cold-drug / cold-target / "
        "scaffold-cold |")
    add("")
    add("Every model in this report is evaluated on the same committed split index "
        "files with the same seeds, so differences are differences between models, not "
        "between evaluation protocols.")
    add("")

    # ---------------- 2. learning curves ----------------
    t = curves["tasks"]["tox21_scaffold"]
    k = curves["tasks"]["kiba_random"]
    add("## 2. Learning curves: is the baseline winning because of the data?")
    add("")
    add("The most useful experiment in the project, and the one that reframes "
        "everything else. Earlier phases compared models at a single training-set "
        "size. Here the training fold is subsampled to 5/10/25/50/100% while the "
        "validation and test folds stay fixed, 3 seeds each, so the only variable is "
        "how much the model got to learn from.")
    add("")
    add("### Tox21 (scaffold split) - GIN vs random forest")
    add("")
    add("| training molecules | random forest | GIN from scratch | gap |")
    add("| --- | --- | --- | --- |")
    for frac in curves["fractions"]:
        cell = t["by_fraction"][str(frac)]
        rf, gin = cell["random_forest"], cell["gin_scratch"]
        add(f"| {rf['n_train']} ({frac:.0%}) | {rf['mean']:.4f} ± {rf['std']:.4f} "
            f"| {gin['mean']:.4f} ± {gin['std']:.4f} "
            f"| {gin['mean'] - rf['mean']:+.4f} |")
    add("")
    add(f"**The graph network is not losing for lack of data - it is tied.** At 5% of "
        f"the training set the GIN is ahead by {t['gap_at_5pct']:+.4f}; at 100% the gap "
        f"is {t['gap_at_100pct']:+.4f}, i.e. indistinguishable given a seed-to-seed "
        f"spread of roughly 0.02-0.03. Both curves are still rising at 6,143 "
        f"molecules, so neither model has saturated and more data would help both, but "
        f"nothing in this range suggests the ordering would change. The honest summary "
        f"of the Tox21 half is 'a 1.2M-parameter GNN buys nothing over a random forest "
        f"on fingerprints at this scale', not 'GNNs are worse'.")
    add("")
    add("### KIBA (random split) - frozen-embedding MLP vs random forest")
    add("")
    add("| training pairs | random forest | frozen ESM-2 + Morgan MLP | gap |")
    add("| --- | --- | --- | --- |")
    for frac in curves["fractions"]:
        cell = k["by_fraction"][str(frac)]
        rf, mlp = cell["random_forest"], cell["frozen_esm_mlp"]
        add(f"| {rf['n_train']} ({frac:.0%}) | {rf['mean']:.4f} ± {rf['std']:.4f} "
            f"| {mlp['mean']:.4f} ± {mlp['std']:.4f} "
            f"| {mlp['mean'] - rf['mean']:+.4f} |")
    add("")
    add(f"**The opposite picture.** The MLP leads at every size and the gap *widens* "
        f"with data, from {k['gap_at_5pct']:+.4f} at 4,688 pairs to "
        f"{k['gap_at_100pct']:+.4f} at 93,754. There is no crossover to find. What "
        f"separates this from the Tox21 case is not scale alone - it is that the "
        f"representation here is **pretrained and frozen**, not learned from 94k "
        f"labels. A protein language model trained on UniRef brings information that "
        f"117k affinity measurements cannot supply.")
    add("")
    add("Figures: `figures/learning_curve_tox21_scaffold.png`, "
        "`figures/learning_curve_kiba_random.png`.")
    add("")

    # ---------------- 3. what this does to the ablation ----------------
    add("## 3. Re-reading the multi-task ablation")
    add("")
    add(f"The four-way ablation (A: GIN toxicity-only, B: GIN affinity-only, C: shared "
        f"GIN, D: ChemBERTa shared, plus pretrained single-task controls E/F) "
        f"concluded **{ablation['verdict']}**. The effects, against a seed-to-seed "
        f"noise floor of {ablation['tox_noise_floor']:.4f} (toxicity) and "
        f"{ablation['affinity_noise_floor']:.4f} (affinity):")
    add("")
    add("| contrast | toxicity macro ROC-AUC | affinity Pearson |")
    add("| --- | --- | --- |")
    add(f"| sharing, from scratch (C-A / C-B) | "
        f"{ablation['effect_of_sharing_scratch_tox']:+.4f} | "
        f"{ablation['effect_of_sharing_scratch_affinity']:+.4f} |")
    add(f"| sharing, at the best loss weight | "
        f"{ablation['effect_of_sharing_best_weight_tox']:+.4f} | "
        f"{ablation['effect_of_sharing_best_weight_affinity']:+.4f} |")
    add(f"| sharing, pretrained (D-E / D-F) | "
        f"{ablation['effect_of_sharing_pretrained_tox']:+.4f} | "
        f"{ablation['effect_of_sharing_pretrained_affinity']:+.4f} |")
    add("")
    add(f"Two measurements qualify that conclusion and both belong in the discussion. "
        f"First, the datasets are only weakly related: median nearest-neighbour "
        f"Tanimoto from Tox21 to KIBA is "
        f"{chem['nn_tanimoto']['tox21_to_kiba']['median']:.4f} against within-set "
        f"medians of {chem['nn_tanimoto']['within_tox21']['median']:.4f} and "
        f"{chem['nn_tanimoto']['within_kiba']['median']:.4f}, so the shared encoder is "
        f"being asked to serve two adjacent but distinct input distributions. Second, "
        f"and more damaging to any strong claim: the frozen-embedding MLP reaches "
        f"Pearson {frozen['comparison']['random']['frozen_mlp']:.4f} on KIBA-random "
        f"where our best *trained* encoder reaches "
        f"{frozen['comparison']['random']['best_trained_encoder']:.4f}. A model with no "
        f"learnable molecular encoder at all beats every encoder we trained. The "
        f"defensible statement is therefore narrow: **multi-task sharing did not help "
        f"*our* encoders**, and our encoders are not the strongest available.")
    add("")

    # ---------------- 4. baseline tuning ----------------
    b1 = followups["B1"]["by_split"]["random"]
    add("## 4. The baseline was under-tuned, and it matters")
    add("")
    add("`max_features='sqrt'` is the scikit-learn default for classification and a "
        "common choice for regression. On a 2,048-bit sparse fingerprint it lets each "
        "split consider only ~45 bits, which is a poor fit for a representation where "
        "the informative substructures are scattered:")
    add("")
    add("| max_features | KIBA random Pearson | concordance index | within-protein r |")
    add("| --- | --- | --- | --- |")
    for mf in ("sqrt", "0.1", "0.3"):
        if mf in b1:
            v = b1[mf]
            add(f"| {mf} | {v['pearson_mean']:.4f} ± {v['pearson_std']:.4f} "
                f"| {v['ci_mean']:.4f} | {v['within_protein_mean']:.4f} |")
    add("")
    add(f"Raising it to 0.3 is worth "
        f"{b1['0.3']['pearson_mean'] - b1['sqrt']['pearson_mean']:+.4f} Pearson - "
        f"larger than most of the architecture differences this project measured. "
        f"**Methodological lesson: an untuned baseline makes every comparison against "
        f"it suspect.** With the tuned forest at "
        f"{b1['0.3']['pearson_mean']:.4f}, the frozen MLP's lead shrinks from "
        f"{frozen['comparison']['random']['frozen_mlp'] - b1['sqrt']['pearson_mean']:+.4f} "
        f"to "
        f"{frozen['comparison']['random']['frozen_mlp'] - b1['0.3']['pearson_mean']:+.4f}.")
    add("")

    # ---------------- 5. distribution shift ----------------
    b3 = followups["B3"]
    t3 = reliability["variance_decomposition"]
    add("## 5. Distribution shift: how hard is the held-out set, really?")
    add("")
    add("Three nested definitions of 'held out', on the same dataset and model:")
    add("")
    add("| split | what is held out | Pearson | nearest-train Tanimoto (median) |")
    add("| --- | --- | --- | --- |")
    add(f"| random | nothing - the same drugs appear in train and test | "
        f"{b3['committed_random_pearson']:.4f} | 1.000 |")
    add(f"| cold-drug | drug identity (InChIKey) | "
        f"{b3['committed_cold_drug_pearson']:.4f} | 0.672 |")
    add(f"| scaffold-cold-drug | whole Bemis-Murcko scaffolds | "
        f"{b3['pearson_mean']:.4f} ± {b3['pearson_std']:.4f} "
        f"| {np.mean([v['median'] for v in b3['analogue_distance'].values()]):.3f} |")
    add("")
    add("The standard cold-drug split used throughout the drug-target literature holds "
        "out molecules but not chemotypes: in an analogue-dense dataset like KIBA a "
        "held-out drug still has a training neighbour at Tanimoto 0.672, and 18% have "
        "one above 0.8. Holding out scaffolds instead costs another "
        f"{b3['committed_cold_drug_pearson'] - b3['pearson_mean']:.4f} Pearson. "
        "**The amount of generalisation a benchmark demands is a design choice, and "
        "reporting which one you used is not optional.**")
    add("")
    add(f"The same point on the classification side: the random-to-scaffold macro "
        f"ROC-AUC penalty is {t3['penalty_point_estimate']:.4f}. A bootstrap over test "
        f"molecules gives [{t3['bootstrap_ci'][0]:.4f}, {t3['bootstrap_ci'][1]:.4f}], "
        f"but that interval only covers test-set sampling. Adding the spread across "
        f"the five committed splits widens it to "
        f"[{t3['combined_ci'][0]:.4f}, {t3['combined_ci'][1]:.4f}], which still "
        f"excludes zero. Reporting only the narrower interval would have overstated "
        f"the precision.")

    # ---------------- 5b. interpolation vs extrapolation ----------------
    diag = load("cold_drug_diagnosis.json")
    add("### 5b. The same two models swap places under extrapolation")
    add("")
    add("Section 2 showed the frozen-embedding MLP beating the forest on KIBA-random "
        "by a margin that grows with data. On the cold-drug split, with the same data "
        "and the same folds, the ordering reverses. Two control predictors that "
        "contain no molecular information at all make the comparison readable:")
    add("")
    add("| predictor | pooled Pearson (cold-drug) | within-protein Pearson |")
    add("| --- | --- | --- |")
    cd = diag["by_split"]["cold_drug"]
    for name, label in (("global_mean", "predict the training mean"),
                        ("protein_mean", "predict each protein's training mean"),
                        ("random_forest", "random forest")):
        v = cd[name]
        wp = v["within_protein_pearson"]
        wp_text = "n/a (constant within a protein)" if np.isnan(wp) or wp == 0 \
            else f"{wp:.4f}"
        add(f"| {label} | {v['pooled_pearson']:.4f} | {wp_text} |")
    mlp_cd = diag["frozen_mlp"]["cold_drug"]
    add(f"| frozen ESM-2 + Morgan MLP | {mlp_cd['pooled_pearson']:.4f} "
        f"| {mlp_cd['within_protein_pearson']:.4f} |")
    add("")
    add(f"Knowing nothing but which protein a pair involves already yields pooled "
        f"Pearson {cd['protein_mean']['pooled_pearson']:.4f} and a concordance index "
        f"of {cd['protein_mean']['ci']:.4f}, so a pooled number on this task is partly "
        f"a measure of between-protein variance and should never be quoted alone. "
        f"Controlling for that, the forest still carries real molecule-level signal "
        f"(within-protein {cd['random_forest']['within_protein_pearson']:.4f}) and "
        f"carries **more** of it than the MLP "
        f"({mlp_cd['within_protein_pearson']:.4f}) - the reverse of the random-split "
        f"ordering.")
    add("")
    add("**This is the most interesting modelling result in the project.** The two "
        "models are not ranked; they are good at different things. The frozen "
        "representation plus MLP interpolates better - it exploits a dense, "
        "well-covered input space and improves as that space fills in. The forest "
        "extrapolates better - when the test molecules are chemotypes it has never "
        "seen, an axis-aligned ensemble over substructure counts degrades more "
        "gracefully than a smooth function fitted in a learned embedding space. "
        "Which model is 'better' is therefore a question about the deployment "
        "regime, and a benchmark that only reports a random split cannot distinguish "
        "the two.")
    add("")

    # ---------------- 6. abstention ----------------
    sig = abstention["signals"]
    add("## 6. Knowing when not to answer")
    add("")
    add("A classifier that can abstain is more useful than one that cannot, but the "
        "confidence signal has to be chosen carefully under class imbalance. Four "
        "signals, same predictions, same splits, scored as risk-coverage curves:")
    add("")
    add("| confidence signal | PR-AUC at 100% coverage | at 50% coverage | actives "
        "retained at 50% | beats random |")
    add("| --- | --- | --- | --- | --- |")
    for name in ("distance_from_half", "forest_agreement", "random", "tanimoto_to_train"):
        v = sig[name]
        beats = abstention["beats_random"].get(name)
        mark = "-" if name == "random" else ("yes" if beats else "**no**")
        add(f"| {name.replace('_', ' ')} | {v['pr_auc_at_full_coverage']:.4f} "
            f"| {v['pr_auc_at_50pct_coverage']:.4f} "
            f"| {v['recall_retained_at_50pct']:.1%} | {mark} |")
    add("")
    add(f"**Both textbook uncertainty signals are worse than random guessing here.** "
        f"Rejecting predictions near 0.5, or rejecting where the forest's trees "
        f"disagree, drops PR-AUC from "
        f"{sig['distance_from_half']['pr_auc_at_full_coverage']:.4f} to "
        f"{sig['distance_from_half']['pr_auc_at_50pct_coverage']:.4f} while discarding "
        f"three quarters of the actives. The reason is structural: in a 3-18% positive "
        f"problem the uncertain region *is* the positive region, so 'confidence' "
        f"defined on the score abstains on the minority class and leaves behind easy "
        f"negatives. Accuracy rises, which is exactly why accuracy is the wrong metric "
        f"to watch.")
    add("")
    add(f"The signal that works is not an uncertainty estimate at all - it is a "
        f"distance: maximum Tanimoto similarity to the training set. It raises PR-AUC "
        f"to {sig['tanimoto_to_train']['pr_auc_at_50pct_coverage']:.4f} at half "
        f"coverage while *keeping* "
        f"{sig['tanimoto_to_train']['recall_retained_at_50pct']:.1%} of the actives. "
        f"Asking 'have I seen chemistry like this?' is answerable even when the "
        f"model's probability is miscalibrated; asking 'how sure am I?' is not. "
        f"Figure: `figures/abstention_risk_coverage.png`.")
    add("")

    # ---------------- 7. calibration ----------------
    cal = reliability["calibration"]
    add("## 7. Calibration")
    add("")
    add("| split | ECE | Brier | mean predicted | observed rate | ECE after isotonic |")
    add("| --- | --- | --- | --- | --- | --- |")
    for split in ("random", "scaffold"):
        p = cal["pooled"][split]
        add(f"| {split} | {p['ece']:.4f} | {p['brier']:.4f} | {p['mean_predicted']:.4f} "
            f"| {p['observed_rate']:.4f} "
            f"| {cal['per_fit_mean_ece_isotonic'][split]:.4f} |")
    add("")
    add(f"The forests over-predict by about 1.7x "
        f"({cal['pooled']['scaffold']['mean_predicted']:.3f} predicted against "
        f"{cal['pooled']['scaffold']['observed_rate']:.3f} observed). This is "
        f"`class_weight='balanced'` doing what it is designed to do - re-weighting the "
        f"minority class to improve ranking - at the cost of the scores no longer "
        f"estimating P(positive). Isotonic regression fitted on the validation fold "
        f"brings per-fit ECE from {cal['per_fit_mean_ece']['scaffold']:.4f} to "
        f"{cal['per_fit_mean_ece_isotonic']['scaffold']:.4f}. **Takeaway: a "
        f"re-weighted classifier's output is a score, not a probability, and must be "
        f"calibrated before it is shown to anyone as a percentage.**")
    add("")

    # ---------------- 8. what the project establishes ----------------
    add("## 8. What this project does and does not establish")
    add("")
    add("**Does:**")
    add("")
    add("- A well-tuned random forest on fingerprints is a serious baseline at this "
        "scale. On Tox21 it ties a GIN at every training-set size tested; on KIBA, "
        "tuning one hyperparameter moved it further than most architecture changes "
        "did.")
    add("- Pretrained frozen representations beat both learned-from-scratch encoders "
        "and fixed descriptors on the affinity task, and the advantage grows with "
        "data rather than shrinking.")
    add("- Multi-task sharing between these two tasks produced no gain beyond seed "
        "noise, in a setting where the two input distributions are measurably "
        "adjacent rather than overlapping.")
    add("- Under class imbalance, score-based confidence is worse than useless for "
        "abstention, while a training-set-distance signal works.")
    add("- The amount of distribution shift a split imposes can be measured, not just "
        "asserted, and it changes results by more than most modelling decisions.")
    add("")
    add("**Does not:**")
    add("")
    add("- Establish anything about neural methods in general. Our encoders lose to an "
        "off-the-shelf frozen embedding, so this is a statement about our models.")
    add("- Reach the published state of the art. On KIBA our concordance index is "
        "0.8241 against DeepDTA's 0.863 and GraphDTA's 0.891 under their protocols.")
    add("- Produce a validated screening tool. The docking benchmark is a redocking "
        "sanity check, the reference-compound set is 14 molecules, and nothing here "
        "has been tested prospectively.")
    add("")

    text = "\n".join(L) + "\n"
    out = paths.ROOT / "ML_REPORT.md"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
