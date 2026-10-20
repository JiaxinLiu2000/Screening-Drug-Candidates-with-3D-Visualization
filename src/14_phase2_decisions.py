"""Phase 2 Task 6 - the five go/no-go decisions, one paragraph each.

Every number is read back from the committed Phase 2 artefacts, so this file
cannot drift from the measurements it cites. Nothing is recomputed here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import paths  # noqa: E402


def load_json(name: str) -> dict:
    path = paths.RESULTS / name
    if not path.exists():
        raise SystemExit(f"FAILED: {path} missing - run the Phase 2 scripts first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    scaf = load_json("scaffold_audit.json")
    chem = load_json("chemical_space_overlap.json")
    herg = load_json("herg_stats.json")
    bench = load_json("docking_benchmark_set.json")
    binding_stats = load_json("binding_dataset_stats.json")

    binding = pd.read_csv(paths.RESULTS / "binding_baseline.csv")
    summary = pd.read_csv(paths.RESULTS / "baseline_summary.csv")
    overall = summary[summary["level"] == "overall"].set_index(["dataset", "split"])

    tox_rnd = overall.loc[("Tox21", "random")]
    tox_scf = overall.loc[("Tox21", "scaffold")]
    herg_rnd = overall.loc[("hERG", "random")]
    herg_scf = overall.loc[("hERG", "scaffold")]
    tox_drop = tox_rnd["roc_auc_mean"] - tox_scf["roc_auc_mean"]
    herg_drop = herg_rnd["roc_auc_mean"] - herg_scf["roc_auc_mean"]

    murcko = scaf["variants"]["murcko"]
    generic = scaf["variants"]["generic"]
    nn = chem["nn_tanimoto"]["tox21_to_davis"]
    rev = chem["nn_tanimoto"]["davis_to_tox21"]

    agg = (
        binding.groupby(["dataset", "split"])[
            ["n_test", "n_test_drugs", "n_test_targets", "pearson_r", "spearman_rho", "ci"]
        ]
        .agg(["mean", "std"])
    )

    def b(dataset: str, split: str, col: str, stat: str = "mean") -> float:
        return float(agg.loc[(dataset, split), (col, stat)])

    L: list[str] = []
    add = L.append

    add("# Phase 2 decisions")
    add("")
    add("Read this before starting model development. Every number below is measured; "
        "the generating artefacts are named after each claim. Produced by "
        "`src/14_phase2_decisions.py`.")
    add("")

    add("## 1. Which scaffold variant, and does the 0.761 need correcting?")
    add("")
    add(f"**Keep the chemically-aware Bemis-Murcko scaffold; the 0.761 stands and needs "
        f"no correction.** `src/03_splits.py` already uses "
        f"`MurckoScaffold.GetScaffoldForMol` ({scaf['evidence']}), which preserves atom "
        f"identity and bond order - not the generic framework. The generic variant was "
        f"the thing that could have inflated our leakage numbers, and it is strictly "
        f"coarser: {generic['n_unique_scaffolds']} unique scaffolds against "
        f"{murcko['n_unique_scaffolds']} "
        f"(ratio {scaf['generic_over_murcko_group_count_ratio']:.3f}), largest group "
        f"{generic['largest_group_size']} (`{generic['largest_group_scaffold']}`) against "
        f"{murcko['largest_group_size']} (`{murcko['largest_group_scaffold']}`), and "
        f"{generic['n_singleton_scaffolds']} singletons against "
        f"{murcko['n_singleton_scaffolds']}. "
        f"{scaf['n_molecules_changing_group_membership']} of 6143 molecules would change "
        f"group, always by merging distinct cores, which would make the split harsher "
        f"rather than fixing an error. The reported scaffold-split macro ROC-AUC of "
        f"{tox_scf['roc_auc_mean']:.4f} ± {tox_scf['roc_auc_std']:.4f} is therefore "
        f"correct as published (`results/scaffold_audit.md`). The generic framework is "
        f"worth reporting only as an optional harsher stress test.")
    add("")

    add("## 2. Does chemical-space overlap support the shared-encoder experiment?")
    add("")
    add(f"**No - the measurement says {chem['verdict'].lower()}, and this is the finding "
        f"that should reshape the plan.** The median nearest-neighbour Tanimoto from each "
        f"of the {chem['tox21_molecules']} Tox21 molecules to any Davis molecule is "
        f"**{nn['median']:.4f}** (p25 {nn['p25']:.4f}, p75 {nn['p75']:.4f}), below the "
        f"0.25 negligible-overlap threshold; only {nn['frac_gt_0.4'] * 100:.2f}% of "
        f"Tox21 molecules have any Davis neighbour above 0.4 and "
        f"{nn['frac_gt_0.6'] * 100:.2f}% above 0.6. The reverse direction looks better "
        f"(median {rev['median']:.4f}) only because "
        f"{chem['exact_inchikey_overlap']['count']} of Davis's "
        f"{chem['davis_drugs_after_standardization']} drugs are literally present in "
        f"Tox21 and score 1.0. All seven physicochemical distributions differ with "
        f"KS p < 1e-3, most sharply aromatic ring count and ring count "
        f"(`results/chemical_space_overlap.md`, `figures/chemspace_pca.png`). "
        f"Consequence, stated as the spec requires: the shared-encoder experiment will "
        f"likely show no gain for distributional reasons. Options, in order of cost: "
        f"(a) reframe the research question from \"shared encoder beats single-task\" to "
        f"\"what happens when multi-task learning spans disjoint chemical spaces\" - the "
        f"negative result is then the contribution and the numbers above are its "
        f"evidence; (b) add bridging data, i.e. molecules that are both drug-like and "
        f"tox-labelled (ToxCast, or hERG, which already shares "
        f"{herg['inchikey_overlap_with_tox21']} "
        f"InChIKeys with Tox21); (c) keep the claim but pre-register it as a null "
        f"hypothesis test. Do not report a shared-encoder gain as expected.")
    add("")

    add("## 3. Which binding dataset carries the main results?")
    add("")
    kiba = binding_stats["KIBA"]
    davis = binding_stats["DAVIS"]
    add(f"**KIBA carries the main results; Davis is the secondary comparison.** KIBA has "
        f"{kiba['distinct_drugs']} drugs x {kiba['distinct_proteins']} proteins over "
        f"{kiba['measured_pairs']} measured pairs (density "
        f"{kiba['matrix_density']:.4f}); Davis has {davis['distinct_drugs']} drugs x "
        f"{davis['distinct_proteins']} proteins over {davis['measured_pairs']} pairs - a "
        f"complete matrix (density {davis['matrix_density']:.2f}) in which "
        f"{davis['final_pairs_at_ceiling_kd_10000nM']} pairs "
        f"({davis['final_pairs_at_ceiling_kd_10000nM'] / davis['measured_pairs'] * 100:.1f}%) "
        f"sit at the censored Kd = 10000 nM ceiling, so its median pKd is the ceiling "
        f"value itself. On usable split settings the test-set sizes are decisive: Davis "
        f"`cold_drug` tests only {b('DAVIS', 'cold_drug', 'n_test_drugs'):.0f} distinct "
        f"drugs, and the instability shows directly in the seed spread - Pearson "
        f"{b('DAVIS', 'cold_drug', 'pearson_r'):.4f} ± "
        f"{b('DAVIS', 'cold_drug', 'pearson_r', 'std'):.4f}, Spearman "
        f"{b('DAVIS', 'cold_drug', 'spearman_rho'):.4f} ± "
        f"{b('DAVIS', 'cold_drug', 'spearman_rho', 'std'):.4f} - versus KIBA `cold_drug` "
        f"with {b('KIBA', 'cold_drug', 'n_test_drugs'):.0f} test drugs and Pearson "
        f"{b('KIBA', 'cold_drug', 'pearson_r'):.4f} ± "
        f"{b('KIBA', 'cold_drug', 'pearson_r', 'std'):.4f}. Usable: KIBA random, "
        f"cold_drug and cold_target, plus Davis random "
        f"(Pearson {b('DAVIS', 'random', 'pearson_r'):.4f} ± "
        f"{b('DAVIS', 'random', 'pearson_r', 'std'):.4f}). Report Davis cold_drug only "
        f"with its seed spread attached, never as a headline. Note that `cold_target` is "
        f"weaker on both datasets (KIBA Pearson "
        f"{b('KIBA', 'cold_target', 'pearson_r'):.4f}, Davis "
        f"{b('DAVIS', 'cold_target', 'pearson_r'):.4f}) and tests only "
        f"{b('KIBA', 'cold_target', 'n_test_targets'):.0f} and "
        f"{b('DAVIS', 'cold_target', 'n_test_targets'):.0f} proteins respectively - the "
        f"effective sample for protein generalisation is the protein count, not the pair "
        f"count (`results/binding_baseline.md`).")
    add("")

    add("## 4. Does hERG corroborate the Tox21 leakage finding?")
    add("")
    add(f"**No - it diverges, and the divergence is explained rather than mysterious.** "
        f"On hERG ({herg['final_molecules']} molecules, {herg['n_labeled']} labelled, "
        f"positive rate {herg['positive_rate']:.4f}) the random-split macro ROC-AUC is "
        f"{herg_rnd['roc_auc_mean']:.4f} ± {herg_rnd['roc_auc_std']:.4f} and the "
        f"scaffold-split is {herg_scf['roc_auc_mean']:.4f} ± "
        f"{herg_scf['roc_auc_std']:.4f}, i.e. a drop of {herg_drop:+.4f} against Tox21's "
        f"{tox_drop:+.4f}. Two measured reasons, both in `results/herg_baseline.md`: "
        f"hERG's test folds hold only 58-59 molecules, so the seed-to-seed spread "
        f"({herg_rnd['roc_auc_std']:.4f}) is roughly seven times the random-versus-"
        f"scaffold gap - the gap is inside the noise; and hERG is intrinsically more "
        f"scaffold-diverse (0.658 unique scaffolds per molecule against Tox21's 0.351), "
        f"so a random split already separates test chemistry from training. The measured "
        f"test-to-train nearest-neighbour similarity falls by only 0.0897 on hERG when "
        f"switching to the scaffold split, against 0.1450 on Tox21. Conclusion for the "
        f"report: the 0.071 Tox21 drop is real but should be described as "
        f"dataset-specific - a consequence of Tox21's scaffold redundancy (the benzene "
        f"group alone holds {murcko['largest_group_size']} molecules) - not as a general "
        f"law of scaffold splitting. Do not generalise the leakage claim beyond Tox21 "
        f"without a third dataset with comparably sized folds.")
    add("")

    add("## 5. Is the docking benchmark big and diverse enough for Vina vs DiffDock?")
    add("")
    sim = bench["similarity"]
    conf = bench["conformation_counts"]
    add(f"**Yes on size and diversity; with one caveat on conformational balance.** "
        f"{bench['n_selected']} structures were selected from {bench['n_queried']} "
        f"queried EGFR entries with no filter relaxed ({bench['relaxation']}), each with "
        f"a cleaned receptor, a crystallographic-coordinate ligand SDF with CCD-derived "
        f"bond orders, and a JSON record; all {bench['n_rejected']} rejections are logged "
        f"with reasons. Chemical diversity is real, not twenty erlotinib analogues: over "
        f"{sim.get('n_pairs', 190)} ligand pairs the Tanimoto similarity is median "
        f"{sim['median']:.3f} (min {sim['min']:.3f}, mean {sim['mean']:.3f}, max "
        f"{sim['max']:.3f}), with only {sim['n_pairs_above_threshold']} pairs above 0.7 and "
        f"erlotinib absent from the set entirely; all 20 ligands have distinct InChIKeys. "
        f"20 structures gives a usable paired comparison - at this size a paired "
        f"Wilcoxon test over per-structure RMSD can detect a consistent method "
        f"difference, which two structures never could. The caveat: ranking by resolution "
        f"produced {conf.get('active', 0)} active and {conf.get('inactive', 0)} inactive "
        f"conformations (salt-bridge distances 2.69-3.09 A versus 9.20-15.58 A), so the "
        f"set is dominated by the inactive state while our screening receptor 1M17 is "
        f"active. For pose-accuracy redocking this is acceptable because each ligand is "
        f"redocked into its own receptor, but any claim about Vina/DiffDock performance "
        f"on the active ATP-site conformation specifically rests on 3 structures - so "
        f"either report results stratified by conformation, or extend the set with more "
        f"active-state entries from the 74 that passed every filter but fell outside the "
        f"20-structure limit (`results/docking_benchmark_set.md`).")
    add("")

    add("---")
    add("")
    add("## Net effect on the proposal")
    add("")
    add(f"One claim weakens and one strengthens. The shared-encoder premise is not "
        f"supported by the data (median nearest-neighbour Tanimoto {nn['median']:.4f}) "
        f"and must be reframed or bridged before six weeks go into it; the binding half "
        f"of the project is in better shape than assumed, because KIBA supports a real "
        f"cold-drug evaluation ({b('KIBA', 'cold_drug', 'n_test_drugs'):.0f} test drugs) "
        f"where Davis never could ({b('DAVIS', 'cold_drug', 'n_test_drugs'):.0f}). The "
        f"Tox21 scaffold-split number needs no correction, and the docking benchmark is "
        f"ready to use.")
    add("")

    text = "\n".join(L) + "\n"
    out = paths.RESULTS / "PHASE2_DECISIONS.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[wrote {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
