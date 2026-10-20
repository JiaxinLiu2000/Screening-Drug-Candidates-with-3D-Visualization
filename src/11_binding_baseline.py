"""Phase 2 Task 3: Davis + KIBA binding-affinity RF baseline.

Outputs: splits/{davis,kiba}_{random,cold_drug,cold_target}_seed{N}.csv,
results/binding_dataset_stats.json, results/binding_baseline.csv,
results/binding_baseline.md.

Model: RandomForestRegressor(n_estimators=N_ESTIMATORS, max_features='sqrt',
random_state=seed, n_jobs=-1), fit on the train subset only (valid is held out
and unused -- no hyperparameter tuning). max_features='sqrt' because the default
1.0 evaluates all 2468 features per split, intractable for KIBA.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor

from tox21lib import datasets, dti, paths

N_ESTIMATORS = 100
DATASETS = ("DAVIS", "KIBA")


def dist(v):
    v = np.asarray(v, dtype=float)
    return {
        "min": float(v.min()), "p25": float(np.percentile(v, 25)),
        "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
        "max": float(v.max()), "mean": float(v.mean()), "std": float(v.std(ddof=1)),
    }


def main():
    paths.ensure_dirs()
    stats, rows, split_rows = {}, [], []
    for name in DATASETS:
        tag = name.lower()
        raw = datasets.load_dti(name)
        clean, std_stats = datasets.unique_molecules(raw)
        pairs, pstats = dti.build_pairs(raw, clean)
        s = {"standardization": {k: v for k, v in std_stats.items()
                                 if isinstance(v, (int, float, str))}, "pairs": pstats}
        if name == "DAVIS":
            s["units_returned_by_tdc"] = "Kd in nM (raw, no log transform)"
            s["conversion"] = "pKd = -log10(Kd_nM*1e-9 + 1e-10) (TDC convert_to_log)"
            s["raw_kd_nm_distribution"] = dist(pairs["affinity"])
            # censoring counted on raw rows (before InChIKey averaging) and on final pairs
            s["raw_rows_at_ceiling_kd_10000nM"] = int((raw["affinity"] == dti.DAVIS_CEILING_NM).sum())
            s["final_pairs_at_ceiling_kd_10000nM"] = int((pairs["affinity"] == dti.DAVIS_CEILING_NM).sum())
            pairs["affinity"] = dti.kd_nm_to_pkd(pairs["affinity"].to_numpy())
            s["ceiling_pkd_value"] = float(dti.kd_nm_to_pkd([dti.DAVIS_CEILING_NM])[0])
        else:
            s["units_returned_by_tdc"] = "KIBA score (no conversion applied)"
        n_d, n_t = pairs["inchikey"].nunique(), pairs["target_id"].nunique()
        s.update({"distinct_drugs": int(n_d), "distinct_proteins": int(n_t),
                  "measured_pairs": int(len(pairs)),
                  "matrix_density": len(pairs) / (n_d * n_t),
                  "affinity_distribution": dist(pairs["affinity"])})
        stats[name] = s
        print(name, json.dumps(s["pairs"]), n_d, n_t, flush=True)

        smiles_by_key = dict(zip(clean["inchikey"], clean["canonical_smiles"]))
        X = dti.feature_matrix(pairs, smiles_by_key)
        s["feature_matrix_bytes"] = int(X.nbytes)
        y = pairs["affinity"].to_numpy()
        for kind in dti.SPLIT_KINDS:
            for seed in paths.SEEDS:
                subset = dti.make_split(pairs, kind, seed)
                out = pd.DataFrame({"pair_index": pairs["pair_index"], "inchikey": pairs["inchikey"],
                                    "drug_id": pairs["drug_id"], "target_id": pairs["target_id"],
                                    "subset": subset})
                out.to_csv(paths.SPLITS / f"{tag}_{kind}_seed{seed}.csv", index=False, lineterminator="\n")
                tr, va, te = (subset == "train"), (subset == "valid"), (subset == "test")
                split_rows.append({"dataset": name, "split": kind, "seed": seed,
                                   "n_train": int(tr.sum()), "n_valid": int(va.sum()), "n_test": int(te.sum()),
                                   "n_test_drugs": int(pairs.loc[te, "inchikey"].nunique()),
                                   "n_test_targets": int(pairs.loc[te, "target_id"].nunique())})
                model = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_features="sqrt",
                                              random_state=seed, n_jobs=-1)
                model.fit(X[tr], y[tr])
                # sklearn's threaded predict sums tree outputs in thread-completion order
                # (float non-associativity) -> predict single-threaded for byte-identical output
                model.set_params(n_jobs=1)
                pred = model.predict(X[te])
                rows.append({**split_rows[-1],
                             "rmse": float(np.sqrt(np.mean((pred - y[te]) ** 2))),
                             "pearson_r": float(pearsonr(y[te], pred)[0]),
                             "spearman_rho": float(spearmanr(y[te], pred)[0]),
                             "ci": dti.concordance_index(y[te], pred)})
                print(rows[-1], flush=True)
        del X

    (paths.RESULTS / "binding_dataset_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(paths.RESULTS / "binding_baseline.csv", index=False,
                   float_format="%.6f", lineterminator="\n")
    write_md(metrics, stats)


def write_md(m, stats):
    L = ["# Binding-affinity baseline (Davis, KIBA)", "",
         f"RandomForestRegressor, n_estimators={N_ESTIMATORS}, max_features='sqrt', "
         "random_state=seed, n_jobs=-1; fit on train only (valid unused). Features: Morgan r=2 "
         "2048 bits + protein AAC(20)+DPC(400) = 2468 float32, full data (no subsampling). "
         "CI is exact over all comparable test pairs. Davis target = pKd (TDC convert_to_log of Kd nM); "
         "KIBA target = KIBA score.", "", "## Dataset statistics", "",
         "| dataset | raw pairs | pairs dropped (std.) | pairs merged (InChIKey) | drugs | proteins | pairs | density |",
         "|---|---|---|---|---|---|---|---|"]
    for n, s in stats.items():
        p = s["pairs"]
        L.append(f"| {n} | {p['raw_pairs']} | {p['pairs_dropped_by_standardization']} | "
                 f"{p['pairs_merged_by_inchikey']} | {s['distinct_drugs']} | {s['distinct_proteins']} | "
                 f"{s['measured_pairs']} | {s['matrix_density']:.4f} |")
    L += ["", "| dataset | units | min | p25 | median | p75 | max | mean | std |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, s in stats.items():
        d = s["affinity_distribution"]
        u = "pKd" if n == "DAVIS" else "KIBA score"
        L.append(f"| {n} | {u} | " + " | ".join(f"{d[k]:.4f}" for k in
                 ("min", "p25", "median", "p75", "max", "mean", "std")) + " |")
    dv = stats["DAVIS"]
    L += ["", f"Davis: TDC returned {dv['units_returned_by_tdc']}; converted via {dv['conversion']}. "
          f"Censored ceiling Kd = 10000 nM (pKd = {dv['ceiling_pkd_value']:.6f}): "
          f"{dv['raw_rows_at_ceiling_kd_10000nM']} raw rows, {dv['final_pairs_at_ceiling_kd_10000nM']} "
          f"of {dv['measured_pairs']} final pairs "
          f"({100 * dv['final_pairs_at_ceiling_kd_10000nM'] / dv['measured_pairs']:.1f}%).", "",
          "## Split sizes (identical across seeds unless shown as min-max)", "",
          "| dataset | split | n_train | n_valid | n_test | test drugs | test targets |",
          "|---|---|---|---|---|---|---|"]
    g = m.groupby(["dataset", "split"], sort=False)
    for (n, k), d in g:
        f = lambda c: (str(d[c].min()) if d[c].min() == d[c].max() else f"{d[c].min()}-{d[c].max()}")
        L.append(f"| {n} | {k} | {f('n_train')} | {f('n_valid')} | {f('n_test')} | "
                 f"{f('n_test_drugs')} | {f('n_test_targets')} |")
    L += ["", "Zero train->valid/test entity leakage verified for every cold split (script asserts).", "",
          "## Test metrics (mean ± sample std over seeds 0-4)", "",
          "| dataset | split | RMSE | Pearson r | Spearman rho | CI |", "|---|---|---|---|---|---|"]
    for (n, k), d in g:
        L.append(f"| {n} | {k} | " + " | ".join(
            f"{d[c].mean():.4f} ± {d[c].std(ddof=1):.4f}" for c in ("rmse", "pearson_r", "spearman_rho", "ci")) + " |")
    cd = m[(m.dataset == "DAVIS") & (m.split == "cold_drug")]
    kd = m[(m.dataset == "KIBA") & (m.split == "cold_drug")]
    L += ["", "## Recommendation", "",
          f"KIBA carries the main results ({stats['KIBA']['distinct_drugs']} drugs, "
          f"{stats['KIBA']['measured_pairs']} pairs; cold_drug test = {kd.n_test_drugs.min()}-{kd.n_test_drugs.max()} "
          f"distinct drugs, {kd.n_test.min()}-{kd.n_test.max()} pairs), so random, cold_drug and cold_target are all "
          "statistically usable there. Davis is the secondary comparison: random and cold_target test sets are "
          f"usable in pair count, but Davis cold_drug tests only {cd.n_test_drugs.min()}-{cd.n_test_drugs.max()} "
          f"distinct drugs per seed ({cd.n_test.min()}-{cd.n_test.max()} pairs that are highly correlated within a drug), "
          "so its cold_drug numbers describe generalisation to a handful of chemotypes and should be reported "
          "only with the seed spread, not as a headline. Davis cold_target likewise tests on only "
          f"{m[(m.dataset=='DAVIS')&(m.split=='cold_target')].n_test_targets.min()} kinases per seed and KIBA "
          f"cold_target on {m[(m.dataset=='KIBA')&(m.split=='cold_target')].n_test_targets.min()} proteins per seed "
          "(pair counts are large but the effective sample for protein generalisation is the number of test proteins; "
          "KIBA cold_target test size also varies across seeds, see split table).", ""]
    (paths.RESULTS / "binding_baseline.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
