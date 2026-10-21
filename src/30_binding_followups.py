"""Phase 4 tasks B1 and B3 - is the affinity floor under-tuned, and is cold-drug
actually hard?

B1  `max_features='sqrt'` on a 2048-bit Morgan fingerprint lets each split see
    only ~45 of 2048 bits. The surveyed MADDIS repo measured +0.03 to +0.05
    scaffold R-squared from raising it. Every "neural did not beat the floor"
    claim rests on that floor, so it is worth knowing whether the floor is real.

B3  Phase 4 showed our KIBA cold-drug split is cold by molecule but warm by
    chemotype (median nearest-training-set Tanimoto 0.672). A scaffold-cold
    split holds out whole Bemis-Murcko scaffolds instead, which is strictly
    harder than anything the DTA literature reports for this dataset.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfg
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import datasets, dti, featurize, paths, splits as splitlib  # noqa: E402

MAX_FEATURES = ("sqrt", 0.1, 0.3)
B1_SPLITS = ("random", "cold_drug")
N_ESTIMATORS = 100


def kiba():
    raw = datasets.load_dti("KIBA")
    clean, _ = datasets.unique_molecules(raw)
    pairs, _ = dti.build_pairs(raw, clean)
    X = dti.feature_matrix(pairs, dict(zip(clean["inchikey"], clean["canonical_smiles"])))
    return pairs, clean, X


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "pearson_r": float(pearsonr(y, pred)[0]),
        "spearman_rho": float(spearmanr(y, pred)[0]),
        "ci": float(dti.concordance_index(y, pred)),
    }


def within_protein(pairs_test: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> float:
    df = pairs_test.copy()
    df["y"], df["pred"] = y, pred
    rs = [pearsonr(g["y"], g["pred"])[0] for _, g in df.groupby("target_id")
          if len(g) >= 10 and g["y"].nunique() > 1]
    return float(np.mean(rs)) if rs else float("nan")


def task_b1(pairs, X) -> pd.DataFrame:
    """Sweep max_features on the committed KIBA splits."""
    y = pairs["affinity"].to_numpy(dtype=float)
    rows = []
    for split in B1_SPLITS:
        for seed in paths.SEEDS:
            frame = pd.read_csv(paths.SPLITS / f"kiba_{split}_seed{seed}.csv")
            subsets = frame.sort_values("pair_index")["subset"].to_numpy()
            tr, te = subsets == "train", subsets == "test"
            for mf in MAX_FEATURES:
                t0 = time.time()
                model = RandomForestRegressor(
                    n_estimators=N_ESTIMATORS, max_features=mf, random_state=seed,
                    n_jobs=-1,
                ).fit(X[tr], y[tr])
                pred = model.predict(X[te])
                rows.append({
                    "task": "B1", "split": split, "seed": seed,
                    "max_features": str(mf),
                    **score(y[te], pred),
                    "within_protein_pearson": within_protein(pairs[te], y[te], pred),
                    "fit_seconds": round(time.time() - t0, 1),
                })
                r = rows[-1]
                print(f"  B1 {split:10s} seed={seed} max_features={mf!s:5s} "
                      f"r={r['pearson_r']:.4f} CI={r['ci']:.4f} "
                      f"[{r['fit_seconds']:.0f}s]")
    return pd.DataFrame(rows)


def scaffold_cold_split(pairs: pd.DataFrame, clean: pd.DataFrame, seed: int) -> np.ndarray:
    """Hold out whole Bemis-Murcko scaffolds of drugs, not just drug identities.

    Scaffold groups are assigned to train/valid/test largest-first, exactly as
    `tox21lib.splits.scaffold_split` does for molecules, then every pair inherits
    its drug's subset. The seed permutes equal-sized groups only.
    """
    smiles = dict(zip(clean["inchikey"], clean["canonical_smiles"]))
    keys = sorted(pairs["inchikey"].unique())
    scaffolds = pd.Series([splitlib.scaffold_smiles(smiles[k]) for k in keys])
    labels = splitlib.scaffold_split(scaffolds, seed)
    by_key = dict(zip(keys, labels))
    return pairs["inchikey"].map(by_key).to_numpy()


def task_b3(pairs, clean, X) -> tuple[pd.DataFrame, dict]:
    """Scaffold-cold-drug split: strictly harder than cold-drug by identity."""
    y = pairs["affinity"].to_numpy(dtype=float)
    gen = rfg.GetMorganGenerator(radius=featurize.MORGAN_RADIUS,
                                 fpSize=featurize.MORGAN_N_BITS)
    fps = {k: gen.GetFingerprint(Chem.MolFromSmiles(s))
           for k, s in zip(clean["inchikey"], clean["canonical_smiles"])}

    rows, analogue = [], {}
    for seed in paths.SEEDS:
        subsets = scaffold_cold_split(pairs, clean, seed)
        frame = pd.DataFrame({
            "pair_index": np.arange(len(pairs)),
            "inchikey": pairs["inchikey"].to_numpy(),
            "drug_id": pairs["drug_id"].to_numpy() if "drug_id" in pairs else "",
            "target_id": pairs["target_id"].to_numpy(),
            "subset": subsets,
        })
        frame.to_csv(paths.SPLITS / f"kiba_scaffold_cold_drug_seed{seed}.csv",
                     index=False, lineterminator="\n")
        tr, te = subsets == "train", subsets == "test"
        train_keys = sorted(set(pairs["inchikey"][tr]))
        test_keys = sorted(set(pairs["inchikey"][te]))
        if set(train_keys) & set(test_keys):
            raise RuntimeError(f"seed {seed}: scaffold-cold split leaks drugs")

        model = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_features="sqrt",
                                      random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
        pred = model.predict(X[te])
        train_fps = [fps[k] for k in train_keys]
        nn = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[k], train_fps))
                       for k in test_keys])
        rows.append({
            "task": "B3", "split": "scaffold_cold_drug", "seed": seed,
            "max_features": "sqrt",
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "n_test_drugs": len(test_keys),
            **score(y[te], pred),
            "within_protein_pearson": within_protein(pairs[te], y[te], pred),
            "nearest_train_tanimoto_median": float(np.median(nn)),
        })
        analogue[seed] = {
            "n_test_drugs": len(test_keys),
            "median": float(np.median(nn)),
            "fraction_above_0.8": float((nn > 0.8).mean()),
        }
        r = rows[-1]
        print(f"  B3 scaffold_cold seed={seed} drugs={r['n_test_drugs']:4d} "
              f"r={r['pearson_r']:.4f} CI={r['ci']:.4f} "
              f"nn-Tanimoto median={r['nearest_train_tanimoto_median']:.3f}")
    return pd.DataFrame(rows), analogue


def main() -> int:
    paths.ensure_dirs()
    pairs, clean, X = kiba()
    print(f"KIBA {len(pairs)} pairs, features {X.shape}")

    b1 = task_b1(pairs, X)
    b3, analogue = task_b3(pairs, clean, X)
    frame = pd.concat([b1, b3], ignore_index=True)
    frame.to_csv(paths.RESULTS / "binding_followups.csv", index=False)

    b1_agg = b1.groupby(["split", "max_features"])[
        ["pearson_r", "ci", "within_protein_pearson"]].agg(["mean", "std"]).round(4)
    print("\nB1 - max_features sweep\n" + b1_agg.to_string())
    b3_agg = b3[["pearson_r", "ci", "within_protein_pearson",
                 "nearest_train_tanimoto_median"]].agg(["mean", "std"]).round(4)
    print("\nB3 - scaffold-cold-drug\n" + b3_agg.to_string())

    committed = pd.read_csv(paths.RESULTS / "binding_baseline.csv")
    committed = committed[committed["dataset"] == "KIBA"].groupby("split")["pearson_r"].mean()

    stats = {
        "B1": {
            "n_estimators": N_ESTIMATORS,
            "committed_setting": "sqrt",
            "by_split": {
                split: {
                    str(mf): {
                        "pearson_mean": float(g["pearson_r"].mean()),
                        "pearson_std": float(g["pearson_r"].std(ddof=1)),
                        "ci_mean": float(g["ci"].mean()),
                        "within_protein_mean": float(g["within_protein_pearson"].mean()),
                    }
                    for mf, g in b1[b1["split"] == split].groupby("max_features")
                }
                for split in B1_SPLITS
            },
        },
        "B3": {
            "pearson_mean": float(b3["pearson_r"].mean()),
            "pearson_std": float(b3["pearson_r"].std(ddof=1)),
            "ci_mean": float(b3["ci"].mean()),
            "within_protein_mean": float(b3["within_protein_pearson"].mean()),
            "n_test_drugs_mean": float(b3["n_test_drugs"].mean()),
            "analogue_distance": analogue,
            "committed_cold_drug_pearson": float(committed.get("cold_drug", np.nan)),
            "committed_random_pearson": float(committed.get("random", np.nan)),
        },
    }
    (paths.RESULTS / "binding_followups.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {paths.RESULTS / 'binding_followups.csv'} and "
          f"{paths.RESULTS / 'binding_followups.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
