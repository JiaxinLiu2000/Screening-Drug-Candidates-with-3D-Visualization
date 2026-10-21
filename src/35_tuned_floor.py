"""Re-measure every neural-vs-baseline comparison against the TUNED forest.

Task B1 found that the committed affinity baseline was under-tuned:
`max_features='sqrt'` lets each split of a 2,048-bit fingerprint consider ~45
bits, and raising it to 0.3 is worth +0.038 Pearson on KIBA-random. Every
"neural beats the floor" statement in the project was made against the weaker
floor, which inflates the neural advantage.

This script re-runs the two places that comparison appears, at the tuned
setting, so the documents can quote a floor that was actually tuned:

    1. the 30-fit binding grid (2 datasets x 3 splits x 5 seeds)
    2. the KIBA arm of the learning curve (5 fractions x 3 seeds)

The committed `binding_baseline.csv` is NOT overwritten - it remains the record
of what the untuned baseline did, and the tuned numbers land beside it so the
difference stays visible rather than being quietly absorbed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import datasets, dti, paths  # noqa: E402

TUNED_MAX_FEATURES = 0.3
COMMITTED_MAX_FEATURES = "sqrt"
N_ESTIMATORS = 100
SPLITS = ("random", "cold_drug", "cold_target")
CURVE_FRACTIONS = (0.05, 0.1, 0.25, 0.5, 1.0)
CURVE_SEEDS = (0, 1, 2)


def load_dataset(name: str):
    raw = datasets.load_dti(name)
    clean, _ = datasets.unique_molecules(raw)
    pairs, _ = dti.build_pairs(raw, clean)
    if name == "DAVIS":
        pairs["affinity"] = dti.kd_nm_to_pkd(pairs["affinity"].to_numpy())
    X = dti.feature_matrix(pairs, dict(zip(clean["inchikey"], clean["canonical_smiles"])))
    return pairs, X, pairs["affinity"].to_numpy(dtype=float)


def fit_score(X, y, tr, te, seed, pairs) -> dict:
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                  max_features=TUNED_MAX_FEATURES,
                                  random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
    pred = model.predict(X[te])
    df = pairs[te].copy()
    df["y"], df["pred"] = y[te], pred
    within = [pearsonr(g["y"], g["pred"])[0] for _, g in df.groupby("target_id")
              if len(g) >= 10 and g["y"].nunique() > 1]
    return {
        "rmse": float(np.sqrt(np.mean((pred - y[te]) ** 2))),
        "pearson_r": float(pearsonr(y[te], pred)[0]),
        "spearman_rho": float(spearmanr(y[te], pred)[0]),
        "ci": float(dti.concordance_index(y[te], pred)),
        "within_protein_pearson": float(np.mean(within)) if within else float("nan"),
        "n_test": int(te.sum()),
    }


def subsample_train(subsets: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Identical rule to src/32_learning_curves.py, so the arms stay comparable."""
    out = subsets.copy()
    if fraction >= 1.0:
        return out
    train_idx = np.flatnonzero(subsets == "train")
    rng = np.random.default_rng(10_000 + seed)
    keep = rng.choice(train_idx, size=max(1, int(round(len(train_idx) * fraction))),
                      replace=False)
    out[np.setdiff1d(train_idx, keep)] = "unused"
    return out


def main() -> int:
    paths.ensure_dirs()
    t0 = time.time()
    rows, curve_rows = [], []

    for name in ("KIBA", "DAVIS"):
        pairs, X, y = load_dataset(name)
        print(f"{name}: {len(pairs)} pairs, features {X.shape}")
        for split in SPLITS:
            for seed in paths.SEEDS:
                frame = pd.read_csv(paths.SPLITS / f"{name.lower()}_{split}_seed{seed}.csv")
                subsets = frame.sort_values("pair_index")["subset"].to_numpy()
                res = fit_score(X, y, subsets == "train", subsets == "test", seed, pairs)
                rows.append({"dataset": name, "split": split, "seed": seed,
                             "max_features": str(TUNED_MAX_FEATURES), **res})
            last = [r for r in rows if r["dataset"] == name and r["split"] == split]
            print(f"  {split:12s} Pearson "
                  f"{np.mean([r['pearson_r'] for r in last]):.4f}  "
                  f"CI {np.mean([r['ci'] for r in last]):.4f}  "
                  f"[{time.time() - t0:.0f}s]")

        if name == "KIBA":
            for seed in CURVE_SEEDS:
                base = pd.read_csv(paths.SPLITS / f"kiba_random_seed{seed}.csv")
                base = base.sort_values("pair_index")["subset"].to_numpy()
                for frac in CURVE_FRACTIONS:
                    subsets = subsample_train(base, frac, seed)
                    res = fit_score(X, y, subsets == "train", subsets == "test",
                                    seed, pairs)
                    curve_rows.append({
                        "task": "kiba_random", "model": "random_forest_tuned",
                        "seed": seed, "fraction": frac,
                        "n_train": int((subsets == "train").sum()),
                        "metric": "pearson_r", "value": res["pearson_r"],
                        "secondary": res["ci"]})
                    print(f"  curve seed={seed} frac={frac:<5} "
                          f"n={curve_rows[-1]['n_train']:6d} "
                          f"Pearson {res['pearson_r']:.4f}  [{time.time() - t0:.0f}s]")

    grid = pd.DataFrame(rows)
    grid.to_csv(paths.RESULTS / "binding_baseline_tuned.csv", index=False)
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(paths.RESULTS / "learning_curve_tuned_rf.csv", index=False)

    committed = pd.read_csv(paths.RESULTS / "binding_baseline.csv")
    frozen = pd.read_csv(paths.RESULTS / "frozen_baseline_metrics.csv")
    neural = pd.read_csv(paths.RESULTS / "shared_encoder_metrics.csv")
    neural = neural[(neural["loss_weight"].isna()) | (neural["loss_weight"] == 1.0)]

    comparison = {}
    for split in SPLITS:
        old = committed[(committed["dataset"] == "KIBA")
                        & (committed["split"] == split)]["pearson_r"].mean()
        new = grid[(grid["dataset"] == "KIBA")
                   & (grid["split"] == split)]["pearson_r"].mean()
        mlp = frozen[frozen["split"] == split]["pearson_r"].mean()
        enc = neural[neural["affinity_split"] == split]["affinity_pearson_r"].mean()
        comparison[split] = {
            "rf_untuned_sqrt": float(old),
            "rf_tuned_0.3": float(new),
            "tuning_gain": float(new - old),
            "frozen_mlp": float(mlp),
            "best_trained_encoder": float(enc),
            "mlp_lead_over_untuned_rf": float(mlp - old),
            "mlp_lead_over_tuned_rf": float(mlp - new),
        }
        print(f"\n{split}: RF {old:.4f} (sqrt) -> {new:.4f} (0.3); "
              f"MLP lead shrinks {mlp - old:+.4f} -> {mlp - new:+.4f}")

    curve_agg = (curve.groupby("fraction")["value"].agg(["mean", "std"]).reset_index())
    old_curve = pd.read_csv(paths.RESULTS / "learning_curves.csv")
    old_curve = old_curve[(old_curve["task"] == "kiba_random")]
    curve_cmp = {}
    for frac in CURVE_FRACTIONS:
        tuned = float(curve_agg[np.isclose(curve_agg["fraction"], frac)]["mean"].iloc[0])
        untuned = float(old_curve[(old_curve["model"] == "random_forest")
                                  & np.isclose(old_curve["fraction"], frac)]["value"].mean())
        mlp = float(old_curve[(old_curve["model"] == "frozen_esm_mlp")
                              & np.isclose(old_curve["fraction"], frac)]["value"].mean())
        curve_cmp[str(frac)] = {
            "rf_untuned": untuned, "rf_tuned": tuned, "frozen_mlp": mlp,
            "mlp_lead_over_untuned": mlp - untuned,
            "mlp_lead_over_tuned": mlp - tuned,
        }
    print("\nlearning curve, MLP lead against each floor:")
    for frac, v in curve_cmp.items():
        print(f"  frac={frac:<5} untuned {v['mlp_lead_over_untuned']:+.4f}  "
              f"tuned {v['mlp_lead_over_tuned']:+.4f}")

    stats = {
        "tuned_max_features": TUNED_MAX_FEATURES,
        "committed_max_features": COMMITTED_MAX_FEATURES,
        "n_estimators": N_ESTIMATORS,
        "kiba_by_split": comparison,
        "davis_by_split": {
            split: {
                "rf_untuned_sqrt": float(committed[(committed["dataset"] == "DAVIS")
                                                   & (committed["split"] == split)][
                                             "pearson_r"].mean()),
                "rf_tuned_0.3": float(grid[(grid["dataset"] == "DAVIS")
                                           & (grid["split"] == split)]["pearson_r"].mean()),
            }
            for split in SPLITS
        },
        "learning_curve_kiba": curve_cmp,
        "note": ("binding_baseline.csv is left untouched as the record of the untuned "
                 "baseline; these rows are the tuned re-measurement"),
    }
    (paths.RESULTS / "tuned_floor.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote binding_baseline_tuned.csv, learning_curve_tuned_rf.csv "
          f"and tuned_floor.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
