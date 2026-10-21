"""Phase 4 follow-up - why does the forest hold up on cold-drug when the MLP does not?

B2 left an open question rather than closing one. On the *identical* KIBA
cold-drug split the frozen-embedding MLP scores Pearson 0.582 (matching the
published 0.576) while our random forest scores 0.7324. Analogue density cannot
explain a gap between two models that see the same analogues.

The obvious suspect is that pooled Pearson rewards predicting *which protein is
easy to bind* rather than which molecule binds it. This script measures that
directly by scoring two trivial predictors that contain no molecular information
at all:

    global mean   - predict the training mean affinity for everything
    protein mean  - predict each protein's training mean affinity

If the protein-mean predictor alone reaches a high pooled Pearson, then any
pooled number on this split is mostly protein-level signal, and the forest's
apparent advantage has to be re-read in that light.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import datasets, dti, paths  # noqa: E402

SPLITS = ("random", "cold_drug")
MIN_PAIRS_PER_PROTEIN = 10


def metrics(y: np.ndarray, pred: np.ndarray, targets: np.ndarray) -> dict:
    """Pooled and within-protein scores for one predictor."""
    out = {
        "pooled_pearson": float(pearsonr(y, pred)[0]) if np.std(pred) > 0 else 0.0,
        "pooled_spearman": float(spearmanr(y, pred)[0]) if np.std(pred) > 0 else 0.0,
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "ci": float(dti.concordance_index(y, pred)),
    }
    rs = []
    for target in np.unique(targets):
        sel = targets == target
        if sel.sum() < MIN_PAIRS_PER_PROTEIN or np.unique(y[sel]).size < 2:
            continue
        # A constant prediction inside a protein has no within-protein signal;
        # score it as 0 rather than letting pearsonr return NaN.
        rs.append(0.0 if np.std(pred[sel]) == 0
                  else float(pearsonr(y[sel], pred[sel])[0]))
    out["within_protein_pearson"] = float(np.mean(rs)) if rs else float("nan")
    out["n_proteins_scored"] = len(rs)
    return out


def main() -> int:
    paths.ensure_dirs()
    raw = datasets.load_dti("KIBA")
    clean, _ = datasets.unique_molecules(raw)
    pairs, _ = dti.build_pairs(raw, clean)
    X = dti.feature_matrix(pairs, dict(zip(clean["inchikey"], clean["canonical_smiles"])))
    y = pairs["affinity"].to_numpy(dtype=float)
    targets = pairs["target_id"].to_numpy()

    rows = []
    for split in SPLITS:
        for seed in paths.SEEDS:
            frame = pd.read_csv(paths.SPLITS / f"kiba_{split}_seed{seed}.csv")
            subsets = frame.sort_values("pair_index")["subset"].to_numpy()
            tr, te = subsets == "train", subsets == "test"

            global_mean = float(y[tr].mean())
            per_protein = pd.Series(y[tr]).groupby(targets[tr]).mean()
            protein_mean_pred = np.array(
                [per_protein.get(t, global_mean) for t in targets[te]]
            )

            model = RandomForestRegressor(
                n_estimators=100, max_features="sqrt", random_state=seed, n_jobs=-1
            ).fit(X[tr], y[tr])
            rf_pred = model.predict(X[te])

            for name, pred in (
                ("global_mean", np.full(te.sum(), global_mean)),
                ("protein_mean", protein_mean_pred),
                ("random_forest", rf_pred),
            ):
                rows.append({"split": split, "seed": seed, "predictor": name,
                             "n_test": int(te.sum()),
                             **metrics(y[te], pred, targets[te])})
            last = rows[-1]
            pm = rows[-2]
            print(f"  {split:10s} seed={seed}  protein-mean pooled r="
                  f"{pm['pooled_pearson']:.4f}  RF pooled r={last['pooled_pearson']:.4f}  "
                  f"RF within-protein r={last['within_protein_pearson']:.4f}")

    frame = pd.DataFrame(rows)
    frame.to_csv(paths.RESULTS / "cold_drug_diagnosis.csv", index=False)
    agg = frame.groupby(["split", "predictor"])[
        ["pooled_pearson", "within_protein_pearson", "ci", "rmse"]
    ].mean().round(4)
    print("\n" + agg.to_string())

    frozen = pd.read_csv(paths.RESULTS / "frozen_baseline_metrics.csv")
    frozen_agg = frozen.groupby("split")[["pearson_r", "within_protein_pearson"]].mean()

    stats = {
        "by_split": {
            split: {
                predictor: {
                    "pooled_pearson": float(g["pooled_pearson"].mean()),
                    "within_protein_pearson": float(g["within_protein_pearson"].mean()),
                    "ci": float(g["ci"].mean()),
                    "rmse": float(g["rmse"].mean()),
                }
                for predictor, g in frame[frame["split"] == split].groupby("predictor")
            }
            for split in SPLITS
        },
        "frozen_mlp": {
            split: {
                "pooled_pearson": float(frozen_agg.loc[split, "pearson_r"]),
                "within_protein_pearson": float(
                    frozen_agg.loc[split, "within_protein_pearson"]),
            }
            for split in SPLITS if split in frozen_agg.index
        },
        "min_pairs_per_protein": MIN_PAIRS_PER_PROTEIN,
    }
    (paths.RESULTS / "cold_drug_diagnosis.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {paths.RESULTS / 'cold_drug_diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
