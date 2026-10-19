"""Task 4 - random forest baseline on Morgan fingerprints, 12 endpoints.

Grid: {random, scaffold} x seeds {0..4} x 12 endpoints = 120 independent models.
The seed controls both the split (loaded from splits/) and the forest.
Outputs results/baseline_metrics.csv and results/baseline_summary.csv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import featurize, paths  # noqa: E402

N_ESTIMATORS = 500


def load_split(kind: str, seed: int, n_rows: int, inchikeys: pd.Series) -> np.ndarray:
    path = paths.SPLITS / f"{kind}_split_seed{seed}.csv"
    frame = pd.read_csv(path)
    if len(frame) != n_rows or not (frame["inchikey"].to_numpy() == inchikeys.to_numpy()).all():
        raise RuntimeError(f"{path} does not match {paths.CLEAN_PARQUET}; rerun src/03_splits.py")
    return frame["subset"].to_numpy()


def main() -> int:
    paths.ensure_dirs()
    if not paths.CLEAN_PARQUET.exists():
        print(f"FAILED: {paths.CLEAN_PARQUET} missing - run src/02_standardize.py first")
        return 1

    clean = pd.read_parquet(paths.CLEAN_PARQUET)
    endpoints = [
        c
        for c in clean.columns
        if c not in ("inchikey", "canonical_smiles", "n_heavy_atoms", "drug_ids", "n_source_rows")
    ]
    print(f"{len(clean)} molecules, {len(endpoints)} endpoints")

    t0 = time.time()
    X = featurize.featurize_smiles(clean["canonical_smiles"])
    print(f"Morgan r={featurize.MORGAN_RADIUS} {featurize.MORGAN_N_BITS} bits -> "
          f"{X.shape} in {time.time() - t0:.1f}s "
          f"(mean bits on = {X.sum(axis=1).mean():.1f})")

    records: list[dict] = []
    for kind in ("random", "scaffold"):
        for seed in paths.SEEDS:
            subsets = load_split(kind, seed, len(clean), clean["inchikey"])
            is_train = subsets == "train"
            is_test = subsets == "test"
            for endpoint in endpoints:
                y = clean[endpoint].to_numpy(dtype=float)
                labeled = ~np.isnan(y)
                tr = is_train & labeled
                te = is_test & labeled
                y_tr, y_te = y[tr].astype(int), y[te].astype(int)

                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    raise RuntimeError(
                        f"{kind}/seed{seed}/{endpoint}: single-class train or test fold "
                        f"(train={np.bincount(y_tr).tolist()}, test={np.bincount(y_te).tolist()})"
                    )

                clf = RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=seed,
                )
                clf.fit(X[tr], y_tr)
                scores = clf.predict_proba(X[te])[:, 1]
                records.append(
                    {
                        "split": kind,
                        "seed": seed,
                        "endpoint": endpoint,
                        "n_train": int(tr.sum()),
                        "n_train_pos": int(y_tr.sum()),
                        "n_test": int(te.sum()),
                        "n_test_pos": int(y_te.sum()),
                        "roc_auc": roc_auc_score(y_te, scores),
                        "pr_auc": average_precision_score(y_te, scores),
                    }
                )
            done = records[-len(endpoints):]
            print(f"  {kind:<8} seed={seed}  "
                  f"mean ROC-AUC={np.mean([r['roc_auc'] for r in done]):.4f}  "
                  f"mean PR-AUC={np.mean([r['pr_auc'] for r in done]):.4f}  "
                  f"[{time.time() - t0:.0f}s]")

    metrics = pd.DataFrame(records)
    metrics.to_csv(paths.RESULTS / "baseline_metrics.csv", index=False)

    per_endpoint = (
        metrics.groupby(["split", "endpoint"])[["roc_auc", "pr_auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_endpoint.columns = [
        "split", "endpoint", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std"
    ]
    per_endpoint.insert(2, "level", "endpoint")

    # Macro average over the 12 endpoints within each seed, then mean/std over seeds.
    per_seed = metrics.groupby(["split", "seed"])[["roc_auc", "pr_auc"]].mean().reset_index()
    overall = per_seed.groupby("split")[["roc_auc", "pr_auc"]].agg(["mean", "std"]).reset_index()
    overall.columns = ["split", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std"]
    overall.insert(1, "endpoint", "ALL_12_MACRO_AVG")
    overall.insert(2, "level", "overall")

    summary = pd.concat([per_endpoint, overall], ignore_index=True)
    summary.to_csv(paths.RESULTS / "baseline_summary.csv", index=False)

    print("\n--- per-endpoint ROC-AUC (mean +/- std over 5 seeds) ---")
    pivot = per_endpoint.pivot(index="endpoint", columns="split",
                               values=["roc_auc_mean", "roc_auc_std"])
    print(pivot.round(4).to_string())
    print("\n--- macro average over the 12 endpoints ---")
    print(overall.round(4).to_string(index=False))
    print(f"\nWrote {paths.RESULTS / 'baseline_metrics.csv'} ({len(metrics)} rows)")
    print(f"Wrote {paths.RESULTS / 'baseline_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
