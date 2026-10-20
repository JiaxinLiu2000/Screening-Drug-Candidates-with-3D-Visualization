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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import baseline, featurize, paths  # noqa: E402


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

    metrics = baseline.run_grid(
        dataset="Tox21",
        X=X,
        labels=clean,
        endpoints=endpoints,
        splits=("random", "scaffold"),
        seeds=paths.SEEDS,
        subsets_for=lambda kind, seed: load_split(
            kind, seed, len(clean), clean["inchikey"]
        ),
    )
    print(f"  [{time.time() - t0:.0f}s for {len(metrics)} fits]")

    # The shared tables are keyed by (dataset, setting); Phase 3 added hERG-large
    # rows with settings `full`/`disjoint`. This script owns only the Tox21 rows,
    # so it writes them with `standard` and the later scripts append their own.
    metrics.insert(1, "setting", "standard")
    metrics.to_csv(paths.RESULTS / "baseline_metrics.csv", index=False)

    summary = baseline.summarize(metrics.drop(columns="setting"))
    summary.insert(1, "setting", "standard")
    summary.to_csv(paths.RESULTS / "baseline_summary.csv", index=False)

    per_endpoint = summary[summary["level"] == "endpoint"]
    overall = summary[summary["level"] == "overall"]
    print("\n--- per-endpoint ROC-AUC (mean +/- std over 5 seeds) ---")
    print(per_endpoint.pivot(index="endpoint", columns="split",
                             values=["roc_auc_mean", "roc_auc_std"]).round(4).to_string())
    print(f"\n--- macro average over the {len(endpoints)} endpoints ---")
    print(overall.round(4).to_string(index=False))
    print(f"\nWrote {paths.RESULTS / 'baseline_metrics.csv'} ({len(metrics)} rows)")
    print(f"Wrote {paths.RESULTS / 'baseline_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
