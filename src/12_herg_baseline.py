"""Phase 2 Task 4 - hERG (TDC) as an independent toxicity check.

Same standardizer, same splitters, same RF baseline as the Tox21 pipeline.
Outputs: results/herg_stats.json, splits/herg_{random,scaffold}_split_seed{N}.csv,
hERG rows appended (idempotently) to results/baseline_metrics.csv and
results/baseline_summary.csv, and results/herg_baseline.md.

``--no-append`` computes everything but never touches the shared CSVs or the
markdown (used while the Tox21 grid is being regenerated).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import baseline, datasets, featurize, paths  # noqa: E402
from tox21lib import splits as splitlib  # noqa: E402

DATASET = "hERG"
LABEL = "hERG"
KINDS = ("random", "scaffold")


def split_path(kind: str, seed: int) -> Path:
    return paths.SPLITS / f"herg_{kind}_split_seed{seed}.csv"


def bitvects(X: np.ndarray) -> list:
    out = []
    for row in X:
        bv = DataStructs.ExplicitBitVect(int(X.shape[1]))
        bv.SetBitsFromList(np.flatnonzero(row).tolist())
        out.append(bv)
    return out


def nn_similarity(fps: list, subsets: np.ndarray) -> float:
    """Mean over test molecules of the max Tanimoto to any train molecule."""
    train = [fps[i] for i in np.flatnonzero(subsets == "train")]
    test = [fps[i] for i in np.flatnonzero(subsets == "test")]
    return float(np.mean([max(DataStructs.BulkTanimotoSimilarity(t, train)) for t in test]))


def leakage_table(clean: pd.DataFrame, scaffolds: pd.Series, prefix: str) -> pd.DataFrame:
    rows = []
    for kind in KINDS:
        for seed in paths.SEEDS:
            frame = splitlib.split_frame(clean, kind, seed, scaffolds)
            if prefix:
                frame.to_csv(split_path(kind, seed), index=False, lineterminator="\n")
            counts = frame["subset"].value_counts()
            per = frame.groupby("scaffold_smiles")["subset"].nunique()
            test = frame["subset"] == "test"
            rows.append({
                "split": kind, "seed": seed,
                "train": int(counts.get("train", 0)),
                "valid": int(counts.get("valid", 0)),
                "test": int(counts.get("test", 0)),
                "scaffold_groups_spanning_subsets": int((per > 1).sum()),
                "test_pos_rate": float(clean.loc[test.to_numpy(), LABEL].mean())
                if LABEL in clean else float("nan"),
            })
    return pd.DataFrame(rows)


def check_shared(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "dataset" not in df.columns or not (df["dataset"] == "Tox21").any():
        raise RuntimeError(
            f"{path} has no 'dataset' column / Tox21 rows - rerun src/04_baseline_rf.py first"
        )
    return df


def append_rows(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Replace this dataset's rows in a shared table, preserving other datasets.

    The shared tables gained a ``setting`` column in Phase 3 (Tox21 and this
    588-molecule hERG are ``standard``; hERG-large is ``full``/``disjoint``), and
    they now also carry hERG-large rows. This function therefore adapts to the
    existing schema instead of demanding an exact column match, and preserves
    every dataset other than its own.
    """
    old = check_shared(path)
    if "setting" in old.columns and "setting" not in new.columns:
        new = new.copy()
        new.insert(1, "setting", "standard")
    missing = set(old.columns) - set(new.columns)
    extra = set(new.columns) - set(old.columns)
    if missing or extra:
        raise RuntimeError(
            f"{path} schema mismatch: missing from new rows {sorted(missing)}, "
            f"unexpected in new rows {sorted(extra)}"
        )
    new = new[list(old.columns)]
    kept = old[old["dataset"] != DATASET]
    out = pd.concat([kept, new], ignore_index=True)
    # Keep a deterministic dataset order so the file is byte-stable.
    order = {name: i for i, name in enumerate(["Tox21", DATASET])}
    out["_order"] = out["dataset"].map(lambda d: order.get(d, len(order)))
    out = out.sort_values(["_order"], kind="stable").drop(columns="_order")
    out.to_csv(path, index=False, lineterminator="\n")
    return out


def main() -> int:
    no_append = "--no-append" in sys.argv[1:]
    paths.ensure_dirs()

    # 1. load + standardize
    raw = datasets.load_tox_single("hERG").rename(columns={"label": LABEL})
    # 22 TDC hERG rows have no Drug_ID; give them a stable row-based id so the
    # standardizer's drug_ids join works (ids are provenance only, never keys).
    missing_id = raw["drug_id"].isna()
    raw.loc[missing_id, "drug_id"] = [f"hERG_row{i}" for i in np.flatnonzero(missing_id)]
    clean, std_stats = datasets.standardize_molecules(raw, label_cols=[LABEL])
    labeled = clean[LABEL].dropna()
    tox21 = pd.read_parquet(paths.CLEAN_PARQUET)
    overlap = int(clean["inchikey"].isin(set(tox21["inchikey"])).sum())
    stats = {
        "rows_missing_drug_id": int(missing_id.sum()),
        **std_stats,
        "n_labeled": int(labeled.size),
        "n_positive": int(labeled.sum()),
        "positive_rate": float(labeled.mean()),
        "tox21_molecules": len(tox21),
        "inchikey_overlap_with_tox21": overlap,
        "inchikey_overlap_pct_of_herg": 100.0 * overlap / len(clean),
    }

    # 2. splits + leakage
    scaffolds = pd.Series([splitlib.scaffold_smiles(s) for s in clean["canonical_smiles"]],
                          name="scaffold_smiles")
    stats["scaffolds"] = splitlib.scaffold_stats(scaffolds)
    sizes = leakage_table(clean, scaffolds, prefix="herg")
    print(json.dumps(stats, indent=2))
    print(sizes.to_string(index=False))
    if (sizes.loc[sizes["split"] == "scaffold", "scaffold_groups_spanning_subsets"] != 0).any():
        print("FAILED: a hERG scaffold group was split across subsets")
        return 1
    if (sizes.loc[sizes["split"] == "random", "scaffold_groups_spanning_subsets"] == 0).all():
        print("FAILED: hERG random split reports zero scaffold leakage")
        return 1

    # Diagnostics for the drop comparison: test->train nearest-neighbour similarity,
    # computed identically for hERG and Tox21 on their committed splits.
    X = featurize.featurize_smiles(clean["canonical_smiles"])
    fps = bitvects(X)
    tox_fps = bitvects(featurize.featurize_smiles(tox21["canonical_smiles"]))
    tox_scaf = pd.read_csv(paths.SPLITS / "scaffolds.csv")["scaffold_smiles"].fillna("")
    nn_rows = []
    for kind in KINDS:
        for seed in paths.SEEDS:
            h = pd.read_csv(split_path(kind, seed))["subset"].to_numpy()
            t = pd.read_csv(paths.SPLITS / f"{kind}_split_seed{seed}.csv")["subset"].to_numpy()
            nn_rows.append({"split": kind, "seed": seed,
                            "herg_nn_tanimoto": nn_similarity(fps, h),
                            "tox21_nn_tanimoto": nn_similarity(tox_fps, t)})
    nn = pd.DataFrame(nn_rows)
    nn_mean = nn.groupby("split")[["herg_nn_tanimoto", "tox21_nn_tanimoto"]].mean()
    tox_sstats = splitlib.scaffold_stats(tox_scaf)
    stats["split_sizes"] = sizes.to_dict(orient="records")
    stats["test_to_train_nn_tanimoto_mean"] = nn_mean.round(6).to_dict(orient="index")
    (paths.RESULTS / "herg_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(nn_mean.round(4).to_string())

    # 3. baseline
    def subsets_for(kind: str, seed: int) -> np.ndarray:
        frame = pd.read_csv(split_path(kind, seed))
        if not (frame["inchikey"].to_numpy() == clean["inchikey"].to_numpy()).all():
            raise RuntimeError(f"{split_path(kind, seed)} does not match hERG clean set")
        return frame["subset"].to_numpy()

    metrics = baseline.run_grid(DATASET, X, clean, [LABEL], KINDS, paths.SEEDS, subsets_for)
    summary = baseline.summarize(metrics)
    print(summary.round(4).to_string(index=False))
    if no_append:
        tmp = paths.RESULTS / "herg_metrics_tmp.csv"
        metrics.to_csv(tmp, index=False, lineterminator="\n")
        print(f"--no-append: wrote {tmp}; shared CSVs untouched")
        return 0

    all_metrics = append_rows(paths.RESULTS / "baseline_metrics.csv", metrics)
    all_summary = append_rows(paths.RESULTS / "baseline_summary.csv", summary)
    (paths.RESULTS / "herg_metrics_tmp.csv").unlink(missing_ok=True)
    print("rows per dataset:", all_metrics["dataset"].value_counts().sort_index().to_dict())

    # 4. report
    ov = all_summary[all_summary["level"] == "overall"].set_index(["dataset", "split"])
    h_drop = ov.loc[("hERG", "random"), "roc_auc_mean"] - ov.loc[("hERG", "scaffold"), "roc_auc_mean"]
    t_drop = ov.loc[("Tox21", "random"), "roc_auc_mean"] - ov.loc[("Tox21", "scaffold"), "roc_auc_mean"]
    ratio = h_drop / t_drop
    hs = stats["scaffolds"]
    hs_ratio = hs["n_unique_scaffolds"] / len(clean)
    tox_ratio = tox_sstats["n_unique_scaffolds"] / len(tox21)
    h_std = float(ov.loc[("hERG", "random"), "roc_auc_std"])
    n_test = int(metrics["n_test"].min()), int(metrics["n_test"].max())
    sim_gap = {c: nn_mean.loc["random", c] - nn_mean.loc["scaffold", c]
               for c in nn_mean.columns}
    context = (
        f"Measured reasons: (i) hERG's test sets hold only {n_test[0]}-{n_test[1]} molecules, "
        f"and the seed-to-seed std of hERG random ROC-AUC ({h_std:.4f}) is "
        f"{h_std / abs(h_drop):.1f}x the absolute random-scaffold gap, so the gap is "
        f"within seed noise; (ii) hERG is far more scaffold-diverse "
        f"({hs_ratio:.4f} unique scaffolds per molecule vs "
        f"{tox_ratio:.4f} for Tox21), so a random split already "
        f"separates test from train chemistry: the random-vs-scaffold gap in mean "
        f"test-to-train nearest-neighbour Tanimoto is {sim_gap['herg_nn_tanimoto']:.4f} for "
        f"hERG vs {sim_gap['tox21_nn_tanimoto']:.4f} for Tox21. The hERG positive rate is "
        f"{stats['positive_rate']:.4f}.")
    if 0.5 <= ratio <= 2.0:
        verdict = (f"**Corroborates.** The hERG drop ({h_drop:.4f}) is of similar magnitude to "
                   f"Tox21's ({t_drop:.4f}); ratio {ratio:.2f}.")
    else:
        verdict = (f"**Diverges.** The hERG random-to-scaffold drop ({h_drop:.4f}) does not "
                   f"reproduce Tox21's {t_drop:.4f} (ratio {ratio:.2f}); hERG therefore does "
                   f"not corroborate the leakage finding. " + context)
    lines = [
        "# hERG independent toxicity check", "",
        "Same standardizer, splitters (80/10/10, seeds 0-4) and RF baseline "
        "(Morgan r=2, 2048 bits, 500 trees, balanced) as Tox21. Generated by "
        "`src/12_herg_baseline.py`.", "",
        "## Dataset", "",
        "| quantity | value |", "|---|---|",
        f"| input rows (TDC hERG) | {stats['input_rows']} |",
        f"| parse failures | {stats['parse_failures']} |",
        f"| duplicate rows merged | {stats['duplicate_rows_merged']} |",
        f"| label ties set to NaN | {stats['label_cells_nan_from_tie']} |",
        f"| filtered out | {stats['filtered_out']} |",
        f"| final molecules | {stats['final_molecules']} |",
        f"| positives / labeled | {stats['n_positive']} / {stats['n_labeled']} "
        f"({stats['positive_rate']:.4f}) |",
        f"| InChIKey overlap with Tox21 ({len(tox21)}) | {overlap} "
        f"({stats['inchikey_overlap_pct_of_herg']:.2f}% of hERG) |", "",
        "## Scaffold diversity", "",
        "| dataset | molecules | unique scaffolds | scaffolds / molecule | singletons | largest group |",
        "|---|---|---|---|---|---|",
        f"| hERG | {len(clean)} | {hs['n_unique_scaffolds']} | "
        f"{hs['n_unique_scaffolds'] / len(clean):.4f} | {hs['n_singleton_scaffolds']} | "
        f"{hs['largest_group_size']} |",
        f"| Tox21 | {len(tox21)} | {tox_sstats['n_unique_scaffolds']} | "
        f"{tox_sstats['n_unique_scaffolds'] / len(tox21):.4f} | "
        f"{tox_sstats['n_singleton_scaffolds']} | {tox_sstats['largest_group_size']} |", "",
        "## Splits", "",
        "| split | seed | train | valid | test | test pos. rate | scaffold groups spanning subsets |",
        "|---|---|---|---|---|---|---|",
        *[f"| {r.split} | {r.seed} | {r.train} | {r.valid} | {r.test} | {r.test_pos_rate:.4f} | "
          f"{r.scaffold_groups_spanning_subsets} |" for r in sizes.itertuples()], "",
        "## Baseline (mean +/- sample std over 5 seeds)", "",
        "| dataset | split | ROC-AUC | PR-AUC | mean test->train NN Tanimoto |",
        "|---|---|---|---|---|",
    ]
    for ds, col in (("hERG", "herg_nn_tanimoto"), ("Tox21", "tox21_nn_tanimoto")):
        for kind in KINDS:
            r = ov.loc[(ds, kind)]
            lines.append(f"| {ds} | {kind} | {r.roc_auc_mean:.4f} +/- {r.roc_auc_std:.4f} | "
                         f"{r.pr_auc_mean:.4f} +/- {r.pr_auc_std:.4f} | "
                         f"{nn_mean.loc[kind, col]:.4f} |")
    lines += ["", "Tox21 rows are the 12-endpoint macro average.", "",
              "## Random -> scaffold ROC-AUC drop", "",
              f"- hERG: {h_drop:.4f}", f"- Tox21: {t_drop:.4f}", "", verdict, ""]
    (paths.RESULTS / "herg_baseline.md").write_text("\n".join(lines), encoding="utf-8")
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
