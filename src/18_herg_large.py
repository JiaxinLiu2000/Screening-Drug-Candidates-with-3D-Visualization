"""Large hERG (TDC ``herg_karim``) external check, Full and Tox21-disjoint settings.

Same standardizer, splitters and RF baseline as Tox21. Appends rows to the shared
baseline CSVs with a ``setting`` column (``standard`` for pre-existing rows,
``full`` / ``disjoint`` for hERG-large). Re-running replaces hERG-large rows.
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

TDC_NAME = "herg_karim"
DATASET = "hERG-large"
LABEL = "hERG"
KINDS = ("random", "scaffold")
SETTINGS = ("full", "disjoint")
STANDARD = "standard"


def split_path(setting: str, kind: str, seed: int) -> Path:
    return paths.SPLITS / f"herg_large_{setting}_{kind}_split_seed{seed}.csv"


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


def build_splits(setting: str, clean: pd.DataFrame, scaffolds: pd.Series) -> pd.DataFrame:
    rows = []
    for kind in KINDS:
        for seed in paths.SEEDS:
            frame = splitlib.split_frame(clean, kind, seed, scaffolds)
            frame.to_csv(split_path(setting, kind, seed), index=False, lineterminator="\n")
            counts = frame["subset"].value_counts()
            per = frame.groupby("scaffold_smiles")["subset"].nunique()
            test = (frame["subset"] == "test").to_numpy()
            rows.append({
                "setting": setting, "split": kind, "seed": seed,
                "train": int(counts.get("train", 0)),
                "valid": int(counts.get("valid", 0)),
                "test": int(counts.get("test", 0)),
                "test_pos_rate": float(clean.loc[test, LABEL].mean()),
                "scaffold_groups_spanning_subsets": int((per > 1).sum()),
            })
    sizes = pd.DataFrame(rows)
    if (sizes.loc[sizes["split"] == "scaffold", "scaffold_groups_spanning_subsets"] != 0).any():
        raise RuntimeError(f"{setting}: a scaffold group was split across subsets")
    if (sizes.loc[sizes["split"] == "random", "scaffold_groups_spanning_subsets"] == 0).any():
        raise RuntimeError(f"{setting}: a random split reports zero scaffold leakage")
    return sizes


def merge_shared(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Add ``setting`` to existing rows (``standard``) and replace hERG-large rows."""
    old = pd.read_csv(path, float_precision="round_trip")
    if "dataset" not in old.columns or not (old["dataset"] == "Tox21").any():
        raise RuntimeError(f"{path} has no Tox21 rows - rerun src/04_baseline_rf.py first")
    if "setting" not in old.columns:
        old.insert(1, "setting", STANDARD)
    kept = old[old["dataset"] != DATASET]
    bad = sorted(set(kept["dataset"]) - {"Tox21", "hERG"})
    if bad or not kept["setting"].eq(STANDARD).all():
        raise RuntimeError(f"{path} contains unexpected rows: datasets {bad}")
    if list(kept.columns) != list(new.columns):
        raise RuntimeError(f"{path} columns {list(kept.columns)} != {list(new.columns)}")
    out = pd.concat([kept, new], ignore_index=True)
    out.to_csv(path, index=False, lineterminator="\n")
    return out


def main() -> int:
    paths.ensure_dirs()

    # 1. load + standardize
    raw = datasets.load_tox_single(TDC_NAME).rename(columns={"label": LABEL})
    missing_id = raw["drug_id"].isna()
    raw["drug_id"] = raw["drug_id"].astype(object)
    raw.loc[missing_id, "drug_id"] = [f"hERGk_row{i}" for i in np.flatnonzero(missing_id)]
    raw["drug_id"] = raw["drug_id"].astype(str)
    clean_full, std_stats = datasets.standardize_molecules(raw, label_cols=[LABEL])
    clean_full = clean_full[clean_full[LABEL].notna()].reset_index(drop=True)
    tox21 = pd.read_parquet(paths.CLEAN_PARQUET)
    in_tox = clean_full["inchikey"].isin(set(tox21["inchikey"]))
    overlap = int(in_tox.sum())
    clean = {"full": clean_full,
             "disjoint": clean_full[~in_tox.to_numpy()].reset_index(drop=True)}
    stats: dict = {
        "tdc_name": TDC_NAME,
        "rows_missing_drug_id": int(missing_id.sum()),
        **std_stats,
        "unlabeled_after_ties_dropped": int(std_stats["final_molecules"] - len(clean_full)),
        "tox21_molecules": len(tox21),
        "inchikey_overlap_with_tox21": overlap,
        "inchikey_overlap_pct_of_herg": 100.0 * overlap / len(clean_full),
        "disjoint_removed": overlap,
        "settings": {},
    }

    # 2. splits + NN diagnostics + 3. baseline
    all_metrics, all_summary, all_sizes = [], [], []
    for setting in SETTINGS:
        df = clean[setting]
        scaffolds = pd.Series([splitlib.scaffold_smiles(s) for s in df["canonical_smiles"]],
                              name="scaffold_smiles")
        sizes = build_splits(setting, df, scaffolds)
        X = featurize.featurize_smiles(df["canonical_smiles"])
        fps = bitvects(X)
        nn = {kind: float(np.mean([nn_similarity(fps, pd.read_csv(
            split_path(setting, kind, s))["subset"].to_numpy()) for s in paths.SEEDS]))
            for kind in KINDS}
        sstats = splitlib.scaffold_stats(scaffolds)
        stats["settings"][setting] = {
            "molecules": len(df),
            "n_positive": int(df[LABEL].sum()),
            "positive_rate": float(df[LABEL].mean()),
            "scaffolds": sstats,
            "scaffolds_per_molecule": sstats["n_unique_scaffolds"] / len(df),
            "test_to_train_nn_tanimoto_mean": nn,
        }
        all_sizes.append(sizes)

        def subsets_for(kind: str, seed: int, setting=setting, df=df) -> np.ndarray:
            frame = pd.read_csv(split_path(setting, kind, seed))
            if not (frame["inchikey"].to_numpy() == df["inchikey"].to_numpy()).all():
                raise RuntimeError(f"{split_path(setting, kind, seed)} mismatches clean set")
            return frame["subset"].to_numpy()

        m = baseline.run_grid(DATASET, X, df, [LABEL], KINDS, paths.SEEDS, subsets_for)
        s = baseline.summarize(m)
        m.insert(1, "setting", setting)
        s.insert(1, "setting", setting)
        all_metrics.append(m)
        all_summary.append(s)

    sizes = pd.concat(all_sizes, ignore_index=True)
    stats["split_sizes"] = sizes.to_dict(orient="records")
    merged_m = merge_shared(paths.RESULTS / "baseline_metrics.csv",
                            pd.concat(all_metrics, ignore_index=True))
    merged_s = merge_shared(paths.RESULTS / "baseline_summary.csv",
                            pd.concat(all_summary, ignore_index=True))
    print("rows per dataset/setting:",
          merged_m.groupby(["dataset", "setting"]).size().to_dict())

    # 4. report
    ov = merged_s[merged_s["level"] == "overall"].set_index(["dataset", "setting", "split"])

    def drop(ds: str, st: str) -> float:
        return float(ov.loc[(ds, st, "random"), "roc_auc_mean"]
                     - ov.loc[(ds, st, "scaffold"), "roc_auc_mean"])

    t_drop = drop("Tox21", STANDARD)
    old_drop = drop("hERG", STANDARD)
    old_stats = json.loads((paths.RESULTS / "herg_stats.json").read_text(encoding="utf-8"))
    old_nn = old_stats["test_to_train_nn_tanimoto_mean"]
    tox_nn_gap = old_nn["random"]["tox21_nn_tanimoto"] - old_nn["scaffold"]["tox21_nn_tanimoto"]
    tox_sstats = splitlib.scaffold_stats(
        pd.read_csv(paths.SPLITS / "scaffolds.csv")["scaffold_smiles"].fillna(""))
    tox_spm = tox_sstats["n_unique_scaffolds"] / len(tox21)

    stats["tox21_drop"] = t_drop
    stats["herg_small_drop"] = old_drop
    stats["tox21_scaffolds_per_molecule"] = tox_spm
    stats["tox21_nn_tanimoto_gap"] = tox_nn_gap
    verdicts = []
    for setting in SETTINGS:
        st = stats["settings"][setting]
        d = drop(DATASET, setting)
        rstd = float(ov.loc[(DATASET, setting, "random"), "roc_auc_std"])
        sstd = float(ov.loc[(DATASET, setting, "scaffold"), "roc_auc_std"])
        gap = (st["test_to_train_nn_tanimoto_mean"]["random"]
               - st["test_to_train_nn_tanimoto_mean"]["scaffold"])
        ratio = d / t_drop
        verdict = "corroborates" if 0.5 <= ratio <= 2.0 else "diverges"
        st.update({"roc_auc_drop": d, "drop_ratio_to_tox21": ratio, "verdict": verdict,
                   "nn_tanimoto_gap": gap})
        evidence = (f"scaffolds per molecule {st['scaffolds_per_molecule']:.4f} vs Tox21 "
                    f"{tox_spm:.4f}; random-minus-scaffold gap in mean test->train NN Tanimoto "
                    f"{gap:.4f} vs Tox21 {tox_nn_gap:.4f}; seed std of ROC-AUC "
                    f"{rstd:.4f} (random) / {sstd:.4f} (scaffold)")
        verdicts.append(f"- **{setting}: {verdict.capitalize()}.** Drop {d:.4f} vs Tox21 "
                        f"{t_drop:.4f} (ratio {ratio:.2f}, verdict band 0.5-2.0). Measured "
                        f"context: {evidence}.")
    (paths.RESULTS / "herg_large_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    f = stats["settings"]["full"]
    lines = [
        "# Large hERG external check (TDC `herg_karim`)", "",
        "Same standardizer, splitters (80/10/10, seeds 0-4) and RF baseline as Tox21. "
        "Generated by `src/18_herg_large.py`.", "",
        "## Dataset", "",
        "| quantity | value |", "|---|---|",
        f"| TDC dataset name | `{TDC_NAME}` |",
        f"| input rows | {stats['input_rows']} |",
        f"| parse failures | {stats['parse_failures']} |",
        f"| duplicate rows merged | {stats['duplicate_rows_merged']} |",
        f"| label ties set to NaN | {stats['label_cells_nan_from_tie']} |",
        f"| filtered out | {stats['filtered_out']} |",
        f"| final molecules (standardizer) | {stats['final_molecules']} |",
        f"| dropped as unlabeled (tie NaN) | {stats['unlabeled_after_ties_dropped']} |",
        f"| Full setting molecules | {f['molecules']} |",
        f"| positives (Full) | {f['n_positive']} ({f['positive_rate']:.4f}) |",
        f"| InChIKey overlap with Tox21 ({len(tox21)}) | {overlap} "
        f"({stats['inchikey_overlap_pct_of_herg']:.2f}% of hERG-large) |",
        f"| removed for Disjoint setting | {overlap} |",
        f"| Disjoint setting molecules | {stats['settings']['disjoint']['molecules']} "
        f"(positive rate {stats['settings']['disjoint']['positive_rate']:.4f}) |", "",
        "## Scaffold diversity", "",
        "| set | molecules | unique scaffolds | scaffolds / molecule | largest group |",
        "|---|---|---|---|---|",
        *[f"| hERG-large {k} | {v['molecules']} | {v['scaffolds']['n_unique_scaffolds']} | "
          f"{v['scaffolds_per_molecule']:.4f} | {v['scaffolds']['largest_group_size']} |"
          for k, v in stats["settings"].items()],
        f"| Tox21 | {len(tox21)} | {tox_sstats['n_unique_scaffolds']} | {tox_spm:.4f} | "
        f"{tox_sstats['largest_group_size']} |", "",
        "## Splits", "",
        "| setting | split | seed | train | valid | test | test pos. rate | "
        "scaffold groups spanning subsets |", "|---|---|---|---|---|---|---|---|",
        *[f"| {r.setting} | {r.split} | {r.seed} | {r.train} | {r.valid} | {r.test} | "
          f"{r.test_pos_rate:.4f} | {r.scaffold_groups_spanning_subsets} |"
          for r in sizes.itertuples()], "",
        "## Baseline (mean +/- sample std over 5 seeds)", "",
        "| dataset | setting | split | ROC-AUC | PR-AUC |", "|---|---|---|---|---|",
    ]
    for ds, st in ((DATASET, "full"), (DATASET, "disjoint"),
                   ("hERG", STANDARD), ("Tox21", STANDARD)):
        label = {"hERG": "hERG (old, 588-molecule TDC `herg`)",
                 "Tox21": "Tox21 (12-endpoint macro)"}.get(ds, ds)
        for kind in KINDS:
            r = ov.loc[(ds, st, kind)]
            lines.append(f"| {label} | {st} | {kind} | {r.roc_auc_mean:.4f} +/- "
                         f"{r.roc_auc_std:.4f} | {r.pr_auc_mean:.4f} +/- {r.pr_auc_std:.4f} |")
    lines += ["", "## Random -> scaffold ROC-AUC drop", "",
              f"- hERG-large full: {drop(DATASET, 'full'):.4f}",
              f"- hERG-large disjoint: {drop(DATASET, 'disjoint'):.4f}",
              f"- hERG old (588 molecules): {old_drop:.4f}",
              f"- Tox21: {t_drop:.4f}", "", "## Verdict", "", *verdicts, ""]
    (paths.RESULTS / "herg_large_baseline.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
