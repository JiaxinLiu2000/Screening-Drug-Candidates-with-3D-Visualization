"""Task 1 - download the 12 Tox21 endpoints from TDC and merge into one wide table.

Endpoint names are retrieved programmatically from TDC, never hardcoded.
Output: data/processed/tox21_wide.parquet (one row per molecule, 12 label cols).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import paths  # noqa: E402


def download_wide() -> tuple[pd.DataFrame, list[str]]:
    from tdc.single_pred import Tox
    from tdc.utils import retrieve_label_name_list

    label_names = retrieve_label_name_list("Tox21")
    print(f"TDC reports {len(label_names)} Tox21 endpoints: {label_names}")

    frames = []
    for name in label_names:
        df = Tox(name="Tox21", label_name=name, path=str(paths.DATA_RAW)).get_data()
        df = df.rename(columns={"Drug_ID": "drug_id", "Drug": "smiles", "Y": name})
        print(f"  {name:<14} rows={len(df):>5}  positives={int(df[name].sum()):>4}")
        frames.append(df[["drug_id", "smiles", name]])

    wide = frames[0]
    for df in frames[1:]:
        wide = wide.merge(df, on=["drug_id", "smiles"], how="outer")
    return wide, list(label_names)


def main() -> int:
    paths.ensure_dirs()
    wide, label_names = download_wide()

    dup_ids = wide["drug_id"].duplicated().sum()
    if dup_ids:
        print(f"NOTE: {dup_ids} Drug_IDs map to more than one SMILES string")

    wide.attrs["label_names"] = label_names
    wide.to_parquet(paths.WIDE_PARQUET, index=False)

    print(f"\nMerged wide table shape: {wide.shape}  -> {paths.WIDE_PARQUET}")
    print(f"  molecules (unique SMILES strings): {wide['smiles'].nunique()}")
    print("\nPer-endpoint positive rate on the merged table:")
    rows = []
    for name in label_names:
        col = wide[name]
        tested = int(col.notna().sum())
        pos = int((col == 1).sum())
        rows.append(
            {
                "endpoint": name,
                "n_tested": tested,
                "n_positive": pos,
                "positive_rate": round(pos / tested, 4) if tested else float("nan"),
                "n_missing": int(col.isna().sum()),
            }
        )
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    out = paths.RESULTS / "tox21_endpoint_stats.csv"
    summary.to_csv(out, index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
