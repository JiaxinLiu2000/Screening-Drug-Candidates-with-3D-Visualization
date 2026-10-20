"""TDC dataset loading + shared standardization for every dataset in the project.

Tox21, hERG, Davis and KIBA all go through the *same* ``standardize`` pipeline,
so cross-dataset comparisons (InChIKey overlap, chemical-space distance) are
apples to apples. Nothing here reimplements chemistry: it delegates to
``tox21lib.standardize``.
"""

from __future__ import annotations

import pandas as pd

from . import paths, standardize


def load_tox_single(name: str, label_name: str | None = None) -> pd.DataFrame:
    """One TDC single-prediction toxicity dataset as (drug_id, smiles, label)."""
    from tdc.single_pred import Tox

    kwargs = {"name": name, "path": str(paths.DATA_RAW)}
    if label_name is not None:
        kwargs["label_name"] = label_name
    df = Tox(**kwargs).get_data()
    return df.rename(columns={"Drug_ID": "drug_id", "Drug": "smiles", "Y": "label"})


def load_dti(name: str) -> pd.DataFrame:
    """One TDC drug-target-interaction dataset as a long table of measured pairs.

    Columns: drug_id, smiles, target_id, target_sequence, affinity.
    """
    from tdc.multi_pred import DTI

    df = DTI(name=name, path=str(paths.DATA_RAW)).get_data()
    return df.rename(
        columns={
            "Drug_ID": "drug_id",
            "Drug": "smiles",
            "Target_ID": "target_id",
            "Target": "target_sequence",
            "Y": "affinity",
        }
    )


def standardize_molecules(
    df: pd.DataFrame,
    *,
    smiles_col: str = "smiles",
    id_col: str = "drug_id",
    label_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run any (drug_id, smiles[, labels]) table through the Tox21 standardizer.

    Returns the deduplicated, filtered frame and the same stats dict that
    ``02_standardize.py`` reports, so every dataset's attrition is auditable in
    the same terms.
    """
    return standardize.standardize_table(
        df, label_cols or [], smiles_col=smiles_col, id_col=id_col
    )


def unique_molecules(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardize the distinct drugs of a DTI table (one row per molecule)."""
    drugs = (
        df[["drug_id", "smiles"]]
        .drop_duplicates(subset=["drug_id", "smiles"])
        .reset_index(drop=True)
    )
    return standardize_molecules(drugs)
