"""Molecule standardization for v1 of the screening pipeline.

Pipeline per SMILES:
  1. parse (failures are reported, not silently dropped)
  2. strip salts -> keep the largest covalently connected fragment
  3. remove stereochemistry (v1 simplification, see README)
  4. canonical SMILES + InChIKey
Then, at table level: InChIKey deduplication with majority-vote label merge and
heavy-atom / element filtering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

#: Elements allowed in v1 (organic subset + halogens; H is implicit/explicit H).
ALLOWED_ELEMENTS: frozenset[str] = frozenset(
    {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"}
)
MIN_HEAVY_ATOMS = 10
MAX_HEAVY_ATOMS = 60


@dataclass(frozen=True)
class StandardizedMol:
    """Result of standardizing one input SMILES."""

    canonical_smiles: str
    inchikey: str
    n_heavy_atoms: int
    ok: bool
    reason: str = ""

    @classmethod
    def failure(cls, reason: str) -> "StandardizedMol":
        return cls("", "", 0, False, reason)


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Return the largest covalently connected fragment (by heavy-atom count).

    Ties are broken by the fragment's canonical SMILES so the choice is
    deterministic across runs and platforms.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) == 1:
        return frags[0]
    return max(frags, key=lambda f: (f.GetNumHeavyAtoms(), Chem.MolToSmiles(f)))


def standardize_smiles(smiles: str) -> StandardizedMol:
    """Parse, desalt, flatten stereochemistry; return canonical SMILES + InChIKey."""
    if not isinstance(smiles, str) or not smiles.strip():
        return StandardizedMol.failure("empty_smiles")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return StandardizedMol.failure("parse_error")

    mol = largest_fragment(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - RDKit raises assorted sanitization errors
        return StandardizedMol.failure("sanitize_error")

    if mol.GetNumHeavyAtoms() == 0:
        return StandardizedMol.failure("no_heavy_atoms")

    # v1: stereochemistry is discarded (both atom and bond stereo).
    Chem.RemoveStereochemistry(mol)

    canonical = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return StandardizedMol.failure("recanonicalize_error")

    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey:
        return StandardizedMol.failure("inchikey_error")

    return StandardizedMol(canonical, inchikey, mol.GetNumHeavyAtoms(), True)


def element_set(smiles: str) -> set[str]:
    """Elements present in a molecule, including explicit/implicit hydrogens."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    if any(a.GetTotalNumHs() > 0 for a in mol.GetAtoms()):
        symbols.add("H")
    return symbols


def passes_property_filter(
    smiles: str,
    n_heavy_atoms: int,
    *,
    min_heavy: int = MIN_HEAVY_ATOMS,
    max_heavy: int = MAX_HEAVY_ATOMS,
    allowed: frozenset[str] = ALLOWED_ELEMENTS,
) -> tuple[bool, str]:
    """Heavy-atom-count and element whitelist filter. Returns (keep, reason)."""
    if n_heavy_atoms < min_heavy:
        return False, "too_few_heavy_atoms"
    if n_heavy_atoms > max_heavy:
        return False, "too_many_heavy_atoms"
    extra = element_set(smiles) - allowed
    if extra:
        return False, "disallowed_element:" + ",".join(sorted(extra))
    return True, ""


def merge_label(values: pd.Series) -> float:
    """Majority vote over duplicate label observations; NaN when tied or absent.

    Tox21 labels are binary {0, 1}; a tie between 0s and 1s is treated as
    unresolvable and becomes NaN (missing), not an arbitrary pick.
    """
    vals = values.dropna()
    if vals.empty:
        return float("nan")
    counts = vals.astype(float).round().astype(int).value_counts()
    top = counts.max()
    winners = counts[counts == top].index.tolist()
    if len(winners) != 1:
        return float("nan")
    return float(winners[0])


def deduplicate(df: pd.DataFrame, label_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Collapse rows sharing an InChIKey; majority-vote each label column.

    Returns the deduplicated frame (sorted by InChIKey for determinism) and a
    stats dict with the number of rows removed and conflicts set to NaN.
    """
    n_before = len(df)
    n_keys = df["inchikey"].nunique()

    conflicts = 0
    records: list[dict] = []
    for key, group in df.groupby("inchikey", sort=True):
        row: dict = {
            "inchikey": key,
            # canonical SMILES is a function of the InChIKey-equivalent graph;
            # min() keeps the choice deterministic if RDKit ever disagrees.
            "canonical_smiles": group["canonical_smiles"].min(),
            "n_heavy_atoms": int(group["n_heavy_atoms"].iloc[0]),
            "drug_ids": "|".join(sorted(set(group["drug_id"].astype(str)))),
            "n_source_rows": len(group),
        }
        for col in label_cols:
            merged = merge_label(group[col])
            observed = group[col].dropna()
            if len(observed) > 1 and np.isnan(merged):
                conflicts += 1
            row[col] = merged
        records.append(row)

    out = pd.DataFrame.from_records(records).sort_values("inchikey", ignore_index=True)
    stats = {
        "rows_before_dedup": n_before,
        "unique_inchikeys": n_keys,
        "duplicate_rows_merged": n_before - n_keys,
        "label_cells_nan_from_tie": conflicts,
    }
    return out, stats


def standardize_table(
    df: pd.DataFrame,
    label_cols: list[str],
    *,
    smiles_col: str = "smiles",
    id_col: str = "drug_id",
) -> tuple[pd.DataFrame, dict]:
    """Full table-level standardization: parse -> dedup -> filter.

    Returns the clean frame and a stats dict with every count the report needs.
    """
    std = [standardize_smiles(s) for s in df[smiles_col]]

    failures = pd.Series([r.reason for r in std if not r.ok]).value_counts()
    keep_mask = np.array([r.ok for r in std])

    parsed = pd.DataFrame(
        {
            "drug_id": df[id_col].to_numpy(),
            "canonical_smiles": [r.canonical_smiles for r in std],
            "inchikey": [r.inchikey for r in std],
            "n_heavy_atoms": [r.n_heavy_atoms for r in std],
        }
    )
    for col in label_cols:
        parsed[col] = df[col].to_numpy()
    parsed = parsed[keep_mask].reset_index(drop=True)

    deduped, dedup_stats = deduplicate(parsed, label_cols)

    checks = [
        passes_property_filter(s, n)
        for s, n in zip(deduped["canonical_smiles"], deduped["n_heavy_atoms"])
    ]
    keep = np.array([c[0] for c in checks])
    reasons = pd.Series([c[1].split(":")[0] for c, k in zip(checks, keep) if not k])
    clean = deduped[keep].reset_index(drop=True)

    stats = {
        "input_rows": len(df),
        "parse_failures": int((~keep_mask).sum()),
        "parse_failure_breakdown": failures.to_dict(),
        **dedup_stats,
        "filtered_out": int((~keep).sum()),
        "filter_breakdown": reasons.value_counts().to_dict(),
        "final_molecules": len(clean),
    }
    return clean, stats
