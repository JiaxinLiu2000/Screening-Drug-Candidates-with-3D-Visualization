"""Random and Bemis-Murcko scaffold splits, computed on the deduplicated set.

Both splitters return a ``pandas.Series`` of subset labels ("train"/"valid"/
"test") aligned to the input frame's row order, so they can be persisted as
index files and replayed without re-running RDKit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

TRAIN_FRAC = 0.8
VALID_FRAC = 0.1
TEST_FRAC = 0.1
SUBSETS = ("train", "valid", "test")


def scaffold_smiles(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES; empty string for acyclic molecules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES passed to scaffold splitter: {smiles!r}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def random_split(
    n: int,
    seed: int,
    *,
    train_frac: float = TRAIN_FRAC,
    valid_frac: float = VALID_FRAC,
) -> np.ndarray:
    """80/10/10 random split of ``n`` rows; returns an array of subset labels."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_valid = int(round(valid_frac * n))
    labels = np.empty(n, dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train : n_train + n_valid]] = "valid"
    labels[order[n_train + n_valid :]] = "test"
    return labels


def scaffold_split(
    scaffolds: pd.Series,
    seed: int,
    *,
    train_frac: float = TRAIN_FRAC,
    valid_frac: float = VALID_FRAC,
) -> np.ndarray:
    """Group-by-scaffold split; whole scaffold groups stay in one subset.

    Groups are sorted by size descending and greedily assigned: train until it
    reaches ``train_frac``, then valid until ``valid_frac``, remainder to test.
    ``seed`` only permutes the order of *equal-sized* groups (the sort is
    stable), so the deterministic size ordering required by the spec is kept
    while the 5 seeds still probe different scaffold assignments.
    """
    n = len(scaffolds)
    groups: dict[str, list[int]] = {}
    for idx, scaf in enumerate(scaffolds):
        groups.setdefault(scaf, []).append(idx)

    keys = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]), reverse=True)  # stable -> seeded ties

    n_train_target = train_frac * n
    n_valid_target = valid_frac * n

    labels = np.empty(n, dtype=object)
    n_train = n_valid = 0
    for key in keys:
        members = groups[key]
        if n_train < n_train_target:
            subset, n_train = "train", n_train + len(members)
        elif n_valid < n_valid_target:
            subset, n_valid = "valid", n_valid + len(members)
        else:
            subset = "test"
        for idx in members:
            labels[idx] = subset
    return labels


def scaffold_stats(scaffolds: pd.Series) -> dict:
    """Unique scaffold count and largest scaffold group size."""
    counts = scaffolds.value_counts()
    largest = counts.index[0]
    return {
        "n_unique_scaffolds": int(counts.size),
        "largest_group_size": int(counts.iloc[0]),
        "largest_group_scaffold": "" if largest == "" else str(largest),
        "n_singleton_scaffolds": int((counts == 1).sum()),
        "n_acyclic_molecules": int(counts.get("", 0)),
    }


def split_frame(
    clean: pd.DataFrame, kind: str, seed: int, scaffolds: pd.Series | None = None
) -> pd.DataFrame:
    """Build the committed index-file frame for one (split kind, seed)."""
    if kind == "random":
        labels = random_split(len(clean), seed)
    elif kind == "scaffold":
        if scaffolds is None:
            raise ValueError("scaffold split requires precomputed scaffolds")
        labels = scaffold_split(scaffolds, seed)
    else:
        raise ValueError(f"unknown split kind: {kind!r}")

    frame = pd.DataFrame(
        {
            "row_index": np.arange(len(clean), dtype=int),
            "inchikey": clean["inchikey"].to_numpy(),
            "subset": labels,
        }
    )
    if scaffolds is not None:
        frame["scaffold_smiles"] = scaffolds.to_numpy()
    return frame
