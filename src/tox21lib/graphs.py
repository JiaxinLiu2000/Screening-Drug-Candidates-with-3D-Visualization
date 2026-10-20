"""Molecular graphs for the GIN encoder (atoms as nodes, bonds as edges).

Feature sets are deliberately small and explicit so the encoder input is
reproducible across runs and documented in the report. Atom ordering follows the
RDKit canonical-SMILES atom order of the molecule as stored in
``tox21_clean.parquet``, which is what the atom-attribution stage indexes into.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

#: Elements kept by the v1 standardizer, plus a catch-all slot.
ATOM_SYMBOLS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "other")
HYBRIDIZATIONS = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
)
BOND_TYPES = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)
DEGREES = (0, 1, 2, 3, 4, 5)
NUM_HS = (0, 1, 2, 3, 4)
CHARGES = (-2, -1, 0, 1, 2)

ATOM_FEATURE_DIM = (
    len(ATOM_SYMBOLS) + len(DEGREES) + len(NUM_HS) + len(CHARGES)
    + len(HYBRIDIZATIONS) + 2  # aromatic, in-ring
)
BOND_FEATURE_DIM = len(BOND_TYPES) + 2  # conjugated, in-ring


def _one_hot(value, choices) -> list[float]:
    """One-hot with a trailing catch-all bucket for unseen values."""
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    """Per-atom feature vector; the masking attribution zeroes exactly this."""
    symbol = atom.GetSymbol()
    return (
        _one_hot(symbol if symbol in ATOM_SYMBOLS else "other", ATOM_SYMBOLS[:-1])
        + _one_hot(atom.GetTotalDegree(), DEGREES)[:-1]
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)[:-1]
        + _one_hot(atom.GetFormalCharge(), CHARGES)[:-1]
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)[:-1]
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)[:-1]
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    )


@dataclass(frozen=True)
class MolGraph:
    """A molecule as plain arrays; converted to a PyG ``Data`` by the loader."""

    x: np.ndarray            # (n_atoms, ATOM_FEATURE_DIM) float32
    edge_index: np.ndarray   # (2, n_edges) int64, both directions
    edge_attr: np.ndarray    # (n_edges, BOND_FEATURE_DIM) float32
    n_atoms: int


def mol_to_graph(smiles: str) -> MolGraph:
    """Build the graph for one canonical SMILES (heavy atoms only, no explicit H)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES passed to the graph builder: {smiles!r}")

    x = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    if x.shape[1] != ATOM_FEATURE_DIM:  # guards against a silent feature-set change
        raise RuntimeError(
            f"atom feature width {x.shape[1]} != declared ATOM_FEATURE_DIM "
            f"{ATOM_FEATURE_DIM}"
        )

    src, dst, attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        attrs += [feats, feats]

    if src:
        edge_index = np.array([src, dst], dtype=np.int64)
        edge_attr = np.array(attrs, dtype=np.float32)
    else:  # single-atom molecules have no bonds
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)

    return MolGraph(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    n_atoms=mol.GetNumAtoms())


def build_graphs(smiles_list) -> list[MolGraph]:
    """Graphs for a list of SMILES, in input order."""
    return [mol_to_graph(s) for s in smiles_list]
