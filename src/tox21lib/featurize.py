"""Morgan fingerprint featurization (current RDKit generator API).

Uses ``rdFingerprintGenerator.GetMorganGenerator``; the deprecated
``GetMorganFingerprintAsBitVect`` is deliberately not used.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048


@lru_cache(maxsize=None)
def _generator(radius: int, n_bits: int):
    """Fingerprint generators are stateless and thread-safe -> cache them."""
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def morgan_bits(
    smiles: str, *, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_N_BITS
) -> np.ndarray:
    """Return the fingerprint of one molecule as a uint8 bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES passed to featurizer: {smiles!r}")
    fp = _generator(radius, n_bits).GetFingerprintAsNumPy(mol)
    return fp.astype(np.uint8, copy=False)


def featurize_smiles(
    smiles_list: Sequence[str] | Iterable[str],
    *,
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_N_BITS,
) -> np.ndarray:
    """Stack fingerprints for a list of SMILES into an (n_mols, n_bits) matrix."""
    smiles_list = list(smiles_list)
    out = np.empty((len(smiles_list), n_bits), dtype=np.uint8)
    gen = _generator(radius, n_bits)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"unparsable SMILES at row {i}: {smi!r}")
        out[i] = gen.GetFingerprintAsNumPy(mol)
    return out
