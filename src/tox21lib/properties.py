"""RDKit physicochemical descriptors shared by chemical-space analyses."""

from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

PROPERTIES = {
    "mol_weight": Descriptors.MolWt,
    "clogp": Crippen.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "ring_count": rdMolDescriptors.CalcNumRings,
    "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings,
    "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
}


def descriptor_table(smiles: list[str]) -> pd.DataFrame:
    """One row per SMILES, one column per entry of PROPERTIES."""
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"unparseable SMILES: {smi}")
        rows.append({k: float(f(mol)) for k, f in PROPERTIES.items()})
    return pd.DataFrame(rows, columns=list(PROPERTIES))
