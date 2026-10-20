"""Atom-masking attribution for the toxicity model.

For each heavy atom: replace its feature row with the unknown-atom vector (all
features zero, which is what the one-hot encoders emit for an unseen atom),
re-run inference, and record the drop in predicted probability. The attribution
vector is ordered to match the molecule's canonical-SMILES atom order, which is
the order ``graphs.mol_to_graph`` builds nodes in.

There is no ground truth for attribution, so validation here is qualitative:
whether the signal concentrates on known structural alerts.
"""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem

from .graphs import mol_to_graph

#: Well-known toxicophores, as SMARTS. Used only to *probe* the attribution, not
#: to train or select anything.
STRUCTURAL_ALERTS: dict[str, str] = {
    "nitroaromatic": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][c]",
    "aromatic_amine": "[NX3;H2,H1;!$(N[C,S]=[O,S,N])][c]",
    "para_quinone": "O=C1C=CC(=O)C=C1",
    "ortho_quinone": "O=C1C(=O)C=CC=C1",
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
    "aromatic_halide": "[F,Cl,Br,I][c]",
    "epoxide": "C1OC1",
    "aldehyde": "[CX3H1](=O)[#6]",
}


def alert_atoms(smiles: str) -> dict[str, list[int]]:
    """Atom indices matched by each alert present in the molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    found: dict[str, list[int]] = {}
    for name, smarts in STRUCTURAL_ALERTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            found[name] = sorted({idx for match in matches for idx in match})
    return found


@torch.no_grad()
def attribute(model, smiles: str, device: str = "cpu") -> dict:
    """Attribution matrix (n_atoms, n_endpoints) by single-atom masking.

    Returns the baseline probabilities, the masked probabilities and the drops
    (baseline - masked), so a positive value means "removing this atom lowered
    the predicted probability", i.e. the atom pushed the prediction upwards.
    """
    from torch_geometric.data import Batch, Data

    model.eval()
    graph = mol_to_graph(smiles)
    n_atoms = graph.n_atoms

    def as_data(x: np.ndarray) -> Data:
        return Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(graph.edge_index),
            edge_attr=torch.from_numpy(graph.edge_attr),
        )

    variants = [as_data(graph.x)]
    for atom_idx in range(n_atoms):
        masked = graph.x.copy()
        masked[atom_idx] = 0.0  # unknown-atom token
        variants.append(as_data(masked))

    batch = Batch.from_data_list(variants).to(device)
    probs = torch.sigmoid(model.predict_toxicity(batch)).float().cpu().numpy()

    baseline = probs[0]
    masked = probs[1:]
    return {
        "smiles": smiles,
        "n_atoms": n_atoms,
        "baseline_probs": baseline,          # (n_endpoints,)
        "masked_probs": masked,              # (n_atoms, n_endpoints)
        "attribution": baseline[None, :] - masked,  # (n_atoms, n_endpoints)
    }


def concentration(values: np.ndarray, atom_subset: list[int]) -> dict:
    """How much of the positive attribution mass sits on ``atom_subset``.

    ``fraction_of_mass`` is the share of total positive attribution on those
    atoms; ``enrichment`` compares it with the share expected if the signal were
    spread uniformly over atoms (i.e. |subset| / n_atoms).
    """
    pos = np.clip(values, 0.0, None)
    total = float(pos.sum())
    if total <= 0 or not atom_subset:
        return {"fraction_of_mass": 0.0, "enrichment": 0.0,
                "mean_on_subset": 0.0, "mean_off_subset": float(values.mean())}
    subset = np.zeros(len(values), dtype=bool)
    subset[atom_subset] = True
    share = float(pos[subset].sum()) / total
    expected = subset.sum() / len(values)
    off = values[~subset]
    return {
        "fraction_of_mass": share,
        "enrichment": share / expected if expected else 0.0,
        "mean_on_subset": float(values[subset].mean()),
        "mean_off_subset": float(off.mean()) if off.size else 0.0,
    }


def diffuseness(values: np.ndarray) -> float:
    """Normalised entropy of the positive attribution; 1.0 = perfectly diffuse.

    A molecule whose attribution is spread evenly over every atom carries no
    localised explanation, which is one of the failure modes worth reporting.
    """
    pos = np.clip(values, 0.0, None)
    total = pos.sum()
    if total <= 0 or len(values) < 2:
        return 1.0
    p = pos / total
    nz = p[p > 0]
    entropy = -float((nz * np.log(nz)).sum())
    return entropy / float(np.log(len(values)))


def top_atoms(values: np.ndarray, k: int = 3) -> list[int]:
    """Indices of the k most strongly attributed atoms, highest first."""
    return [int(i) for i in np.argsort(-values)[:k]]
