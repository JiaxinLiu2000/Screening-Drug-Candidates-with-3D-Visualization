"""Load, verify and swap trained models.

This is the boundary between the research pipeline and anything that consumes
its models. Everything a consumer needs is here:

    >>> from tox21lib.model_io import load_model
    >>> tox = load_model("toxicity_gin")
    >>> tox.predict(["CCO", "c1ccccc1N(=O)=O"])     # (2, 12) probabilities

Swapping in your own model needs no code change in the callers. Either point at
a file directly::

    load_model("path/to/your_model.pt")

or register it permanently by adding an entry to ``models/registry.json``. A
checkpoint is self-describing - it carries its architecture, the endpoint order
and the feature contract it was trained under - so a replacement is rejected at
load time if it does not honour the same contract, rather than silently
producing numbers in the wrong column order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from . import paths

MODELS = paths.ROOT / "models"
REGISTRY = MODELS / "registry.json"

#: Bumped when the checkpoint envelope changes shape. A checkpoint from an
#: older format is refused with an explanation rather than mis-read.
FORMAT_VERSION = 1


class ModelContractError(RuntimeError):
    """A checkpoint loaded, but does not honour the interface it claims."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def registry() -> dict:
    if not REGISTRY.exists():
        raise FileNotFoundError(
            f"{REGISTRY} missing. Run `python src/38_export_models.py` to build "
            f"the model directory from trained checkpoints.")
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------

@dataclass
class ToxicityPredictor:
    """12-endpoint toxicity probabilities from SMILES."""

    model: object
    endpoints: list[str]
    device: str = "cpu"
    metadata: dict = field(default_factory=dict)

    def predict(self, smiles: Sequence[str]) -> np.ndarray:
        """Return an ``(n_molecules, n_endpoints)`` array of probabilities.

        Unparsable SMILES yield a row of NaN rather than being dropped, so the
        output always aligns row-for-row with the input.
        """
        import torch
        from torch_geometric.loader import DataLoader

        from .ablation_data import _to_data
        from . import graphs

        parsed, keep = [], []
        for i, smi in enumerate(smiles):
            try:
                parsed.append(_to_data(graphs.mol_to_graph(smi)))
            except ValueError:
                continue          # unparsable -> NaN row, input order preserved
            keep.append(i)

        out = np.full((len(smiles), len(self.endpoints)), np.nan, dtype=np.float32)
        if not parsed:
            return out

        self.model.eval()
        rows = []
        with torch.no_grad():
            for batch in DataLoader(parsed, batch_size=256):
                batch = batch.to(self.device)
                logits = self.model.predict_toxicity(batch)
                rows.append(torch.sigmoid(logits).cpu().numpy())
        out[keep] = np.concatenate(rows, axis=0)
        return out

    def predict_frame(self, smiles: Sequence[str]):
        import pandas as pd

        frame = pd.DataFrame(self.predict(smiles), columns=self.endpoints)
        frame.insert(0, "smiles", list(smiles))
        return frame


@dataclass
class AffinityPredictor:
    """pKd-scale affinity for (molecule, protein) pairs.

    Protein embeddings are looked up from the committed ESM-2 table by UniProt
    accession. A protein outside that table needs its embedding computed first
    (`src/19_esm_embed.py`); asking for one raises rather than silently
    substituting a zero vector, because a zero vector is a valid-looking input
    that produces a plausible-looking wrong answer.
    """

    model: object
    protein_table: dict
    n_bits: int
    radius: int
    device: str = "cpu"
    metadata: dict = field(default_factory=dict)
    #: Column-wise standardisation applied to protein embeddings at training
    #: time. Stored with the weights because feeding raw ESM-2 vectors silently
    #: costs ~0.37 Pearson rather than failing.
    protein_mean: np.ndarray | None = None
    protein_std: np.ndarray | None = None

    @property
    def known_proteins(self) -> list[str]:
        return sorted(self.protein_table)

    def predict(self, smiles: Sequence[str], protein_ids: Sequence[str]) -> np.ndarray:
        import torch

        from . import featurize

        if len(smiles) != len(protein_ids):
            raise ValueError(
                f"{len(smiles)} molecules against {len(protein_ids)} proteins; "
                f"predict() is pairwise, pass one protein id per molecule")
        unknown = sorted(set(protein_ids) - set(self.protein_table))
        if unknown:
            raise ModelContractError(
                f"no embedding for {unknown[:5]}{'...' if len(unknown) > 5 else ''}. "
                f"Known: {len(self.protein_table)} proteins. Compute new ones with "
                f"src/19_esm_embed.py.")

        from rdkit import Chem

        smiles = list(smiles)
        valid = np.array([Chem.MolFromSmiles(s) is not None for s in smiles])
        out = np.full(len(smiles), np.nan, dtype=np.float32)
        if not valid.any():
            return out

        fps = featurize.featurize_smiles(
            [s for s, ok in zip(smiles, valid) if ok],
            radius=self.radius, n_bits=self.n_bits)
        prot = np.stack([self.protein_table[p]
                         for p, ok in zip(protein_ids, valid) if ok])
        if self.protein_mean is not None:
            prot = (prot - self.protein_mean) / self.protein_std

        self.model.eval()
        with torch.no_grad():
            pred = self.model(
                torch.tensor(fps, dtype=torch.float32, device=self.device),
                torch.tensor(prot, dtype=torch.float32, device=self.device),
            )
        out[valid] = pred.cpu().numpy()
        return out


@dataclass
class MultiTaskPredictor:
    """One shared GIN encoder feeding a toxicity head and an affinity head."""

    model: object
    endpoints: list[str]
    protein_table: dict
    device: str = "cpu"
    metadata: dict = field(default_factory=dict)
    protein_mean: np.ndarray | None = None
    protein_std: np.ndarray | None = None

    def _graphs(self, smiles: Sequence[str]):
        from .ablation_data import _to_data
        from . import graphs

        parsed, keep = [], []
        for i, smi in enumerate(smiles):
            try:
                parsed.append(_to_data(graphs.mol_to_graph(smi)))
            except ValueError:
                continue
            keep.append(i)
        return parsed, keep

    def predict_toxicity(self, smiles: Sequence[str]) -> np.ndarray:
        import torch
        from torch_geometric.loader import DataLoader

        parsed, keep = self._graphs(smiles)
        out = np.full((len(smiles), len(self.endpoints)), np.nan, dtype=np.float32)
        if not parsed:
            return out
        self.model.eval()
        rows = []
        with torch.no_grad():
            for batch in DataLoader(parsed, batch_size=256):
                rows.append(torch.sigmoid(
                    self.model.predict_toxicity(batch.to(self.device))).cpu().numpy())
        out[keep] = np.concatenate(rows, axis=0)
        return out

    def predict_affinity(self, smiles: Sequence[str],
                         protein_ids: Sequence[str]) -> np.ndarray:
        import torch
        from torch_geometric.loader import DataLoader

        if len(smiles) != len(protein_ids):
            raise ValueError("predict_affinity is pairwise; pass one protein per molecule")
        unknown = sorted(set(protein_ids) - set(self.protein_table))
        if unknown:
            raise ModelContractError(f"no embedding for {unknown[:5]}")

        parsed, keep = self._graphs(smiles)
        out = np.full(len(smiles), np.nan, dtype=np.float32)
        if not parsed:
            return out
        prot = np.stack([self.protein_table[protein_ids[i]] for i in keep])
        if self.protein_mean is not None:
            prot = (prot - self.protein_mean) / self.protein_std

        self.model.eval()
        preds, at = [], 0
        with torch.no_grad():
            for batch in DataLoader(parsed, batch_size=256):
                n = batch.num_graphs
                emb = torch.tensor(prot[at:at + n], dtype=torch.float32,
                                   device=self.device)
                preds.append(self.model.predict_affinity(
                    batch.to(self.device), emb).cpu().numpy())
                at += n
        out[keep] = np.concatenate(preds, axis=0)
        return out


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _build_toxicity(ckpt: dict, device: str):
    """Reconstruct the exact MultiTaskModel the ablation trained."""
    from .models import GINEncoder, MultiTaskModel

    arch = ckpt["architecture"]
    encoder = GINEncoder(out_dim=arch["out_dim"], hidden_dim=arch["hidden_dim"],
                         n_layers=arch["n_layers"], dropout=arch["dropout"])
    model = MultiTaskModel(encoder, n_endpoints=arch["n_endpoints"],
                           protein_dim=arch.get("protein_dim"),
                           dropout=arch["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)


def _build_affinity(ckpt: dict, device: str):
    """Reconstruct the frozen-embedding MLP from its stored architecture."""
    from .models import ConcatMLP

    arch = ckpt["architecture"]
    model = ConcatMLP(arch["drug_dim"], arch["protein_dim"],
                      hidden=tuple(arch["hidden"]), dropout=arch["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)


def load_model(name_or_path: str, device: str = "cpu", verify: bool = True):
    """Load a model by registry name or by path to a checkpoint file."""
    import torch

    reg = {}
    path = Path(name_or_path)
    if not path.exists():
        reg = registry()
        entry = reg["models"].get(name_or_path)
        if entry is None:
            raise KeyError(
                f"unknown model {name_or_path!r}; registry has "
                f"{sorted(reg['models'])}, or pass a path to a .pt file")
        path = MODELS / entry["file"]
        if verify and sha256(path) != entry["sha256"]:
            raise ModelContractError(
                f"{path.name} does not match the sha256 in registry.json. Either "
                f"the file changed or the registry is stale; if you replaced the "
                f"model deliberately, update its sha256 entry.")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    for key in ("format_version", "task", "architecture", "state_dict"):
        if key not in ckpt:
            raise ModelContractError(
                f"{path.name} has no {key!r}. Models consumed through this "
                f"interface must carry their own architecture and task; see "
                f"models/README.md for the envelope a replacement must provide.")
    if ckpt["format_version"] != FORMAT_VERSION:
        raise ModelContractError(
            f"{path.name} is format v{ckpt['format_version']}, this code reads "
            f"v{FORMAT_VERSION}")

    task = ckpt["task"]
    if task == "toxicity":
        endpoints = ckpt["endpoints"]
        if len(endpoints) != ckpt["architecture"]["n_endpoints"]:
            raise ModelContractError(
                f"{len(endpoints)} endpoint names for "
                f"{ckpt['architecture']['n_endpoints']} outputs")
        return ToxicityPredictor(_build_toxicity(ckpt, device), endpoints, device,
                                 ckpt.get("metadata", {}))
    if task == "affinity":
        table = load_protein_embeddings()
        dim = ckpt["architecture"]["protein_dim"]
        bad = [k for k, v in table.items() if v.shape[0] != dim]
        if bad:
            raise ModelContractError(
                f"model expects {dim}-d protein embeddings but the table holds "
                f"{table[bad[0]].shape[0]}-d vectors")
        pre = ckpt.get("preprocessing")
        if pre is None:
            raise ModelContractError(
                f"affinity checkpoint has no 'preprocessing' block. The protein "
                f"standardisation is part of the input contract; a model without "
                f"it cannot be served correctly.")
        return AffinityPredictor(
            _build_affinity(ckpt, device), table,
            ckpt["architecture"]["drug_dim"],
            ckpt["architecture"]["morgan_radius"], device,
            ckpt.get("metadata", {}),
            np.asarray(pre["protein_mean"], dtype=np.float32),
            np.asarray(pre["protein_std"], dtype=np.float32))
    if task == "multitask":
        pre = ckpt.get("preprocessing") or {}
        mean = pre.get("protein_mean")
        return MultiTaskPredictor(
            _build_toxicity(ckpt, device), ckpt["endpoints"],
            load_protein_embeddings(), device, ckpt.get("metadata", {}),
            np.asarray(mean, dtype=np.float32) if mean else None,
            np.asarray(pre["protein_std"], dtype=np.float32) if mean else None)
    raise ModelContractError(f"unsupported task {task!r}")


def load_protein_embeddings() -> dict[str, np.ndarray]:
    """Committed ESM-2 mean-pooled embeddings, keyed by UniProt accession."""
    reg = registry()
    npz = MODELS / reg["protein_embeddings"]["file"]
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} missing; it is committed with the repository, so a fresh "
            f"clone should have it. Regenerate with src/19_esm_embed.py "
            f"(downloads a 2.5 GB model).")
    from . import protein_embed

    matrix, rows, _ = protein_embed.load_cache(npz)
    return {accession: matrix[row] for accession, row in rows.items()}
