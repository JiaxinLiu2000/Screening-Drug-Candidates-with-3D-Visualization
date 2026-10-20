"""Dataset plumbing shared by the four ablation configurations.

Both tasks are served from the SAME committed split index files the random
forests used, so the neural results sit on exactly the same folds as the floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from . import datasets, dti, graphs, paths

TOX_ENDPOINTS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def _to_data(graph: graphs.MolGraph) -> Data:
    return Data(
        x=torch.from_numpy(graph.x),
        edge_index=torch.from_numpy(graph.edge_index),
        edge_attr=torch.from_numpy(graph.edge_attr),
    )


def tox21_graph_dataset() -> tuple[list[Data], pd.DataFrame]:
    """Graph per Tox21 molecule, with masked labels attached."""
    clean = pd.read_parquet(paths.CLEAN_PARQUET)
    built = graphs.build_graphs(clean["canonical_smiles"])
    labels = clean[TOX_ENDPOINTS].to_numpy(dtype=np.float32)
    mask = (~np.isnan(labels)).astype(np.float32)
    labels = np.nan_to_num(labels, nan=0.0)

    out = []
    for i, graph in enumerate(built):
        data = _to_data(graph)
        data.y_tox = torch.from_numpy(labels[i]).view(1, -1)
        data.mask_tox = torch.from_numpy(mask[i]).view(1, -1)
        # NOTE: PyG increments any attribute whose key contains "index" by the
        # node offset when batching, which would corrupt row pointers - hence
        # "_row" names here.
        data.mol_row = torch.tensor([i], dtype=torch.long)
        out.append(data)
    return out, clean


def tox21_subsets(kind: str, seed: int) -> np.ndarray:
    """Subset labels from the committed Tox21 split index file."""
    frame = pd.read_csv(paths.SPLITS / f"{kind}_split_seed{seed}.csv")
    return frame["subset"].to_numpy()


def kiba_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardized KIBA pairs plus the molecule table (InChIKey keyed)."""
    raw = datasets.load_dti("KIBA")
    clean, _ = datasets.unique_molecules(raw)
    pairs, _ = dti.build_pairs(raw, clean)
    return pairs, clean


def kiba_graph_dataset(pairs: pd.DataFrame, clean: pd.DataFrame,
                       protein_rows: dict[str, int]) -> list[Data]:
    """One graph-backed sample per measured drug-protein pair.

    Graphs are built once per distinct drug and shared by reference across the
    pairs that use them, so 117k pairs cost 2030 graph builds.
    """
    smiles_by_key = dict(zip(clean["inchikey"], clean["canonical_smiles"]))
    keys = sorted(pairs["inchikey"].unique())
    graph_by_key = {k: graphs.mol_to_graph(smiles_by_key[k]) for k in keys}

    affinity = pairs["affinity"].to_numpy(dtype=np.float32)
    out = []
    for i, (key, target, y) in enumerate(
        zip(pairs["inchikey"], pairs["target_id"], affinity)
    ):
        data = _to_data(graph_by_key[key])
        data.y_affinity = torch.tensor([y], dtype=torch.float32)
        data.protein_row = torch.tensor([protein_rows[target]], dtype=torch.long)
        data.pair_row = torch.tensor([i], dtype=torch.long)
        out.append(data)
    return out


def kiba_subsets(kind: str, seed: int, n_pairs: int) -> np.ndarray:
    """Subset labels from the committed KIBA split index file."""
    frame = pd.read_csv(paths.SPLITS / f"kiba_{kind}_seed{seed}.csv")
    if len(frame) != n_pairs:
        raise RuntimeError(
            f"kiba_{kind}_seed{seed}.csv has {len(frame)} rows, expected {n_pairs}"
        )
    return frame.sort_values("pair_index")["subset"].to_numpy()


def loaders_for(
    dataset: list[Data],
    subsets: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    """train/valid/test loaders over one split, with a seeded train shuffle."""
    out = {}
    for name in ("train", "valid", "test"):
        idx = np.flatnonzero(subsets == name)
        items = [dataset[i] for i in idx]
        generator = torch.Generator()
        generator.manual_seed(seed)
        out[name] = DataLoader(
            items,
            batch_size=batch_size,
            shuffle=shuffle_train and name == "train",
            generator=generator if (shuffle_train and name == "train") else None,
        )
    return out


def positive_weights(clean: pd.DataFrame) -> torch.Tensor:
    """negatives/positives per endpoint, to balance the masked BCE like the RF."""
    weights = []
    for col in TOX_ENDPOINTS:
        y = clean[col].to_numpy(dtype=float)
        y = y[~np.isnan(y)]
        pos = max(float((y == 1).sum()), 1.0)
        neg = float((y == 0).sum())
        weights.append(neg / pos)
    return torch.tensor(weights, dtype=torch.float32)


#: ChemBERTa context; SMILES longer than this are truncated and the count is
#: reported by the ablation script rather than hidden.
SMILES_MAX_TOKENS = 128


def attach_tokens(dataset: list[Data], smiles_per_sample: list[str], tokenizer,
                  max_length: int = SMILES_MAX_TOKENS) -> int:
    """Attach padded SMILES token ids to each sample; returns the truncated count.

    Tokens are padded to a fixed width so PyG's batching concatenates them into
    (batch, max_length) without a custom collate function. Distinct SMILES are
    tokenized once and shared by reference, mirroring the graph cache.
    """
    unique = sorted(set(smiles_per_sample))
    encoded = tokenizer(
        unique, padding="max_length", truncation=True, max_length=max_length,
        return_tensors="pt",
    )
    lengths = tokenizer(unique, truncation=False)["input_ids"]
    n_truncated = sum(1 for ids in lengths if len(ids) > max_length)
    by_smiles = {
        smi: (encoded["input_ids"][i], encoded["attention_mask"][i])
        for i, smi in enumerate(unique)
    }
    for data, smi in zip(dataset, smiles_per_sample):
        ids, attn = by_smiles[smi]
        data.input_ids = ids.view(1, -1)
        data.attention_mask = attn.view(1, -1)
    return n_truncated
