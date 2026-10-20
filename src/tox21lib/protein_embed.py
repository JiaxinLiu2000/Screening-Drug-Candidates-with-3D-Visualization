"""Frozen ESM-2 protein embeddings for the affinity head.

The embeddings never change during training, so they are computed once and
cached to ``data/processed/esm2_<model>.npz``. ESM-2 accepts at most 1022
residues per forward pass; 42 of KIBA's 229 proteins are longer (up to 4128), so
long sequences are embedded in non-overlapping windows and combined by a
length-weighted mean - documented here because it is a modelling choice, not a
detail.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: ESM-2 650M: the largest checkpoint that embeds 1022-residue windows inside
#: 8 GB of VRAM at batch size 1 in half precision.
DEFAULT_MODEL = "esm2_t33_650M_UR50D"
MAX_WINDOW = 1022


def cache_path(processed_dir: Path, model_name: str = DEFAULT_MODEL) -> Path:
    return processed_dir / f"{model_name}.npz"


def embed_sequences(
    sequences: dict[str, str],
    *,
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    half: bool = True,
    progress_every: int = 25,
) -> tuple[np.ndarray, list[str], int]:
    """Mean-pooled per-protein embeddings. Returns (matrix, ids, embedding_dim).

    Pooling excludes the BOS/EOS tokens; windows of a long sequence are pooled
    individually and averaged weighted by window length, which equals pooling
    over all residues at once.
    """
    import esm
    import torch

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device)
    if half and device == "cuda":
        model = model.half()
    converter = alphabet.get_batch_converter()
    n_layers = model.num_layers
    dim = model.embed_dim

    ids = sorted(sequences)
    out = np.zeros((len(ids), dim), dtype=np.float32)

    with torch.no_grad():
        for row, target_id in enumerate(ids):
            seq = sequences[target_id]
            windows = [seq[i:i + MAX_WINDOW] for i in range(0, len(seq), MAX_WINDOW)]
            pooled, weights = [], []
            for window in windows:
                _, _, tokens = converter([(target_id, window)])
                tokens = tokens.to(device)
                reps = model(tokens, repr_layers=[n_layers])["representations"][n_layers]
                # drop BOS (0) and EOS (last) before pooling
                residues = reps[0, 1: len(window) + 1].float()
                pooled.append(residues.mean(dim=0).cpu().numpy())
                weights.append(len(window))
            weights = np.asarray(weights, dtype=np.float64)
            out[row] = np.average(np.stack(pooled), axis=0, weights=weights)
            if progress_every and (row + 1) % progress_every == 0:
                print(f"    embedded {row + 1}/{len(ids)} proteins")

    return out, ids, dim


def save_cache(path: Path, matrix: np.ndarray, ids: list[str], model_name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=matrix, ids=np.array(ids, dtype=object),
                        model=model_name)
    meta = path.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "model": model_name,
                "n_proteins": len(ids),
                "embedding_dim": int(matrix.shape[1]),
                "max_window": MAX_WINDOW,
                "pooling": "length-weighted mean over residue representations, "
                           "BOS/EOS excluded, windows of at most 1022 residues",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def load_cache(path: Path) -> tuple[np.ndarray, dict[str, int], str]:
    """Returns (matrix, {target_id: row}, model_name)."""
    with np.load(path, allow_pickle=True) as data:
        matrix = data["embeddings"]
        ids = [str(x) for x in data["ids"]]
        model = str(data["model"])
    return matrix, {t: i for i, t in enumerate(ids)}, model
