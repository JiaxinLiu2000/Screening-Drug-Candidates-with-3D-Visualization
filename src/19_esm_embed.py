"""Phase 3 Task 5 (prerequisite) - cache frozen ESM-2 embeddings for KIBA proteins.

Run once; `src/20_shared_encoder_ablation.py` loads the cache. The embeddings are
frozen, so this is pure preprocessing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import datasets, paths, protein_embed  # noqa: E402


def main() -> int:
    paths.ensure_dirs()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    kiba = datasets.load_dti("KIBA")
    sequences = (
        kiba.drop_duplicates("target_id")
        .set_index("target_id")["target_sequence"]
        .to_dict()
    )
    lengths = np.array([len(s) for s in sequences.values()])
    print(f"{len(sequences)} distinct KIBA proteins; residue length "
          f"min {lengths.min()}, median {int(np.median(lengths))}, max {lengths.max()}; "
          f"{int((lengths > protein_embed.MAX_WINDOW).sum())} exceed the "
          f"{protein_embed.MAX_WINDOW}-residue window and are pooled over windows")

    out_path = protein_embed.cache_path(paths.DATA_PROCESSED)
    t0 = time.time()
    matrix, ids, dim = protein_embed.embed_sequences(sequences, device=device)
    protein_embed.save_cache(out_path, matrix, ids, protein_embed.DEFAULT_MODEL)

    norms = np.linalg.norm(matrix, axis=1)
    print(f"\nembedded {len(ids)} proteins to {dim}-d in {time.time() - t0:.0f}s")
    print(f"embedding L2 norm: min {norms.min():.3f}, median {np.median(norms):.3f}, "
          f"max {norms.max():.3f}; any NaN: {bool(np.isnan(matrix).any())}")
    print(f"Wrote {out_path} and {out_path.with_suffix('.json')}")

    stats = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    stats.update(
        {
            "n_sequences_over_window": int((lengths > protein_embed.MAX_WINDOW).sum()),
            "residue_length_min": int(lengths.min()),
            "residue_length_median": int(np.median(lengths)),
            "residue_length_max": int(lengths.max()),
            "embedding_norm_median": float(np.median(norms)),
            "device": device,
        }
    )
    (paths.RESULTS / "esm2_embedding_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {paths.RESULTS / 'esm2_embedding_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
