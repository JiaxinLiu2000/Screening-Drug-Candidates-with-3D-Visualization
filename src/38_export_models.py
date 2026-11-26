"""Build the committed `models/` directory from trained checkpoints.

Research checkpoints live under `data/processed/checkpoints/` and are not
committed: they are large, there are six of them, and four exist only to support
ablation tables. What a consumer of this project needs is a small set of
deployment models that work from a fresh clone with no training and no large
downloads.

This script produces that set:

    models/toxicity_gin.pt          12-endpoint toxicity, GIN trained from scratch
    models/multitask_gin.pt         shared encoder, both heads
    models/affinity_frozen_mlp.pt   the project's best affinity model
    models/protein_embeddings.npz   ESM-2 mean-pooled vectors for 229 proteins
    models/registry.json            file, sha256, architecture, metrics per entry

Each checkpoint is rewritten into a self-describing envelope - architecture,
task, endpoint order, feature contract, provenance and measured metrics - so
`tox21lib.model_io.load_model` can reconstruct it without consulting any
numbered driver script, and so a replacement model can be checked against the
same contract instead of being trusted.

The affinity model is not among the ablation checkpoints (src/27 keeps its best
weights in memory and reports metrics), so it is retrained here on the committed
seed-0 random split: ~3 minutes on GPU for the project's strongest affinity
result.
"""

from __future__ import annotations

import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import ablation_data, model_io, paths  # noqa: E402

SOURCE_CHECKPOINTS = paths.DATA_PROCESSED / "checkpoints"
OUT = paths.ROOT / "models"

#: Architecture constants the ablation used; stored with the weights so a loader
#: never has to assume them.
GIN_ARCH = {"out_dim": 256, "hidden_dim": 256, "n_layers": 4, "dropout": 0.1,
            "n_endpoints": 12}


def tox_metrics() -> dict:
    frame = pd.read_csv(paths.RESULTS / "shared_encoder_metrics.csv")
    row = frame[(frame["config"] == "A") & (frame["seed"] == 0)]
    if row.empty:
        raise SystemExit("FAILED: no config A seed-0 row in shared_encoder_metrics")
    return {"tox_macro_roc_auc_scaffold": float(row["tox_macro_roc_auc"].iloc[0]),
            "tox_macro_pr_auc_scaffold": float(row["tox_macro_pr_auc"].iloc[0]),
            "split": "scaffold", "seed": 0}


def multitask_metrics() -> dict:
    frame = pd.read_csv(paths.RESULTS / "shared_encoder_metrics.csv")
    row = frame[(frame["config"] == "C") & (frame["seed"] == 0)
                & (frame["affinity_split"] == "random")
                & (frame["loss_weight"] == 1.0)]
    if row.empty:
        raise SystemExit("FAILED: no config C seed-0 weight-1 random row")
    return {"tox_macro_roc_auc_scaffold": float(row["tox_macro_roc_auc"].iloc[0]),
            "affinity_pearson_r_random": float(row["affinity_pearson_r"].iloc[0]),
            "affinity_ci_random": float(row["affinity_ci"].iloc[0]),
            "seed": 0}


def convert_ablation(config: str, dest: str, task: str, metrics: dict) -> dict:
    src = SOURCE_CHECKPOINTS / f"config{config}_seed0.pt"
    if not src.exists():
        raise SystemExit(
            f"FAILED: {src} missing. Run src/20_shared_encoder_ablation.py first; "
            f"models/ is built from its checkpoints, not independently trained.")
    old = torch.load(src, map_location="cpu", weights_only=False)
    arch = dict(GIN_ARCH)
    arch["protein_dim"] = old.get("protein_dim")
    if old.get("n_endpoints") is None:
        arch["n_endpoints"] = None

    payload = {
        "format_version": model_io.FORMAT_VERSION,
        "task": task,
        "architecture": arch,
        "state_dict": old["state_dict"],
        "endpoints": list(ablation_data.TOX_ENDPOINTS),
        "metadata": {
            "source_checkpoint": f"data/processed/checkpoints/{src.name}",
            "ablation_config": config,
            "encoder": old["encoder"],
            "trained_on": "Tox21 scaffold split seed 0"
                          + (" + KIBA random split seed 0" if task == "multitask"
                             else ""),
            "metrics": metrics,
            "input_contract": "molecular graph from tox21lib.graphs.mol_to_graph"
                              + ("; raw (un-standardised) ESM-2 protein vectors, "
                                 "unlike affinity_frozen_mlp which standardises them"
                                 if task == "multitask" else ""),
        },
    }
    torch.save(payload, OUT / dest)
    return payload


def export_affinity(device: str) -> dict:
    """Retrain the frozen-embedding MLP on the committed seed-0 random split."""
    frozen = import_module("27_frozen_baseline")
    from tox21lib import train as trainlib

    trainlib.set_seed(0)
    tables = frozen.build_tables(device)
    subsets = frozen.subsets_for("random", 0, tables["n_pairs"])
    idx = {k: torch.tensor(np.flatnonzero(subsets == k), dtype=torch.long,
                           device=device) for k in ("train", "valid", "test")}

    model = frozen.ConcatMLP(tables["drug"].shape[1],
                             tables["protein"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=frozen.LR, weight_decay=1e-5)
    best, stale = {"pearson": -np.inf, "state": None}, 0
    t0 = time.time()
    for _ in range(frozen.EPOCHS):
        model.train()
        perm = idx["train"][torch.randperm(len(idx["train"]), device=device)]
        for start in range(0, len(perm), frozen.BATCH):
            chunk = perm[start:start + frozen.BATCH]
            opt.zero_grad(set_to_none=True)
            pred = model(tables["drug"][tables["drug_idx"][chunk]],
                         tables["protein"][tables["protein_idx"][chunk]])
            torch.nn.functional.mse_loss(pred, tables["y"][chunk]).backward()
            opt.step()
        valid = frozen.evaluate(model, tables, idx["valid"])
        if valid["pearson_r"] > best["pearson"] + 1e-6:
            best = {"pearson": valid["pearson_r"],
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= frozen.PATIENCE:
                break
    model.load_state_dict(best["state"])
    test = frozen.evaluate(model, tables, idx["test"])
    print(f"  retrained in {time.time() - t0:.0f}s; test Pearson "
          f"{test['pearson_r']:.4f}, CI {test['ci']:.4f}")

    reference = json.loads((paths.RESULTS / "frozen_baseline.json")
                           .read_text(encoding="utf-8"))
    expected = reference["comparison"]["random"]["frozen_mlp"]
    # Seed 0 alone is not the 5-seed mean, so a gap is expected; a large gap
    # means this export is not the model the report describes.
    if abs(test["pearson_r"] - expected) > 0.02:
        raise SystemExit(
            f"FAILED: exported affinity model scores {test['pearson_r']:.4f} but "
            f"results/frozen_baseline.json reports {expected:.4f} for this split. "
            f"Refusing to ship a model that does not reproduce the paper number.")

    # build_tables standardises the protein block column-wise before training.
    # That transform is part of the model's input contract, so it ships with the
    # weights; a consumer feeding raw ESM-2 vectors would otherwise get plausible
    # but badly wrong numbers (measured: Pearson 0.48 instead of 0.85).
    raw_matrix, _, _ = __import__("tox21lib.protein_embed", fromlist=["x"]).load_cache(
        __import__("tox21lib.protein_embed", fromlist=["x"]).cache_path(
            paths.DATA_PROCESSED))
    raw_matrix = raw_matrix.astype(np.float32)
    prot_mean = raw_matrix.mean(axis=0)
    prot_std = raw_matrix.std(axis=0) + 1e-6
    check = (raw_matrix - prot_mean) / prot_std
    if not np.allclose(check, tables["protein"].cpu().numpy(), atol=1e-5):
        raise SystemExit(
            "FAILED: recomputed protein standardisation does not match the table "
            "the model trained on; refusing to ship mismatched statistics.")

    payload = {
        "format_version": model_io.FORMAT_VERSION,
        "task": "affinity",
        "architecture": {
            "drug_dim": int(tables["drug"].shape[1]),
            "protein_dim": int(tables["protein"].shape[1]),
            "hidden": list(frozen.HIDDEN),
            "dropout": frozen.DROPOUT,
            "morgan_radius": 2,
        },
        "preprocessing": {
            "protein_standardisation": "column-wise (x - mean) / (std + 1e-6)",
            "protein_mean": prot_mean.tolist(),
            "protein_std": prot_std.tolist(),
        },
        "state_dict": best["state"],
        "metadata": {
            "trained_on": "KIBA random split seed 0 (splits/kiba_random_seed0.csv)",
            "protein_encoder": "ESM-2 esm2_t33_650M_UR50D, frozen, mean-pooled",
            "metrics": {"pearson_r": test["pearson_r"], "ci": test["ci"],
                        "rmse": test["rmse"],
                        "five_seed_mean_pearson": expected},
            "input_contract": "Morgan r=2 2048-bit fingerprint + standardised "
                              "1280-d ESM-2 vector",
            "units": "KIBA score as distributed by TDC (observed range 0.0-17.2, "
                     "median 11.6); higher is stronger binding. NOT pKd.",
        },
    }
    torch.save(payload, OUT / "affinity_frozen_mlp.pt")
    return payload


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}; writing {OUT}")

    entries: dict[str, dict] = {}

    print("toxicity_gin.pt  <- config A")
    tox = convert_ablation("A", "toxicity_gin.pt", "toxicity", tox_metrics())
    entries["toxicity_gin"] = {
        "file": "toxicity_gin.pt", "task": "toxicity",
        "description": "12-endpoint Tox21 classifier, GIN trained from scratch",
        "metrics": tox["metadata"]["metrics"]}

    print("multitask_gin.pt <- config C")
    mt = convert_ablation("C", "multitask_gin.pt", "multitask", multitask_metrics())
    entries["multitask_gin"] = {
        "file": "multitask_gin.pt", "task": "multitask",
        "description": "shared GIN encoder with toxicity and affinity heads",
        "metrics": mt["metadata"]["metrics"]}

    print("affinity_frozen_mlp.pt <- retraining on seed-0 random split")
    aff = export_affinity(device)
    entries["affinity_frozen_mlp"] = {
        "file": "affinity_frozen_mlp.pt", "task": "affinity",
        "description": "frozen ESM-2 650M + Morgan fingerprint -> MLP; the "
                       "project's strongest affinity model",
        "metrics": aff["metadata"]["metrics"]}

    # The embeddings are 1.1 MB but need a 2.5 GB model download to regenerate,
    # so they ship with the repository.
    src_npz = paths.DATA_PROCESSED / "esm2_t33_650M_UR50D.npz"
    if not src_npz.exists():
        raise SystemExit(f"FAILED: {src_npz} missing; run src/19_esm_embed.py")
    (OUT / "protein_embeddings.npz").write_bytes(src_npz.read_bytes())
    with np.load(OUT / "protein_embeddings.npz", allow_pickle=True) as handle:
        n_prot, dim = handle["embeddings"].shape

    registry = {
        "format_version": model_io.FORMAT_VERSION,
        "generated_by": "src/38_export_models.py",
        "models": {},
        "protein_embeddings": {
            "file": "protein_embeddings.npz",
            "model": "facebook/esm2_t33_650M_UR50D",
            "pooling": "mean over residues, last layer",
            "n_proteins": int(n_prot), "dim": int(dim),
            "sha256": model_io.sha256(OUT / "protein_embeddings.npz"),
        },
    }
    for name, entry in entries.items():
        path = OUT / entry["file"]
        entry["sha256"] = model_io.sha256(path)
        entry["bytes"] = path.stat().st_size
        registry["models"][name] = entry

    (OUT / "registry.json").write_text(json.dumps(registry, indent=2) + "\n",
                                       encoding="utf-8")
    total = sum(e["bytes"] for e in entries.values()) + \
        (OUT / "protein_embeddings.npz").stat().st_size
    print(f"\n{len(entries)} models + {n_prot} protein embeddings ({dim}-d), "
          f"{total / 1e6:.1f} MB total")
    print(f"Wrote {OUT / 'registry.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
