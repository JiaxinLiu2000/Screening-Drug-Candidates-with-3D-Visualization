"""Phase 4 task B2 - the published frozen-embedding baseline, on OUR splits.

Otter-Knowledge (arXiv 2306.12802v3, Table 3) reports KIBA Pearson 0.852 on a
random split from a deliberately plain model: frozen protein embeddings and
Morgan fingerprints concatenated into an MLP head. Our forest reaches 0.7841 and
our best trained encoder 0.7563 on the same dataset, which is why Phase 3
concluded "no neural configuration beats the random-forest floor".

That conclusion has two possible causes and they need separating:

    (a) multi-task learning / neural encoders genuinely do not help here, or
    (b) our encoder and head are under-built.

This script builds exactly Otter's baseline shape - frozen ESM-2 650M protein
embedding (already cached) + Morgan 2048 drug fingerprint -> concat -> MLP - and
evaluates it on the committed KIBA splits with the same five seeds. Nothing is
trained end to end: if a model with NO learned molecular encoder beats our GIN,
the Phase 3 conclusion is about our implementation, not about neural methods.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import datasets, dti, featurize, paths, protein_embed, train  # noqa: E402
from tox21lib.models import ConcatMLP  # noqa: E402

SPLITS = ("random", "cold_drug", "cold_target")
EPOCHS = 100
PATIENCE = 10
BATCH = 1024
LR = 1e-3
HIDDEN = (1024, 512)
DROPOUT = 0.1


def build_tables(device: str):
    """Per-entity feature matrices on the device, plus the pair index arrays."""
    raw = datasets.load_dti("KIBA")
    clean, _ = datasets.unique_molecules(raw)
    pairs, _ = dti.build_pairs(raw, clean)

    matrix, protein_rows, model_name = protein_embed.load_cache(
        protein_embed.cache_path(paths.DATA_PROCESSED)
    )
    smiles_by_key = dict(zip(clean["inchikey"], clean["canonical_smiles"]))
    drug_keys = sorted(pairs["inchikey"].unique())
    drug_fp = featurize.featurize_smiles([smiles_by_key[k] for k in drug_keys])
    drug_row = {k: i for i, k in enumerate(drug_keys)}

    # Standardise the protein embedding; ESM-2 features are not zero-centred and
    # an un-normalised 1280-d block otherwise dominates the 2048 binary bits.
    prot = matrix.astype(np.float32)
    prot = (prot - prot.mean(axis=0)) / (prot.std(axis=0) + 1e-6)

    return {
        "drug": torch.from_numpy(drug_fp.astype(np.float32)).to(device),
        "protein": torch.from_numpy(prot).to(device),
        "drug_idx": torch.tensor([drug_row[k] for k in pairs["inchikey"]],
                                 dtype=torch.long, device=device),
        "protein_idx": torch.tensor([protein_rows[t] for t in pairs["target_id"]],
                                    dtype=torch.long, device=device),
        "y": torch.tensor(pairs["affinity"].to_numpy(dtype=np.float32), device=device),
        "n_pairs": len(pairs),
        "protein_model": model_name,
        "pairs": pairs,
    }


def subsets_for(split: str, seed: int, n_pairs: int) -> np.ndarray:
    frame = pd.read_csv(paths.SPLITS / f"kiba_{split}_seed{seed}.csv")
    if len(frame) != n_pairs:
        raise RuntimeError(f"kiba_{split}_seed{seed}.csv has {len(frame)} rows, "
                           f"expected {n_pairs}")
    return frame.sort_values("pair_index")["subset"].to_numpy()


def evaluate(model, tables, idx: torch.Tensor) -> dict:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(idx), 8192):
            chunk = idx[start:start + 8192]
            preds.append(model(tables["drug"][tables["drug_idx"][chunk]],
                               tables["protein"][tables["protein_idx"][chunk]]))
    pred = torch.cat(preds).float().cpu().numpy()
    y = tables["y"][idx].float().cpu().numpy()
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mse": float(np.mean((pred - y) ** 2)),
        "pearson_r": float(pearsonr(y, pred)[0]),
        "spearman_rho": float(spearmanr(y, pred)[0]),
        "ci": float(dti.concordance_index(y, pred)),
        "_pred": pred,
        "_y": y,
    }


def run_one(tables, split: str, seed: int, device: str) -> dict:
    train.set_seed(seed)
    subsets = subsets_for(split, seed, tables["n_pairs"])
    idx = {name: torch.tensor(np.flatnonzero(subsets == name), dtype=torch.long,
                              device=device)
           for name in ("train", "valid", "test")}

    model = ConcatMLP(tables["drug"].shape[1], tables["protein"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    best = {"pearson": -np.inf, "epoch": -1, "state": None}
    stale = 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        perm = idx["train"][torch.randperm(len(idx["train"]), device=device)]
        for start in range(0, len(perm), BATCH):
            chunk = perm[start:start + BATCH]
            opt.zero_grad(set_to_none=True)
            pred = model(tables["drug"][tables["drug_idx"][chunk]],
                         tables["protein"][tables["protein_idx"][chunk]])
            loss = nn.functional.mse_loss(pred, tables["y"][chunk])
            loss.backward()
            opt.step()

        valid = evaluate(model, tables, idx["valid"])
        if valid["pearson_r"] > best["pearson"] + 1e-6:
            best = {"pearson": valid["pearson_r"], "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best["state"])
    test = evaluate(model, tables, idx["test"])

    # Within-protein correlation: the metric Phase 4 established as the honest one
    # for screening, where molecules are ranked against a single target.
    frame = tables["pairs"].iloc[idx["test"].cpu().numpy()].copy()
    frame["y"], frame["pred"] = test["_y"], test["_pred"]
    per_protein = [
        pearsonr(g["y"], g["pred"])[0]
        for _, g in frame.groupby("target_id")
        if len(g) >= 10 and g["y"].nunique() > 1
    ]

    return {
        "model": "frozen ESM-2 650M + Morgan 2048 -> concat MLP",
        "split": split,
        "seed": seed,
        "epochs_run": best["epoch"] + 1 + stale,
        "best_epoch": best["epoch"],
        "n_test": test["n"],
        "rmse": test["rmse"],
        "mse": test["mse"],
        "pearson_r": test["pearson_r"],
        "spearman_rho": test["spearman_rho"],
        "ci": test["ci"],
        "within_protein_pearson": float(np.mean(per_protein)) if per_protein else float("nan"),
        "n_proteins_scored": len(per_protein),
        "train_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    paths.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    tables = build_tables(device)
    print(f"KIBA {tables['n_pairs']} pairs, drug {tuple(tables['drug'].shape)}, "
          f"protein {tuple(tables['protein'].shape)} ({tables['protein_model']})")

    rows = []
    for split in SPLITS:
        for seed in paths.SEEDS:
            row = run_one(tables, split, seed, device)
            rows.append(row)
            print(f"  {split:12s} seed={seed}  Pearson {row['pearson_r']:.4f}  "
                  f"CI {row['ci']:.4f}  MSE {row['mse']:.4f}  "
                  f"within-protein r {row['within_protein_pearson']:.4f}  "
                  f"[{row['train_seconds']:.0f}s, {row['epochs_run']} epochs]")

    frame = pd.DataFrame(rows)
    frame.to_csv(paths.RESULTS / "frozen_baseline_metrics.csv", index=False)

    agg = frame.groupby("split").agg(
        pearson_mean=("pearson_r", "mean"), pearson_std=("pearson_r", "std"),
        ci_mean=("ci", "mean"), mse_mean=("mse", "mean"),
        within_protein_mean=("within_protein_pearson", "mean"),
    ).reset_index()
    print("\n" + agg.round(4).to_string(index=False))

    rf = pd.read_csv(paths.RESULTS / "binding_baseline.csv")
    rf = rf[rf["dataset"] == "KIBA"].groupby("split")["pearson_r"].mean()
    neural = pd.read_csv(paths.RESULTS / "shared_encoder_metrics.csv")
    neural = neural[(neural["loss_weight"].isna()) | (neural["loss_weight"] == 1.0)]
    neural = neural.groupby("affinity_split")["affinity_pearson_r"].mean()

    comparison = {
        row["split"]: {
            "frozen_mlp": float(row["pearson_mean"]),
            "frozen_mlp_std": float(row["pearson_std"]),
            "random_forest": float(rf.get(row["split"], np.nan)),
            "best_trained_encoder": float(neural.get(row["split"], np.nan)),
            "within_protein_frozen_mlp": float(row["within_protein_mean"]),
        }
        for _, row in agg.iterrows()
    }
    (paths.RESULTS / "frozen_baseline.json").write_text(
        json.dumps({"comparison": comparison,
                    "published_otter_kiba_random_pearson": 0.852,
                    "n_runs": len(frame),
                    "protein_model": tables["protein_model"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {paths.RESULTS / 'frozen_baseline_metrics.csv'} and "
          f"{paths.RESULTS / 'frozen_baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
