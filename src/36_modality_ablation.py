"""Phase 4 task E3 - which input is the affinity model actually using?

The cold-drug diagnosis found that a predictor knowing only *which protein* a
pair involves already reaches pooled Pearson 0.3665. That raises the obvious
follow-up: of the performance the full model reports, how much comes from the
molecule, how much from the protein, and how much from the two together?

A modality ablation answers it directly. The same frozen-embedding MLP is
trained four ways on identical folds:

    both              Morgan fingerprint + ESM-2 embedding   (the real model)
    drug_only         Morgan fingerprint, protein block zeroed
    protein_only      ESM-2 embedding, drug block zeroed
    shuffled_protein  Morgan + an ESM-2 embedding belonging to a DIFFERENT
                      protein, held fixed per protein

The last one is the control that makes the others interpretable. If
`shuffled_protein` scores as well as `both`, the protein channel is not carrying
protein-specific information - the model would merely be using it as an
arbitrary grouping key. Architecture, optimiser, epochs and seeds are identical
across the four, so the only thing varying is what the model is allowed to see.
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
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import dti, paths, train  # noqa: E402

MODES = ("both", "drug_only", "protein_only", "shuffled_protein")
SPLITS = ("random", "cold_drug", "cold_target")
SEEDS = (0, 1, 2)
EPOCHS, PATIENCE = 100, 10


def masked_tables(tables: dict, mode: str, seed: int, device: str) -> dict:
    """Return a copy of the feature tables with one modality removed or shuffled.

    Zeroing a block keeps the architecture and parameter count identical across
    modes, so a difference in score is a difference in available information and
    not in model capacity.
    """
    out = dict(tables)
    if mode == "drug_only":
        out["protein"] = torch.zeros_like(tables["protein"])
    elif mode == "protein_only":
        out["drug"] = torch.zeros_like(tables["drug"])
    elif mode == "shuffled_protein":
        # A derangement of the protein rows: every protein keeps a real, fixed
        # embedding, just not its own.
        n = tables["protein"].shape[0]
        g = torch.Generator().manual_seed(30_000 + seed)
        perm = torch.randperm(n, generator=g)
        fixed = perm == torch.arange(n)
        if fixed.any():  # rotate the fixed points so none maps to itself
            idx = torch.nonzero(fixed).flatten()
            perm[idx] = perm[idx.roll(1)]
        out["protein"] = tables["protein"][perm.to(device)]
    return out


def run_one(tables: dict, mode: str, split: str, seed: int, device: str) -> dict:
    from importlib import import_module

    frozen = import_module("27_frozen_baseline")
    train.set_seed(seed)
    view = masked_tables(tables, mode, seed, device)
    subsets = frozen.subsets_for(split, seed, tables["n_pairs"])
    idx = {name: torch.tensor(np.flatnonzero(subsets == name), dtype=torch.long,
                              device=device)
           for name in ("train", "valid", "test")}

    model = frozen.ConcatMLP(view["drug"].shape[1], view["protein"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=frozen.LR, weight_decay=1e-5)
    best = {"pearson": -np.inf, "state": None}
    stale = 0
    t0 = time.time()

    for _ in range(EPOCHS):
        model.train()
        perm = idx["train"][torch.randperm(len(idx["train"]), device=device)]
        for start in range(0, len(perm), frozen.BATCH):
            chunk = perm[start:start + frozen.BATCH]
            opt.zero_grad(set_to_none=True)
            pred = model(view["drug"][view["drug_idx"][chunk]],
                         view["protein"][view["protein_idx"][chunk]])
            nn.functional.mse_loss(pred, view["y"][chunk]).backward()
            opt.step()
        valid = frozen.evaluate(model, view, idx["valid"])
        if valid["pearson_r"] > best["pearson"] + 1e-6:
            best = {"pearson": valid["pearson_r"],
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best["state"])
    test = frozen.evaluate(model, view, idx["test"])
    frame = tables["pairs"].iloc[idx["test"].cpu().numpy()].copy()
    frame["y"], frame["pred"] = test["_y"], test["_pred"]
    within = [pearsonr(g["y"], g["pred"])[0] for _, g in frame.groupby("target_id")
              if len(g) >= 10 and g["y"].nunique() > 1]

    del model
    torch.cuda.empty_cache()
    return {
        "mode": mode, "split": split, "seed": seed,
        "pearson_r": test["pearson_r"], "rmse": test["rmse"], "ci": test["ci"],
        "within_protein_pearson": float(np.mean(within)) if within else float("nan"),
        "n_test": test["n"], "train_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    paths.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from importlib import import_module

    frozen = import_module("27_frozen_baseline")
    tables = frozen.build_tables(device)
    print(f"device {device}; {tables['n_pairs']} KIBA pairs; "
          f"{len(MODES)} modes x {len(SPLITS)} splits x {len(SEEDS)} seeds = "
          f"{len(MODES) * len(SPLITS) * len(SEEDS)} runs")

    rows = []
    for split in SPLITS:
        for mode in MODES:
            for seed in SEEDS:
                row = run_one(tables, mode, split, seed, device)
                rows.append(row)
            got = [r for r in rows if r["split"] == split and r["mode"] == mode]
            print(f"  {split:12s} {mode:17s} Pearson "
                  f"{np.mean([r['pearson_r'] for r in got]):.4f}  within-protein "
                  f"{np.nanmean([r['within_protein_pearson'] for r in got]):.4f}")

    frame = pd.DataFrame(rows)
    frame.to_csv(paths.RESULTS / "modality_ablation.csv", index=False)

    agg = frame.groupby(["split", "mode"])[
        ["pearson_r", "within_protein_pearson", "ci", "rmse"]].agg(["mean", "std"])
    print("\n" + agg.round(4).to_string())

    stats = {"modes": list(MODES), "splits": list(SPLITS), "seeds": list(SEEDS),
             "by_split": {}}
    for split in SPLITS:
        sub = frame[frame["split"] == split]
        means = sub.groupby("mode")["pearson_r"].mean()
        both = float(means["both"])
        stats["by_split"][split] = {
            mode: {
                "pearson_mean": float(means[mode]),
                "pearson_std": float(sub[sub["mode"] == mode]["pearson_r"].std(ddof=1)),
                "within_protein": float(
                    sub[sub["mode"] == mode]["within_protein_pearson"].mean()),
                "share_of_full_model": float(means[mode] / both) if both else float("nan"),
            }
            for mode in MODES
        }
        # The control: if shuffling the protein identity costs nothing, the
        # protein channel was not supplying protein-specific information.
        stats["by_split"][split]["shuffle_cost"] = float(
            means["both"] - means["shuffled_protein"])
    (paths.RESULTS / "modality_ablation.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {paths.RESULTS / 'modality_ablation.csv'} and "
          f"{paths.RESULTS / 'modality_ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
