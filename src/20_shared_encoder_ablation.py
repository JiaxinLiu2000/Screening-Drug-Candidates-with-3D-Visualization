"""Phase 3 Task 5 - the four-way shared-encoder ablation.

    A  GIN from scratch, toxicity only
    B  GIN from scratch, affinity only
    C  GIN from scratch, both tasks, shared encoder
    D  ChemBERTa fine-tuned, both tasks, shared encoder

Toxicity is evaluated on the committed Tox21 scaffold split; affinity on the
committed KIBA random / cold_drug / cold_target splits - the same folds the
random forests used, so the RF numbers are a like-for-like floor.

    python src/20_shared_encoder_ablation.py --smoke          # 1 seed, 2 epochs
    python src/20_shared_encoder_ablation.py --stage main
    python src/20_shared_encoder_ablation.py --stage sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import ablation_data as adata  # noqa: E402
from tox21lib import models, paths, protein_embed, train  # noqa: E402

AFFINITY_SPLITS = ("random", "cold_drug", "cold_target")
LOSS_WEIGHTS = (0.1, 0.5, 1.0, 2.0, 5.0)
TOX_SPLIT = "scaffold"
METRICS_CSV = paths.RESULTS / "shared_encoder_metrics.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("main", "sweep", "controls", "all"),
                   default="all")
    p.add_argument("--smoke", action="store_true",
                   help="1 seed, 2 epochs, for wiring checks only")
    p.add_argument("--epochs", type=int, default=1000,
                   help="upper bound; the per-config BUDGET normally governs")
    p.add_argument("--sweep-seeds", type=int, default=2,
                   help="seeds per loss-weight cell (the main comparison always uses 5)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--no-checkpoints", action="store_true")
    return p.parse_args()


#: Reference checkpoints kept for the attribution stage (Task 6). Weights are
#: large and derivable, so they live under the gitignored processed-data tree.
CHECKPOINT_DIR = paths.DATA_PROCESSED / "checkpoints"


def checkpoint_name(config: str, seed: int, affinity_split: str | None,
                    loss_weight: float) -> str | None:
    """Only the canonical cells are saved: seed 0, loss weight 1, random split."""
    if seed != 0 or loss_weight != 1.0:
        return None
    if affinity_split not in (None, "random"):
        return None
    return f"config{config}_seed0.pt"

#: epochs and early-stopping patience per configuration. The GIN configurations
#: cost ~0.5 s/epoch (toxicity) to ~9 s/epoch (with affinity) and converge slowly
#: from scratch; ChemBERTa costs ~25 s/epoch and starts from pretrained weights.
#: Budgets differ accordingly, and the epochs actually run are recorded per cell.
BUDGET = {
    "A": (150, 15),
    "B": (60, 10),
    "C": (60, 10),
    "D": (25, 5),
    "E": (25, 5),
    "F": (25, 5),
}


def build_encoder(config: str, out_dim: int = 256):
    if config in ("D", "E", "F"):
        return models.ChemBertaEncoder(out_dim=out_dim)
    return models.GINEncoder(out_dim=out_dim)


def run_one(
    config: str,
    *,
    seed: int,
    affinity_split: str | None,
    loss_weight: float,
    bundle: dict,
    device: str,
    epochs: int,
    batch_size: int,
) -> dict:
    """Train and evaluate one (config, seed, split, loss weight) cell."""
    train.set_seed(seed)
    # E and F are pretrained single-task controls: without them, configuration D
    # changes two things at once (pretrained encoder AND task sharing) and the
    # "helps only when pretrained" hypothesis cannot be separated from "pretraining
    # helps, sharing is irrelevant".
    needs_tox = config in ("A", "C", "D", "E")
    needs_affinity = config in ("B", "C", "D", "F")
    use_tokens = config in ("D", "E", "F")

    tox_loaders = None
    if needs_tox:
        subsets = adata.tox21_subsets(TOX_SPLIT, seed)
        tox = adata.loaders_for(
            bundle["tox_tokens"] if use_tokens else bundle["tox_graphs"],
            subsets, batch_size=batch_size, seed=seed,
        )
        tox_loaders = (tox["train"], tox["valid"])

    affinity_loaders = None
    if needs_affinity:
        subsets = adata.kiba_subsets(affinity_split, seed, bundle["n_pairs"])
        aff = adata.loaders_for(
            bundle["kiba_tokens"] if use_tokens else bundle["kiba_graphs"],
            subsets, batch_size=batch_size, seed=seed,
        )
        affinity_loaders = (aff["train"], aff["valid"])

    model = models.MultiTaskModel(
        build_encoder(config),
        n_endpoints=12 if needs_tox else None,
        protein_dim=bundle["protein_dim"] if needs_affinity else None,
    )
    budget_epochs, patience = BUDGET[config]
    cfg = train.TrainConfig(
        epochs=min(epochs, budget_epochs) if epochs < 1000 else budget_epochs,
        patience=patience,
        batch_size=batch_size,
        loss_weight=loss_weight,
        encoder_lr=2e-5 if config in ("D", "E", "F") else None,
    )

    t0 = time.time()
    outcome = train.train_model(
        model,
        tox_loaders=tox_loaders,
        affinity_loaders=affinity_loaders,
        protein_emb=bundle["protein_emb"],
        cfg=cfg,
        device=device,
        pos_weight=bundle["pos_weight"].to(device) if needs_tox else None,
        verbose=False,
    )
    elapsed = time.time() - t0

    row = {
        "config": config,
        "encoder": "ChemBERTa-77M (fine-tuned)" if config in ("D", "E", "F")
                   else "GIN (scratch)",
        "tasks": {"A": "toxicity", "B": "affinity", "E": "toxicity",
                  "F": "affinity"}.get(config, "toxicity+affinity"),
        "seed": seed,
        "loss_weight": loss_weight if needs_affinity and needs_tox else float("nan"),
        "affinity_split": affinity_split if needs_affinity else "",
        "n_parameters": models.count_parameters(model),
        "epochs_run": outcome["epochs_run"],
        "best_epoch": outcome["best_epoch"],
        "train_seconds": round(elapsed, 1),
    }

    if needs_tox:
        subsets = adata.tox21_subsets(TOX_SPLIT, seed)
        tox = adata.loaders_for(
            bundle["tox_tokens"] if use_tokens else bundle["tox_graphs"],
            subsets, batch_size=batch_size, seed=seed, shuffle_train=False,
        )
        test_tox = train.evaluate_toxicity(model, tox["test"], device)
        row["tox_macro_roc_auc"] = test_tox["macro_roc_auc"]
        row["tox_macro_pr_auc"] = test_tox["macro_pr_auc"]
        row["tox_per_endpoint"] = json.dumps(test_tox["per_endpoint"])
    if needs_affinity:
        subsets = adata.kiba_subsets(affinity_split, seed, bundle["n_pairs"])
        aff = adata.loaders_for(
            bundle["kiba_tokens"] if use_tokens else bundle["kiba_graphs"],
            subsets, batch_size=batch_size, seed=seed, shuffle_train=False,
        )
        test_aff = train.evaluate_affinity(model, aff["test"], bundle["protein_emb"], device)
        row.update(
            {
                "affinity_n_test": test_aff["n"],
                "affinity_rmse": test_aff["rmse"],
                "affinity_pearson_r": test_aff["pearson_r"],
                "affinity_spearman_rho": test_aff["spearman_rho"],
                "affinity_ci": test_aff["ci"],
            }
        )

    name = checkpoint_name(config, seed, affinity_split, loss_weight)
    if name and not bundle.get("no_checkpoints"):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": config,
                "state_dict": model.state_dict(),
                "encoder": row["encoder"],
                "n_endpoints": 12 if needs_tox else None,
                "protein_dim": bundle["protein_dim"] if needs_affinity else None,
                "tox_split": TOX_SPLIT,
                "seed": seed,
            },
            CHECKPOINT_DIR / name,
        )
        row["checkpoint"] = f"data/processed/checkpoints/{name}"

    del model
    torch.cuda.empty_cache()
    return row


def load_bundle(device: str, need_tokens: bool) -> dict:
    """Graphs, labels, splits and frozen protein embeddings, built once."""
    print("Building molecular graphs...")
    t0 = time.time()
    tox_graphs, clean = adata.tox21_graph_dataset()
    pairs, kiba_clean = adata.kiba_pairs()

    cache = protein_embed.cache_path(paths.DATA_PROCESSED)
    matrix, protein_rows, model_name = protein_embed.load_cache(cache)
    protein_emb = torch.from_numpy(matrix).to(device)

    kiba_graphs = adata.kiba_graph_dataset(pairs, kiba_clean, protein_rows)
    print(f"  Tox21 {len(tox_graphs)} graphs, KIBA {len(kiba_graphs)} pairs "
          f"({pairs['inchikey'].nunique()} distinct drugs, "
          f"{len(protein_rows)} proteins, {model_name} {matrix.shape[1]}-d) "
          f"in {time.time() - t0:.0f}s")

    bundle = {
        "tox_graphs": tox_graphs,
        "kiba_graphs": kiba_graphs,
        "n_pairs": len(kiba_graphs),
        "protein_emb": protein_emb,
        "protein_dim": matrix.shape[1],
        "pos_weight": adata.positive_weights(clean),
        "tox_tokens": None,
        "kiba_tokens": None,
        "n_truncated": 0,
    }

    if need_tokens:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(models.CHEMBERTA_CHECKPOINT)
        smiles_by_key = dict(zip(kiba_clean["inchikey"], kiba_clean["canonical_smiles"]))
        n1 = adata.attach_tokens(tox_graphs, list(clean["canonical_smiles"]), tokenizer)
        n2 = adata.attach_tokens(
            kiba_graphs, [smiles_by_key[k] for k in pairs["inchikey"]], tokenizer
        )
        # The same Data objects carry both graph and token fields; configuration D
        # reads the tokens, A-C read the graph, so the sample sets are identical.
        bundle["tox_tokens"] = tox_graphs
        bundle["kiba_tokens"] = kiba_graphs
        bundle["n_truncated"] = int(n1 + n2)
        print(f"  tokenized SMILES for ChemBERTa "
              f"(max {adata.SMILES_MAX_TOKENS} tokens, {n1 + n2} truncated)")
    return bundle


def main() -> int:
    args = parse_args()
    paths.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = (0,) if args.smoke else paths.SEEDS
    epochs = 2 if args.smoke else args.epochs
    sweep_seeds = seeds[: 1 if args.smoke else args.sweep_seeds]
    print(f"device {device}; seeds {list(seeds)}; epochs {epochs}; "
          f"sweep seeds {list(sweep_seeds)}")

    bundle = load_bundle(device, need_tokens=True)
    bundle["no_checkpoints"] = args.no_checkpoints

    jobs: list[dict] = []
    if args.stage in ("main", "all"):
        for seed in seeds:
            jobs.append({"config": "A", "seed": seed, "affinity_split": None,
                         "loss_weight": 1.0})
        for config in ("B", "C", "D"):
            for split in AFFINITY_SPLITS:
                for seed in seeds:
                    jobs.append({"config": config, "seed": seed,
                                 "affinity_split": split, "loss_weight": 1.0})
    if args.stage in ("controls", "all"):
        # Pretrained single-task controls on the random affinity split, so D can be
        # read against a pretrained encoder that does NOT share tasks.
        for config in ("E", "F"):
            for seed in seeds:
                jobs.append({"config": config, "seed": seed,
                             "affinity_split": "random" if config == "F" else None,
                             "loss_weight": 1.0})
    if args.stage in ("sweep", "all"):
        for config in ("C", "D"):
            for weight in LOSS_WEIGHTS:
                if weight == 1.0 and args.stage == "all":
                    continue  # the ratio-1 cell is already in the main comparison
                for seed in sweep_seeds:
                    jobs.append({"config": config, "seed": seed,
                                 "affinity_split": "random", "loss_weight": weight})

    print(f"\n{len(jobs)} runs queued\n")
    rows: list[dict] = []
    t_start = time.time()
    for i, job in enumerate(jobs, 1):
        label = (f"{job['config']} seed={job['seed']} "
                 f"split={job['affinity_split'] or TOX_SPLIT} w={job['loss_weight']}")
        print(f"[{i}/{len(jobs)}] {label}")
        row = run_one(
            job["config"],
            seed=job["seed"],
            affinity_split=job["affinity_split"],
            loss_weight=job["loss_weight"],
            bundle=bundle,
            device=device,
            epochs=epochs,
            batch_size=args.batch_size,
        )
        rows.append(row)
        summary = []
        if "tox_macro_roc_auc" in row:
            summary.append(f"tox ROC-AUC {row['tox_macro_roc_auc']:.4f}")
        if "affinity_pearson_r" in row:
            summary.append(f"affinity r {row['affinity_pearson_r']:.4f} "
                           f"RMSE {row['affinity_rmse']:.4f}")
        print(f"        {'; '.join(summary)}  [{row['train_seconds']:.0f}s, "
              f"{row['epochs_run']} epochs]  total {time.time() - t_start:.0f}s")

        frame = pd.DataFrame(rows)
        out = METRICS_CSV if not args.smoke else METRICS_CSV.with_name(
            "shared_encoder_metrics_smoke.csv")
        frame.to_csv(out, index=False)

    print(f"\nWrote {out} ({len(rows)} rows) in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
