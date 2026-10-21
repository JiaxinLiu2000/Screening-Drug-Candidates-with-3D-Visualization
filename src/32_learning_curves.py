"""Phase 4 task E1 - learning curves: is our negative result about data or models?

Phases 3-4 established that on this data a random forest on Morgan fingerprints
beats a GIN trained from scratch (Tox21), and that a frozen-embedding MLP beats
the forest on affinity (KIBA). Both are statements at ONE training-set size. The
machine-learning question underneath them is a different one:

    at what training-set size, if any, does the learned representation overtake
    the fixed one?

That is what a learning curve answers. The training fold is subsampled to
{5, 10, 25, 50, 100}% while the validation and test folds are held FIXED, so the
only thing varying is how much the model got to learn from. Where the curves
cross - or whether they cross at all, and whether the gap is still shrinking at
100% - says whether "collect more data" would have changed the conclusion.

Two settings, chosen because they are the two places the earlier results
disagree with each other:

    Tox21 scaffold split   RF (Morgan 2048)  vs  GIN trained from scratch
    KIBA random split      RF (Morgan + AAC) vs  frozen ESM-2 + Morgan -> MLP
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import ablation_data as adata  # noqa: E402
from tox21lib import datasets, dti, featurize, models, paths, protein_embed, train  # noqa: E402

FRACTIONS = (0.05, 0.1, 0.25, 0.5, 1.0)
CURVE_SEEDS = (0, 1, 2)
GIN_EPOCHS, GIN_PATIENCE = 150, 15
MLP_EPOCHS, MLP_PATIENCE = 100, 10


def subsample_train(subsets: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Keep a seeded random `fraction` of the train fold; valid/test untouched.

    Dropped rows are relabelled "unused" rather than deleted, so every loader and
    index stays aligned with the committed split files.
    """
    out = subsets.copy()
    train_idx = np.flatnonzero(subsets == "train")
    if fraction >= 1.0:
        return out
    rng = np.random.default_rng(10_000 + seed)
    keep = rng.choice(train_idx, size=max(1, int(round(len(train_idx) * fraction))),
                      replace=False)
    drop = np.setdiff1d(train_idx, keep)
    out[drop] = "unused"
    return out


# --------------------------------------------------------------------------
# Tox21: random forest vs GIN
# --------------------------------------------------------------------------

def tox21_rf(X, clean, endpoints, subsets, seed) -> dict:
    tr, te = subsets == "train", subsets == "test"
    rocs, prs = [], []
    for ep in endpoints:
        y = clean[ep].to_numpy(dtype=float)
        labeled = ~np.isnan(y)
        a, b = tr & labeled, te & labeled
        if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
            continue
        model = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                       n_jobs=-1, random_state=seed)
        model.fit(X[a], y[a].astype(int))
        score = model.predict_proba(X[b])[:, 1]
        rocs.append(roc_auc_score(y[b].astype(int), score))
        prs.append(average_precision_score(y[b].astype(int), score))
    return {"macro_roc_auc": float(np.mean(rocs)), "macro_pr_auc": float(np.mean(prs)),
            "n_endpoints_scored": len(rocs)}


def tox21_gin(graphs, pos_weight, subsets, seed, device) -> dict:
    train.set_seed(seed)
    loaders = adata.loaders_for(graphs, subsets, batch_size=512, seed=seed)
    model = models.MultiTaskModel(models.GINEncoder(), n_endpoints=12, protein_dim=None)
    cfg = train.TrainConfig(epochs=GIN_EPOCHS, batch_size=512,
                            patience=GIN_PATIENCE)
    outcome = train.train_model(
        model, tox_loaders=(loaders["train"], loaders["valid"]),
        affinity_loaders=None, protein_emb=None, cfg=cfg, device=device,
        pos_weight=pos_weight.to(device), verbose=False,
    )
    eval_loaders = adata.loaders_for(graphs, subsets, batch_size=512, seed=seed,
                                     shuffle_train=False)
    test = train.evaluate_toxicity(model, eval_loaders["test"], device)
    del model
    torch.cuda.empty_cache()
    return {"macro_roc_auc": test["macro_roc_auc"], "macro_pr_auc": test["macro_pr_auc"],
            "epochs_run": outcome["epochs_run"]}


# --------------------------------------------------------------------------
# KIBA: random forest vs frozen-embedding MLP
# --------------------------------------------------------------------------

def kiba_rf(X, y, subsets, seed) -> dict:
    tr, te = subsets == "train", subsets == "test"
    model = RandomForestRegressor(n_estimators=100, max_features="sqrt",
                                  random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
    pred = model.predict(X[te])
    return {"pearson_r": float(pearsonr(y[te], pred)[0]),
            "rmse": float(np.sqrt(np.mean((pred - y[te]) ** 2))),
            "ci": float(dti.concordance_index(y[te], pred))}


def kiba_mlp(tables, subsets, seed, device) -> dict:
    import torch.nn as nn

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module

    frozen = import_module("27_frozen_baseline")
    train.set_seed(seed)
    idx = {name: torch.tensor(np.flatnonzero(subsets == name), dtype=torch.long,
                              device=device)
           for name in ("train", "valid", "test")}
    model = frozen.ConcatMLP(tables["drug"].shape[1], tables["protein"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=frozen.LR, weight_decay=1e-5)
    best = {"pearson": -np.inf, "state": None}
    stale = 0
    for _ in range(MLP_EPOCHS):
        model.train()
        perm = idx["train"][torch.randperm(len(idx["train"]), device=device)]
        for start in range(0, len(perm), frozen.BATCH):
            chunk = perm[start:start + frozen.BATCH]
            opt.zero_grad(set_to_none=True)
            pred = model(tables["drug"][tables["drug_idx"][chunk]],
                         tables["protein"][tables["protein_idx"][chunk]])
            nn.functional.mse_loss(pred, tables["y"][chunk]).backward()
            opt.step()
        valid = frozen.evaluate(model, tables, idx["valid"])
        if valid["pearson_r"] > best["pearson"] + 1e-6:
            best = {"pearson": valid["pearson_r"],
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= MLP_PATIENCE:
                break
    model.load_state_dict(best["state"])
    test = frozen.evaluate(model, tables, idx["test"])
    del model
    torch.cuda.empty_cache()
    return {"pearson_r": test["pearson_r"], "rmse": test["rmse"], "ci": test["ci"]}


def main() -> int:
    paths.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}; fractions {FRACTIONS}; seeds {CURVE_SEEDS}")
    rows: list[dict] = []
    t_start = time.time()

    # ---- Tox21 ----------------------------------------------------------
    clean = pd.read_parquet(paths.CLEAN_PARQUET)
    endpoints = adata.TOX_ENDPOINTS
    X_tox = featurize.featurize_smiles(clean["canonical_smiles"])
    graphs, _ = adata.tox21_graph_dataset()
    pos_weight = adata.positive_weights(clean)

    for seed in CURVE_SEEDS:
        base = adata.tox21_subsets("scaffold", seed)
        n_train_full = int((base == "train").sum())
        for frac in FRACTIONS:
            subsets = subsample_train(base, frac, seed)
            n_train = int((subsets == "train").sum())
            rf = tox21_rf(X_tox, clean, endpoints, subsets, seed)
            gin = tox21_gin(graphs, pos_weight, subsets, seed, device)
            for name, res in (("random_forest", rf), ("gin_scratch", gin)):
                rows.append({"task": "tox21_scaffold", "model": name, "seed": seed,
                             "fraction": frac, "n_train": n_train,
                             "n_train_full": n_train_full,
                             "metric": "macro_roc_auc",
                             "value": res["macro_roc_auc"],
                             "secondary": res["macro_pr_auc"]})
            print(f"  tox21 seed={seed} frac={frac:<5} n={n_train:5d}  "
                  f"RF {rf['macro_roc_auc']:.4f}   GIN {gin['macro_roc_auc']:.4f}  "
                  f"[{time.time() - t_start:.0f}s]")

    # ---- KIBA -----------------------------------------------------------
    raw = datasets.load_dti("KIBA")
    kclean, _ = datasets.unique_molecules(raw)
    pairs, _ = dti.build_pairs(raw, kclean)
    y_kiba = pairs["affinity"].to_numpy(dtype=float)
    X_kiba = dti.feature_matrix(
        pairs, dict(zip(kclean["inchikey"], kclean["canonical_smiles"])))

    from importlib import import_module
    frozen = import_module("27_frozen_baseline")
    tables = frozen.build_tables(device)

    for seed in CURVE_SEEDS:
        base = frozen.subsets_for("random", seed, len(pairs))
        n_train_full = int((base == "train").sum())
        for frac in FRACTIONS:
            subsets = subsample_train(base, frac, seed)
            n_train = int((subsets == "train").sum())
            rf = kiba_rf(X_kiba, y_kiba, subsets, seed)
            mlp = kiba_mlp(tables, subsets, seed, device)
            for name, res in (("random_forest", rf), ("frozen_esm_mlp", mlp)):
                rows.append({"task": "kiba_random", "model": name, "seed": seed,
                             "fraction": frac, "n_train": n_train,
                             "n_train_full": n_train_full,
                             "metric": "pearson_r", "value": res["pearson_r"],
                             "secondary": res["ci"]})
            print(f"  kiba  seed={seed} frac={frac:<5} n={n_train:6d}  "
                  f"RF {rf['pearson_r']:.4f}   MLP {mlp['pearson_r']:.4f}  "
                  f"[{time.time() - t_start:.0f}s]")

    frame = pd.DataFrame(rows)
    frame.to_csv(paths.RESULTS / "learning_curves.csv", index=False)
    summarise(frame)
    print(f"\nWrote {paths.RESULTS / 'learning_curves.csv'} "
          f"({len(frame)} rows, {time.time() - t_start:.0f}s)")
    return 0


def summarise(frame: pd.DataFrame) -> None:
    """Aggregate, find the crossover, plot, and write the json sibling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = (frame.groupby(["task", "model", "fraction", "n_train"])["value"]
           .agg(["mean", "std"]).reset_index())
    stats: dict = {"fractions": list(FRACTIONS), "seeds": list(CURVE_SEEDS), "tasks": {}}

    for task, label, ylabel in (
        ("tox21_scaffold", "Tox21, scaffold split", "macro ROC-AUC"),
        ("kiba_random", "KIBA, random split", "Pearson r"),
    ):
        sub = agg[agg["task"] == task]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
        curves = {}
        for model, g in sub.groupby("model"):
            g = g.sort_values("n_train")
            curves[model] = g
            ax.errorbar(g["n_train"], g["mean"], yerr=g["std"].fillna(0), marker="o",
                        capsize=3, label=model.replace("_", " "))
        ax.set_xscale("log")
        ax.set_xlabel("training examples (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Learning curve - {label}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(paths.FIGURES / f"learning_curve_{task}.png")
        plt.close(fig)

        names = sorted(curves)
        a, b = curves[names[0]], curves[names[1]]
        gaps = (a.set_index("fraction")["mean"] - b.set_index("fraction")["mean"])
        crossover = None
        signs = np.sign(gaps.to_numpy())
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1] and signs[i] != 0:
                crossover = float(gaps.index[i])
                break
        stats["tasks"][task] = {
            "metric": ylabel,
            "models": names,
            "by_fraction": {
                str(f): {m: {"mean": float(curves[m].set_index("fraction").loc[f, "mean"]),
                             "std": float(curves[m].set_index("fraction").loc[f, "std"]),
                             "n_train": int(curves[m].set_index("fraction").loc[f, "n_train"])}
                         for m in names}
                for f in FRACTIONS
            },
            "gap_first_minus_second": {str(k): float(v) for k, v in gaps.items()},
            "crossover_fraction": crossover,
            "gap_at_5pct": float(gaps.iloc[0]),
            "gap_at_100pct": float(gaps.iloc[-1]),
            "gap_narrowing": bool(abs(gaps.iloc[-1]) < abs(gaps.iloc[0])),
        }
        print(f"\n{label}: gap ({names[0]} - {names[1]}) "
              f"{gaps.iloc[0]:+.4f} at 5% -> {gaps.iloc[-1]:+.4f} at 100%; "
              f"crossover at fraction {crossover}")

    (paths.RESULTS / "learning_curves.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
