"""Training / evaluation loops for the shared-encoder ablation.

Everything is seeded explicitly and evaluated on the committed split index files,
so a configuration's numbers depend only on (config, loss weight, split, seed).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .dti import concordance_index
from .models import masked_bce


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch, and ask cuDNN for deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    """Hyperparameters; identical across configurations unless stated otherwise."""

    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    encoder_lr: float | None = None   # configuration D uses a lower encoder LR
    weight_decay: float = 1e-5
    patience: int = 5                 # early stopping on the validation objective
    grad_clip: float = 5.0
    amp: bool = True
    loss_weight: float = 1.0          # affinity weight relative to toxicity
    history: list = field(default_factory=list)


def _amp_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


@torch.no_grad()
def evaluate_toxicity(model, loader, device: str, n_endpoints: int = 12) -> dict:
    """Per-endpoint ROC-AUC / PR-AUC over observed labels in the loader."""
    model.eval()
    logits_all, y_all, mask_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model.predict_toxicity(batch)
        logits_all.append(logits.float().cpu())
        y_all.append(batch.y_tox.view(-1, n_endpoints).float().cpu())
        mask_all.append(batch.mask_tox.view(-1, n_endpoints).float().cpu())
    logits = torch.cat(logits_all).numpy()
    y = torch.cat(y_all).numpy()
    mask = torch.cat(mask_all).numpy().astype(bool)

    per_endpoint = []
    for j in range(n_endpoints):
        sel = mask[:, j]
        yj, pj = y[sel, j], logits[sel, j]
        if sel.sum() == 0 or len(np.unique(yj)) < 2:
            per_endpoint.append({"endpoint_index": j, "n": int(sel.sum()),
                                 "n_pos": int(yj.sum()), "roc_auc": float("nan"),
                                 "pr_auc": float("nan")})
            continue
        per_endpoint.append(
            {
                "endpoint_index": j,
                "n": int(sel.sum()),
                "n_pos": int(yj.sum()),
                "roc_auc": float(roc_auc_score(yj, pj)),
                "pr_auc": float(average_precision_score(yj, pj)),
            }
        )
    roc = [e["roc_auc"] for e in per_endpoint if not np.isnan(e["roc_auc"])]
    pr = [e["pr_auc"] for e in per_endpoint if not np.isnan(e["pr_auc"])]
    return {
        "per_endpoint": per_endpoint,
        "macro_roc_auc": float(np.mean(roc)) if roc else float("nan"),
        "macro_pr_auc": float(np.mean(pr)) if pr else float("nan"),
    }


@torch.no_grad()
def evaluate_affinity(model, loader, protein_emb: torch.Tensor, device: str) -> dict:
    """RMSE, Pearson, Spearman and concordance index on the loader's pairs."""
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        p_emb = protein_emb[batch.protein_row]
        pred = model.predict_affinity(batch, p_emb)
        preds.append(pred.float().cpu())
        targets.append(batch.y_affinity.float().cpu())
    pred = torch.cat(preds).numpy()
    y = torch.cat(targets).numpy()
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "pearson_r": float(pearsonr(y, pred)[0]),
        "spearman_rho": float(spearmanr(y, pred)[0]),
        "ci": float(concordance_index(y, pred)),
    }


def train_model(
    model,
    *,
    tox_loaders: tuple | None,
    affinity_loaders: tuple | None,
    protein_emb: torch.Tensor | None,
    cfg: TrainConfig,
    device: str,
    pos_weight: torch.Tensor | None = None,
    verbose: bool = True,
) -> dict:
    """Train one configuration; returns the training history and best epoch.

    ``tox_loaders`` / ``affinity_loaders`` are ``(train, valid)`` pairs; passing
    only one of them gives a single-task configuration (A or B). With both, each
    epoch alternates: it consumes the toxicity loader and the affinity loader,
    summing ``loss_tox + loss_weight * loss_affinity`` over the shared encoder.
    """
    model = model.to(device)
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    groups = [{"params": head_params, "lr": cfg.lr}]
    if encoder_params:
        groups.append({"params": encoder_params,
                       "lr": cfg.encoder_lr if cfg.encoder_lr else cfg.lr})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    best = {"objective": -float("inf"), "epoch": -1, "state": None}
    history = []
    stale = 0

    for epoch in range(cfg.epochs):
        model.train()
        sums = {"tox": 0.0, "affinity": 0.0}
        counts = {"tox": 0, "affinity": 0}

        iterators = []
        if tox_loaders is not None:
            iterators.append(("tox", tox_loaders[0]))
        if affinity_loaders is not None:
            iterators.append(("affinity", affinity_loaders[0]))

        for task, loader in iterators:
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=_amp_dtype(device),
                                    enabled=cfg.amp and device == "cuda"):
                    if task == "tox":
                        logits = model.predict_toxicity(batch)
                        n_end = logits.shape[1]
                        loss = masked_bce(
                            logits,
                            batch.y_tox.view(-1, n_end).to(logits.dtype),
                            batch.mask_tox.view(-1, n_end).to(logits.dtype),
                            pos_weight=pos_weight,
                        )
                    else:
                        pred = model.predict_affinity(batch, protein_emb[batch.protein_row])
                        loss = torch.nn.functional.mse_loss(
                            pred, batch.y_affinity.to(pred.dtype)
                        )
                        loss = loss * cfg.loss_weight
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                sums[task] += float(loss.detach())
                counts[task] += 1

        record = {"epoch": epoch}
        for task in ("tox", "affinity"):
            if counts[task]:
                record[f"train_loss_{task}"] = sums[task] / counts[task]

        # Validation objective, "higher is better" in every configuration. For a
        # multi-task model it MUST include both tasks: selecting on toxicity alone
        # stops training while the affinity head is still far from converged, which
        # would make the ablation measure early stopping rather than sharing. Both
        # terms are on a 0-1 scale (macro ROC-AUC, Pearson r), so they are summed
        # unweighted; the loss weight stays the thing being swept.
        objective = 0.0
        if tox_loaders is not None:
            val_tox = evaluate_toxicity(model, tox_loaders[1], device)
            record["valid_macro_roc_auc"] = val_tox["macro_roc_auc"]
            objective += val_tox["macro_roc_auc"]
        if affinity_loaders is not None:
            val_aff = evaluate_affinity(model, affinity_loaders[1], protein_emb, device)
            record["valid_rmse"] = val_aff["rmse"]
            record["valid_pearson"] = val_aff["pearson_r"]
            objective += val_aff["pearson_r"]
        record["objective"] = objective
        history.append(record)
        if verbose:
            print("      " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in record.items()))

        if objective > best["objective"] + 1e-6:
            best = {
                "objective": objective,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                if verbose:
                    print(f"      early stop at epoch {epoch} "
                          f"(best epoch {best['epoch']})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"history": history, "best_epoch": best["epoch"],
            "best_objective": best["objective"], "epochs_run": len(history)}
