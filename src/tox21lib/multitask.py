"""Multi-task interference on the shared encoder: diagnosis and four repairs.

Configuration C of the Phase 3 ablation (GIN encoder shared between toxicity and
affinity) scores 0.6820 macro ROC-AUC on toxicity against 0.7689 for the
single-task control A -- a 0.0869 penalty, far outside the 0.0285 seed spread.
Configuration D (ChemBERTa, same sharing) loses nothing (0.7619 vs 0.7603).

Reading :func:`tox21lib.train.train_model` explains the asymmetry, and the
explanation is not gradient conflict:

* The loop in ``train_model`` iterates the two task loaders **one after the
  other**, calling ``opt.zero_grad(); loss.backward(); opt.step()`` inside each.
  The two task gradients therefore never coexist -- they are applied in separate
  optimiser steps. The docstring in that function claimed the losses were summed;
  they are not, and the docstring has been corrected rather than the behaviour,
  because configuration C's published numbers were produced by this code path.
* Affinity is iterated last, so the shared encoder ends every epoch having been
  updated only by affinity gradients, and toxicity is then validated on it.
* The two loaders are also very unequal: KIBA contributes roughly 18x as many
  batches per epoch as Tox21, so the affinity task takes ~18x more optimiser
  steps on the shared weights.

That predicts exactly the observed asymmetry: a from-scratch GIN is plastic and
forgets, while ChemBERTa is pretrained and runs at ``encoder_lr=2e-5``, 50x lower
than the heads, so it barely moves. It also predicts that ``loss_weight=0.1``
should recover most of the loss, which it does (0.7453, penalty 0.0236).

So the arms below separate "stale encoder" from "conflicting gradients" before
applying any of the fixes the usual multi-task literature recommends:

``sequential_affinity_last``
    The published configuration C path, reimplemented here so every arm shares
    one evaluation harness. Reference point.
``sequential_toxicity_last``
    Identical, with the task order swapped. Isolates *update order* alone: if
    staleness is the mechanism, toxicity should recover here with no other
    change, and affinity should degrade instead.
``sequential_balanced``
    Affinity truncated to the same number of batches per epoch as toxicity.
    Isolates the ~18x *step imbalance* alone.
``joint``
    Both task gradients summed into one optimiser step -- what the original
    docstring described. This is the prerequisite for any gradient-surgery
    method, since PCGrad needs the two gradients to exist simultaneously.
``pcgrad``
    Joint stepping plus the gradient surgery of Yu et al. (2020): when the two
    task gradients on the shared encoder point in conflicting directions
    (negative cosine), each is projected onto the normal plane of the other.
``soft``
    No shared weights at all: one encoder per task, coupled by an L2 penalty
    between corresponding parameters. Sharing becomes a tunable pull rather than
    a hard constraint.
``adapter``
    One shared encoder with a small per-task bottleneck adapter before each head,
    so each task owns a private transform of the shared representation.

Every arm records the cosine between the two task gradients on the shared
encoder, which is the quantity the "gradient conflict" explanation is about.

Reference
---------
T. Yu et al. Gradient Surgery for Multi-Task Learning. NeurIPS, 2020.
"""

from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from . import models, train

#: Arms, in the order the report presents them.
MODES = (
    "sequential_affinity_last",
    "sequential_toxicity_last",
    "sequential_balanced",
    "joint",
    "pcgrad",
    "soft",
    "adapter",
)

#: Arms that need both task gradients in the same optimiser step.
_JOINT_MODES = frozenset({"joint", "pcgrad", "soft", "adapter"})


@dataclass
class ShareConfig:
    """Training knobs for :func:`train_shared`.

    Mirrors :class:`tox21lib.train.TrainConfig` so the two harnesses stay
    comparable; ``mode`` and ``soft_lambda`` are the only additions.
    """

    mode: str = "joint"
    epochs: int = 60
    patience: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    encoder_lr: float | None = None
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    amp: bool = True
    loss_weight: float = 1.0
    soft_lambda: float = 0.01
    adapter_bottleneck: int = 64
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of {MODES}")


class TaskAdapter(nn.Module):
    """Residual bottleneck adapter: ``x + up(gelu(down(x)))``.

    Initialised so the block starts as the identity (``up`` is zeroed), which
    keeps the ``adapter`` arm's epoch 0 identical to plain sharing and makes any
    later difference attributable to the adapters rather than to a different
    initialisation.
    """

    def __init__(self, dim: int, bottleneck: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.up(self.act(self.down(x))))


class AdapterMultiTaskModel(nn.Module):
    """Shared encoder, one private adapter per task, then the usual heads."""

    def __init__(self, encoder: nn.Module, *, n_endpoints: int = 12,
                 protein_dim: int, dropout: float = 0.1,
                 bottleneck: int = 64) -> None:
        super().__init__()
        self.encoder = encoder
        dim = encoder.out_dim
        self.tox_adapter = TaskAdapter(dim, bottleneck, dropout)
        self.affinity_adapter = TaskAdapter(dim, bottleneck, dropout)
        self.tox_head = models.ToxicityHead(dim, n_endpoints, dropout=dropout)
        self.affinity_head = models.AffinityHead(dim, protein_dim, dropout=dropout)

    def embed(self, batch) -> torch.Tensor:
        return self.encoder(batch)

    def predict_toxicity(self, batch) -> torch.Tensor:
        return self.tox_head(self.tox_adapter(self.embed(batch)))

    def predict_affinity(self, batch, protein_emb: torch.Tensor) -> torch.Tensor:
        return self.affinity_head(self.affinity_adapter(self.embed(batch)), protein_emb)

    def shared_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())


class SoftSharedModel(nn.Module):
    """Two encoders of identical architecture, coupled by an L2 penalty.

    ``coupling_penalty()`` returns ``sum ||theta_tox - theta_affinity||^2`` over
    corresponding parameters. At ``soft_lambda=0`` this is two independent
    single-task models; as the penalty grows it approaches hard sharing, so the
    arm spans the whole range between configurations A/B and C.
    """

    def __init__(self, encoder_factory, *, n_endpoints: int = 12,
                 protein_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.tox_encoder = encoder_factory()
        self.affinity_encoder = encoder_factory()
        # Start both encoders from the same draw: otherwise the penalty spends
        # the early epochs cancelling an initialisation difference that has
        # nothing to do with the tasks.
        self.affinity_encoder.load_state_dict(self.tox_encoder.state_dict())
        dim = self.tox_encoder.out_dim
        self.tox_head = models.ToxicityHead(dim, n_endpoints, dropout=dropout)
        self.affinity_head = models.AffinityHead(dim, protein_dim, dropout=dropout)

    # The evaluation helpers in tox21lib.train call .predict_* only, so this
    # class does not need a single `encoder` attribute.
    def predict_toxicity(self, batch) -> torch.Tensor:
        return self.tox_head(self.tox_encoder(batch))

    def predict_affinity(self, batch, protein_emb: torch.Tensor) -> torch.Tensor:
        return self.affinity_head(self.affinity_encoder(batch), protein_emb)

    def coupling_penalty(self) -> torch.Tensor:
        total = None
        for a, b in zip(self.tox_encoder.parameters(),
                        self.affinity_encoder.parameters()):
            term = (a - b).pow(2).sum()
            total = term if total is None else total + term
        if total is None:
            raise RuntimeError("encoders expose no parameters to couple")
        return total

    def shared_parameters(self) -> list[nn.Parameter]:
        """The toxicity encoder stands in as the 'shared' block for diagnostics."""
        return list(self.tox_encoder.parameters())


def build_model(mode: str, encoder_factory, *, protein_dim: int,
                n_endpoints: int = 12, dropout: float = 0.1,
                bottleneck: int = 64) -> nn.Module:
    """The model an arm needs; all arms other than soft/adapter share one class."""
    if mode == "soft":
        return SoftSharedModel(encoder_factory, n_endpoints=n_endpoints,
                               protein_dim=protein_dim, dropout=dropout)
    if mode == "adapter":
        return AdapterMultiTaskModel(encoder_factory(), n_endpoints=n_endpoints,
                                     protein_dim=protein_dim, dropout=dropout,
                                     bottleneck=bottleneck)
    model = models.MultiTaskModel(encoder_factory(), n_endpoints=n_endpoints,
                                  protein_dim=protein_dim, dropout=dropout)
    return model


def _shared_parameters(model: nn.Module) -> list[nn.Parameter]:
    if hasattr(model, "shared_parameters"):
        return model.shared_parameters()
    return list(model.encoder.parameters())


def _flat(grads: list[torch.Tensor | None]) -> torch.Tensor | None:
    present = [g.reshape(-1) for g in grads if g is not None]
    if not present:
        return None
    return torch.cat(present)


def pcgrad_project(g_a: list[torch.Tensor | None],
                   g_b: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
    """Gradient surgery for two tasks (Yu et al., 2020).

    Each task gradient is projected onto the normal plane of the other whenever
    their inner product is negative, then the two are summed. With a negative
    cosine this removes the component of each gradient that actively undoes the
    other task; with a non-negative cosine it is exactly the plain sum, so the
    arm differs from ``joint`` only on steps where the tasks really do conflict.
    """
    flat_a, flat_b = _flat(g_a), _flat(g_b)
    if flat_a is None or flat_b is None:
        return [a if a is not None else b for a, b in zip(g_a, g_b)]

    dot = torch.dot(flat_a, flat_b)
    if dot >= 0:
        return [_add(a, b) for a, b in zip(g_a, g_b)]

    # Project a off b and b off a. Both use the *original* gradients, matching
    # the reference implementation: projecting sequentially would make the
    # result depend on task order.
    sq_a = flat_a.dot(flat_a)
    sq_b = flat_b.dot(flat_b)
    coef_a = dot / sq_b if sq_b > 0 else torch.zeros_like(dot)
    coef_b = dot / sq_a if sq_a > 0 else torch.zeros_like(dot)
    out = []
    for a, b in zip(g_a, g_b):
        if a is None and b is None:
            out.append(None)
            continue
        pa = a - coef_a * b if (a is not None and b is not None) else a
        pb = b - coef_b * a if (a is not None and b is not None) else b
        out.append(_add(pa, pb))
    return out


def _add(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _cosine(g_a: list[torch.Tensor | None],
            g_b: list[torch.Tensor | None]) -> float | None:
    flat_a, flat_b = _flat(g_a), _flat(g_b)
    if flat_a is None or flat_b is None:
        return None
    na, nb = flat_a.norm(), flat_b.norm()
    if na == 0 or nb == 0:
        return None
    return float(torch.dot(flat_a, flat_b) / (na * nb))


def _tox_loss(model, batch, pos_weight) -> torch.Tensor:
    logits = model.predict_toxicity(batch)
    n_end = logits.shape[1]
    return models.masked_bce(
        logits,
        batch.y_tox.view(-1, n_end).to(logits.dtype),
        batch.mask_tox.view(-1, n_end).to(logits.dtype),
        pos_weight=pos_weight,
    )


def _affinity_loss(model, batch, protein_emb) -> torch.Tensor:
    pred = model.predict_affinity(batch, protein_emb[batch.protein_row])
    return torch.nn.functional.mse_loss(pred, batch.y_affinity.to(pred.dtype))


def train_shared(
    model: nn.Module,
    *,
    tox_loaders,
    affinity_loaders,
    protein_emb: torch.Tensor,
    cfg: ShareConfig,
    device: str,
    pos_weight: torch.Tensor | None = None,
    verbose: bool = False,
) -> dict:
    """Train one arm; returns history, best epoch, and gradient diagnostics.

    The validation objective is ``macro ROC-AUC + Pearson r``, unweighted, which
    is the objective :func:`tox21lib.train.train_model` already uses -- keeping it
    identical is what makes these arms comparable with configurations A through F.
    """
    model = model.to(device)
    shared = _shared_parameters(model)
    shared_ids = {id(p) for p in shared}
    other = [p for p in model.parameters() if id(p) not in shared_ids]
    groups = [{"params": other, "lr": cfg.lr}]
    if shared:
        groups.append({"params": shared,
                       "lr": cfg.encoder_lr if cfg.encoder_lr else cfg.lr})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    amp_on = cfg.amp and device == "cuda"
    amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tox_train, tox_valid = tox_loaders
    aff_train, aff_valid = affinity_loaders
    n_tox_batches, n_aff_batches = len(tox_train), len(aff_train)

    best = {"objective": -float("inf"), "epoch": -1, "state": None}
    history: list[dict] = []
    cosines: list[float] = []
    conflict_steps = 0
    total_joint_steps = 0
    stale = 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        sums = {"tox": 0.0, "affinity": 0.0}
        counts = {"tox": 0, "affinity": 0}
        epoch_cos: list[float] = []

        if cfg.mode in _JOINT_MODES:
            # One step consumes one batch from each task. Affinity is the longer
            # loader, so toxicity cycles; iterating the longer one keeps the
            # affinity task's step count equal to the sequential arms, which is
            # what makes the affinity column comparable across the table.
            tox_stream = itertools.cycle(tox_train)
            for aff_batch in aff_train:
                aff_batch = aff_batch.to(device)
                tox_batch = next(tox_stream).to(device)

                # Per-task backward passes. Summing the two gradients afterwards
                # is algebraically identical to backward on the summed loss, so
                # the `joint` arm is unaffected by doing it this way -- and it
                # makes the task-gradient cosine available at no extra cost.
                per_task: list[list[torch.Tensor | None]] = []
                losses = {}
                for task in ("tox", "affinity"):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device, dtype=amp_dtype,
                                        enabled=amp_on):
                        if task == "tox":
                            loss = _tox_loss(model, tox_batch, pos_weight)
                        else:
                            loss = _affinity_loss(model, aff_batch, protein_emb)
                            loss = loss * cfg.loss_weight
                    loss.backward()
                    per_task.append([
                        p.grad.detach().clone() if p.grad is not None else None
                        for p in shared
                    ])
                    losses[task] = float(loss.detach())
                    # Heads are task-exclusive, so their gradients are already
                    # final; stash them before the next zero_grad wipes them.
                    if task == "tox":
                        head_grads = {id(p): (p.grad.detach().clone()
                                              if p.grad is not None else None)
                                      for p in other}
                    else:
                        for p in other:
                            if p.grad is not None:
                                prev = head_grads.get(id(p))
                                head_grads[id(p)] = (p.grad.detach().clone()
                                                     if prev is None
                                                     else prev + p.grad.detach())

                cos = _cosine(per_task[0], per_task[1])
                if cos is not None:
                    epoch_cos.append(cos)
                    total_joint_steps += 1
                    if cos < 0:
                        conflict_steps += 1

                if cfg.mode == "pcgrad":
                    merged = pcgrad_project(per_task[0], per_task[1])
                else:
                    merged = [_add(a, b) for a, b in zip(per_task[0], per_task[1])]

                opt.zero_grad(set_to_none=True)
                for p, g in zip(shared, merged):
                    p.grad = g
                for p in other:
                    p.grad = head_grads.get(id(p))

                if cfg.mode == "soft":
                    # The coupling term touches both encoders only, so it is
                    # added after the task gradients are in place.
                    penalty = cfg.soft_lambda * model.coupling_penalty()
                    penalty.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                sums["tox"] += losses["tox"]
                sums["affinity"] += losses["affinity"]
                counts["tox"] += 1
                counts["affinity"] += 1
        else:
            order = (["tox", "affinity"] if cfg.mode != "sequential_toxicity_last"
                     else ["affinity", "tox"])
            limit = (n_tox_batches if cfg.mode == "sequential_balanced" else None)
            for task in order:
                loader = tox_train if task == "tox" else aff_train
                stream = loader if (task == "tox" or limit is None) \
                    else itertools.islice(loader, limit)
                for batch in stream:
                    batch = batch.to(device)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device, dtype=amp_dtype,
                                        enabled=amp_on):
                        if task == "tox":
                            loss = _tox_loss(model, batch, pos_weight)
                        else:
                            loss = _affinity_loss(model, batch, protein_emb)
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
        if epoch_cos:
            record["grad_cosine_mean"] = float(np.mean(epoch_cos))
            record["grad_conflict_rate"] = float(np.mean([c < 0 for c in epoch_cos]))
            cosines.extend(epoch_cos)

        val_tox = train.evaluate_toxicity(model, tox_valid, device)
        val_aff = train.evaluate_affinity(model, aff_valid, protein_emb, device)
        record["valid_macro_roc_auc"] = val_tox["macro_roc_auc"]
        record["valid_pearson"] = val_aff["pearson_r"]
        objective = val_tox["macro_roc_auc"] + val_aff["pearson_r"]
        record["objective"] = objective
        history.append(record)
        if verbose:
            print("      " + "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in record.items()))

        if objective > best["objective"] + 1e-6:
            best = {
                "objective": objective,
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
            }
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    return {
        "history": history,
        "best_epoch": best["epoch"],
        "best_objective": best["objective"],
        "epochs_run": len(history),
        "train_seconds": time.time() - t0,
        "n_tox_batches_per_epoch": n_tox_batches,
        "n_affinity_batches_per_epoch": n_aff_batches,
        "batch_ratio_affinity_over_tox": n_aff_batches / max(n_tox_batches, 1),
        "grad_cosine_mean": float(np.mean(cosines)) if cosines else None,
        "grad_cosine_std": float(np.std(cosines)) if cosines else None,
        "grad_conflict_rate": (conflict_steps / total_joint_steps
                               if total_joint_steps else None),
        "n_gradient_pairs_measured": total_joint_steps,
    }
