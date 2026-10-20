"""Encoders and task heads for the shared-encoder ablation.

Four configurations share this code:
  A  GIN from scratch, toxicity only
  B  GIN from scratch, affinity only
  C  GIN from scratch, both tasks through one shared encoder
  D  pretrained chemical language model (ChemBERTa), fine-tuned, both tasks

The encoder is swappable precisely so A-C and D differ only in the encoder, not
in the heads, the data, the splits or the loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool

from .graphs import ATOM_FEATURE_DIM, BOND_FEATURE_DIM

#: Default pretrained chemical language model for configuration D. Its corpus
#: (PubChem/ZINC) spans industrial chemicals and drug-like molecules, which is
#: the property configuration D is meant to test.
CHEMBERTA_CHECKPOINT = "DeepChem/ChemBERTa-77M-MLM"


class GINEncoder(nn.Module):
    """GIN over molecular graphs with bond features (GINEConv).

    Readout concatenates mean and max pooling; ``out_dim`` is the molecular
    embedding width handed to both heads.
    """

    def __init__(
        self,
        out_dim: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.atom_encoder = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)
        self.bond_encoder = nn.Linear(BOND_FEATURE_DIM, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.project = nn.Linear(2 * hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, data) -> torch.Tensor:
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, data.edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = torch.cat(
            [global_mean_pool(x, data.batch), global_max_pool(x, data.batch)], dim=-1
        )
        return self.project(pooled)


class ChemBertaEncoder(nn.Module):
    """Pretrained chemical language model, fully fine-tuned (configuration D)."""

    def __init__(self, out_dim: int = 256, checkpoint: str = CHEMBERTA_CHECKPOINT,
                 dropout: float = 0.1) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(checkpoint)
        hidden = self.backbone.config.hidden_size
        self.project = nn.Linear(hidden, out_dim)
        self.dropout = dropout
        self.out_dim = out_dim

    def forward(self, batch) -> torch.Tensor:
        out = self.backbone(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
        hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = F.dropout(pooled, p=self.dropout, training=self.training)
        return self.project(pooled)


class ToxicityHead(nn.Module):
    """One logit per endpoint; missing labels are masked in the loss, not here."""

    def __init__(self, in_dim: int, n_endpoints: int = 12, hidden_dim: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_endpoints),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class AffinityHead(nn.Module):
    """Concatenates the molecular embedding with a frozen ESM-2 protein embedding."""

    def __init__(self, mol_dim: int, protein_dim: int, hidden_dim: int = 512,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(mol_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, mol_emb: torch.Tensor, protein_emb: torch.Tensor) -> torch.Tensor:
        p = self.protein_proj(protein_emb)
        return self.net(torch.cat([mol_emb, p], dim=-1)).squeeze(-1)


class MultiTaskModel(nn.Module):
    """One encoder, optionally two heads - the object the ablation varies."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        n_endpoints: int | None = 12,
        protein_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.tox_head = (
            ToxicityHead(encoder.out_dim, n_endpoints, dropout=dropout)
            if n_endpoints
            else None
        )
        self.affinity_head = (
            AffinityHead(encoder.out_dim, protein_dim, dropout=dropout)
            if protein_dim
            else None
        )

    def embed(self, batch) -> torch.Tensor:
        return self.encoder(batch)

    def predict_toxicity(self, batch) -> torch.Tensor:
        if self.tox_head is None:
            raise RuntimeError("this configuration has no toxicity head")
        return self.tox_head(self.embed(batch))

    def predict_affinity(self, batch, protein_emb: torch.Tensor) -> torch.Tensor:
        if self.affinity_head is None:
            raise RuntimeError("this configuration has no affinity head")
        return self.affinity_head(self.embed(batch), protein_emb)


def masked_bce(logits: torch.Tensor, targets: torch.Tensor,
               mask: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCE over observed labels only; Tox21 is ~25% missing per endpoint."""
    if mask.sum() == 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
        pos_weight=pos_weight,
    )
    return (loss * mask).sum() / mask.sum()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
