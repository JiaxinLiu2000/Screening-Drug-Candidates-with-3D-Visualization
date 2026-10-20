"""Drug-target affinity (Davis / KIBA) baseline: pair tables, splits, features, metrics.

Units
-----
TDC's ``DTI(name='DAVIS')`` returns Kd in **nM** (raw). We convert to pKd with
exactly TDC's ``convert_to_log`` formula, ``pKd = -log10(Kd * 1e-9 + 1e-10)``,
so the censored ceiling Kd = 10000 nM maps to pKd = 4.99999566 (~5).
KIBA is returned as a KIBA score and is used untransformed.

Protein features
----------------
420 dims = amino-acid composition (20, alphabet ``ACDEFGHIKLMNPQRSTVWY``) followed
by dipeptide composition (400, row-major over the same alphabet: AA, AC, ...).
Rule for non-standard residues (X, B, Z, U, O, lowercase is upper-cased first):
they are excluded from the numerators *and* denominators -- AAC fractions are
over standard residues only, DPC fractions over dipeptides whose both residues
are standard. Each block therefore sums to 1 (unless the sequence has no
standard residue/dipeptide, which raises).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import featurize
from .splits import random_split

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX = {a: i for i, a in enumerate(AMINO_ACIDS)}
N_PROTEIN_FEATURES = 20 + 400
DAVIS_CEILING_NM = 10000.0
SPLIT_KINDS = ("random", "cold_drug", "cold_target")


def kd_nm_to_pkd(kd_nm: np.ndarray) -> np.ndarray:
    """TDC ``convert_to_log`` (binding form): -log10(Kd[nM]*1e-9 + 1e-10)."""
    return -np.log10(np.asarray(kd_nm, dtype=float) * 1e-9 + 1e-10)


def protein_features(sequence: str) -> np.ndarray:
    """AAC(20) + DPC(400) fractions as float32; see module docstring for the rule."""
    idx = [_AA_INDEX.get(c, -1) for c in sequence.upper()]
    aac = np.zeros(20, dtype=np.float64)
    dpc = np.zeros(400, dtype=np.float64)
    for i in idx:
        if i >= 0:
            aac[i] += 1
    for a, b in zip(idx[:-1], idx[1:]):
        if a >= 0 and b >= 0:
            dpc[a * 20 + b] += 1
    if aac.sum() == 0 or dpc.sum() == 0:
        raise ValueError("sequence has no standard residues/dipeptides")
    return np.concatenate([aac / aac.sum(), dpc / dpc.sum()]).astype(np.float32)


def build_pairs(raw: pd.DataFrame, clean: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep pairs whose drug survived standardization; map drugs to InChIKey.

    Pairs that collapse onto the same (inchikey, target_id) are averaged.
    Returns pairs sorted by (inchikey, target_id) with a 0-based ``pair_index``.
    """
    id_to_key = {}
    for key, ids in zip(clean["inchikey"], clean["drug_ids"]):
        for d in str(ids).split("|"):
            id_to_key[d] = key
    raw = raw.copy()
    raw["inchikey"] = raw["drug_id"].astype(str).map(id_to_key)
    kept = raw.dropna(subset=["inchikey"])
    grouped = (
        kept.groupby(["inchikey", "target_id"], sort=True)
        .agg(
            drug_id=("drug_id", lambda s: "|".join(sorted(set(s.astype(str))))),
            target_sequence=("target_sequence", "first"),
            affinity=("affinity", "mean"),
            n_raw=("affinity", "size"),
        )
        .reset_index()
    )
    grouped.insert(0, "pair_index", np.arange(len(grouped)))
    stats = {
        "raw_pairs": int(len(raw)),
        "pairs_dropped_by_standardization": int(len(raw) - len(kept)),
        "pairs_after_standardization": int(len(kept)),
        "pairs_merged_by_inchikey": int(len(kept) - len(grouped)),
        "final_pairs": int(len(grouped)),
    }
    return grouped, stats


def _entity_split(entities: np.ndarray, seed: int) -> dict:
    labels = random_split(len(entities), seed)
    return dict(zip(entities, labels))


def make_split(pairs: pd.DataFrame, kind: str, seed: int) -> np.ndarray:
    """Subset label per pair. Cold splits partition the sorted entity list 80/10/10."""
    if kind == "random":
        return random_split(len(pairs), seed)
    col = {"cold_drug": "inchikey", "cold_target": "target_id"}[kind]
    mapping = _entity_split(np.array(sorted(pairs[col].unique())), seed)
    subset = pairs[col].map(mapping).to_numpy()
    tr = set(pairs.loc[subset == "train", col])
    for other in ("valid", "test"):
        leak = tr & set(pairs.loc[subset == other, col])
        if leak:
            raise AssertionError(f"{kind} seed {seed}: {len(leak)} {col} leak train->{other}")
    return subset


def feature_matrix(pairs: pd.DataFrame, smiles_by_key: dict) -> np.ndarray:
    """Pair matrix [Morgan 2048 | protein 420] float32, assembled from entity caches."""
    drugs = sorted(pairs["inchikey"].unique())
    d_idx = {k: i for i, k in enumerate(drugs)}
    d_mat = featurize.featurize_smiles([smiles_by_key[k] for k in drugs]).astype(np.float32)
    seqs = pairs.drop_duplicates("target_id").set_index("target_id")["target_sequence"]
    targets = sorted(seqs.index)
    t_idx = {t: i for i, t in enumerate(targets)}
    t_mat = np.stack([protein_features(seqs[t]) for t in targets])
    X = np.empty((len(pairs), d_mat.shape[1] + N_PROTEIN_FEATURES), dtype=np.float32)
    X[:, : d_mat.shape[1]] = d_mat[pairs["inchikey"].map(d_idx).to_numpy()]
    X[:, d_mat.shape[1]:] = t_mat[pairs["target_id"].map(t_idx).to_numpy()]
    return X


def concordance_index(y: np.ndarray, pred: np.ndarray, block: int = 2048) -> float:
    """Exact CI over all comparable pairs (y_i != y_j); prediction ties count 0.5.

    Blockwise O(n^2) in numpy -- exact, no sampling.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    num = 0.0
    den = 0
    for s in range(0, len(y), block):
        dy = y[s:s + block, None] - y[None, :]
        dp = p[s:s + block, None] - p[None, :]
        comp = dy > 0
        den += int(comp.sum())
        num += float((comp & (dp > 0)).sum()) + 0.5 * float((comp & (dp == 0)).sum())
    return num / den
