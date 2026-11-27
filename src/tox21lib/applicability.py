"""Applicability domain: how far a query molecule is from the training set.

The abstention experiment in `src/33_abstention.py` found that the model's own
confidence is worthless as a guide to when it is wrong - abstaining on the
molecules whose predicted probability sits closest to 0.5 scored PR-AUC 0.2079,
*worse* than abstaining at random (0.3983) - while abstaining on the molecules
furthest from the training set by Tanimoto similarity scored 0.4904, the best of
the four signals tested.

So distance-to-training-set is the signal the application must surface, and it
has to be computable without the raw datasets, which are not committed. This
module builds a compact reference of training-set fingerprints that ships inside
``models/`` and answers nearest-neighbour queries against it.

The packed form is ``numpy.packbits`` over binary Morgan bits: 2048 bits become
256 bytes per molecule, so the whole Tox21 training fold is about 1.2 MB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import featurize

#: Bumped whenever the stored layout changes; `load_reference` refuses older files.
REFERENCE_FORMAT_VERSION = 1

#: Below this nearest-neighbour Tanimoto a prediction is flagged out-of-domain.
#: 0.4 is the operating point the abstention study used to define its best half
#: (`results/abstention.json`, coverage 0.5 on `tanimoto_to_train`); it is a
#: reporting threshold, not a tuned hyperparameter, and callers may override it.
DEFAULT_DOMAIN_THRESHOLD = 0.4


class ApplicabilityError(RuntimeError):
    """Raised when the reference is missing, stale, or shaped unexpectedly."""


@dataclass(frozen=True)
class Reference:
    """Packed training fingerprints for one task."""

    task: str
    bits: np.ndarray            # (n_molecules, n_bits // 8) uint8, packbits order
    popcounts: np.ndarray       # (n_molecules,) int32, bits set per molecule
    n_bits: int
    radius: int
    split: str
    seed: int
    n_molecules: int

    def __len__(self) -> int:
        return self.n_molecules


def _popcount(packed: np.ndarray) -> np.ndarray:
    """Bits set per row of a packbits array."""
    # np.bitwise_count lands on uint8 directly and avoids a 256-entry lookup.
    return np.bitwise_count(packed).sum(axis=1).astype(np.int32)


def build_reference(smiles: list[str], *, task: str, split: str, seed: int,
                    radius: int = featurize.MORGAN_RADIUS,
                    n_bits: int = featurize.MORGAN_N_BITS) -> Reference:
    """Fingerprint a training fold and pack it.

    Unparseable SMILES raise rather than being skipped: a reference silently
    missing molecules would make every query look slightly more in-domain than
    it is.
    """
    matrix = featurize.featurize_smiles(smiles, radius=radius, n_bits=n_bits)
    if matrix.shape[0] != len(smiles):
        raise ApplicabilityError(
            f"featurized {matrix.shape[0]} of {len(smiles)} molecules")
    packed = np.packbits(matrix.astype(bool), axis=1)
    return Reference(
        task=task, bits=packed, popcounts=_popcount(packed), n_bits=n_bits,
        radius=radius, split=split, seed=seed, n_molecules=matrix.shape[0],
    )


def save_reference(path: Path, references: dict[str, Reference]) -> Path:
    """Write one npz holding every task's reference plus a sidecar manifest."""
    payload: dict[str, np.ndarray] = {}
    manifest: dict[str, dict] = {}
    for name, ref in references.items():
        payload[f"{name}__bits"] = ref.bits
        payload[f"{name}__popcounts"] = ref.popcounts
        manifest[name] = {
            "task": ref.task, "n_bits": ref.n_bits, "radius": ref.radius,
            "split": ref.split, "seed": ref.seed, "n_molecules": ref.n_molecules,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, format_version=REFERENCE_FORMAT_VERSION,
                        manifest=json.dumps(manifest), **payload)
    path.with_suffix(".json").write_text(
        json.dumps({"format_version": REFERENCE_FORMAT_VERSION,
                    "references": manifest}, indent=2) + "\n",
        encoding="utf-8")
    return path


def load_reference(path: Path) -> dict[str, Reference]:
    """Read the shipped reference; raises if absent or from a newer layout."""
    if not Path(path).exists():
        raise ApplicabilityError(
            f"{path} missing; run src/38_export_models.py to build it")
    with np.load(path, allow_pickle=False) as data:
        version = int(data["format_version"])
        if version != REFERENCE_FORMAT_VERSION:
            raise ApplicabilityError(
                f"{path} is format version {version}, this build expects "
                f"{REFERENCE_FORMAT_VERSION}")
        manifest = json.loads(str(data["manifest"]))
        out: dict[str, Reference] = {}
        for name, meta in manifest.items():
            bits = data[f"{name}__bits"]
            popcounts = data[f"{name}__popcounts"]
            if bits.shape[0] != meta["n_molecules"]:
                raise ApplicabilityError(
                    f"{name}: manifest claims {meta['n_molecules']} molecules, "
                    f"array holds {bits.shape[0]}")
            out[name] = Reference(
                task=meta["task"], bits=bits, popcounts=popcounts,
                n_bits=meta["n_bits"], radius=meta["radius"],
                split=meta["split"], seed=meta["seed"],
                n_molecules=meta["n_molecules"],
            )
    return out


def nearest_tanimoto(smiles: list[str], reference: Reference) -> dict:
    """Nearest-neighbour Tanimoto from each query to the reference set.

    Returns parallel arrays in input order. A query that cannot be parsed gets
    ``nan`` similarity and ``-1`` neighbour index, so callers can keep the row
    and mark it rather than dropping it and silently shifting the output.
    """
    sims = np.full(len(smiles), np.nan, dtype=np.float64)
    neighbours = np.full(len(smiles), -1, dtype=np.int64)

    parsed: list[tuple[int, np.ndarray]] = []
    for i, smi in enumerate(smiles):
        try:
            bits = featurize.morgan_bits(
                smi, radius=reference.radius, n_bits=reference.n_bits)
        except Exception:
            continue
        parsed.append((i, np.packbits(bits.astype(bool))))

    if not parsed:
        return {"nn_tanimoto": sims, "nn_index": neighbours}

    query = np.stack([p for _, p in parsed])
    query_pop = _popcount(query)
    ref_bits, ref_pop = reference.bits, reference.popcounts

    # Chunked so a large query against a large reference does not allocate an
    # (n_query, n_reference, n_bytes) intermediate.
    chunk = max(1, 2_000_000 // max(ref_bits.shape[0], 1))
    for start in range(0, query.shape[0], chunk):
        block = query[start:start + chunk]
        # (block, ref, bytes) -> popcount of the AND is the intersection
        inter = np.bitwise_count(
            block[:, None, :] & ref_bits[None, :, :]).sum(axis=2)
        union = query_pop[start:start + chunk, None] + ref_pop[None, :] - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            tan = np.where(union > 0, inter / union, 0.0)
        best = tan.argmax(axis=1)
        for row, (orig_index, _) in enumerate(parsed[start:start + chunk]):
            sims[orig_index] = float(tan[row, best[row]])
            neighbours[orig_index] = int(best[row])

    return {"nn_tanimoto": sims, "nn_index": neighbours}


def assess(smiles: list[str], reference: Reference, *,
           threshold: float = DEFAULT_DOMAIN_THRESHOLD) -> list[dict]:
    """Per-molecule applicability verdict, in input order.

    ``advice`` is deliberately blunt. The abstention study showed the model is
    not merely less accurate out of domain, it is confidently wrong there, so
    the application states "do not trust" rather than "lower confidence".
    """
    result = nearest_tanimoto(smiles, reference)
    rows = []
    for sim, idx in zip(result["nn_tanimoto"], result["nn_index"]):
        if np.isnan(sim):
            rows.append({"nn_tanimoto": None, "nn_index": None,
                         "in_domain": False,
                         "advice": "unparseable molecule; no prediction"})
            continue
        in_domain = bool(sim >= threshold)
        rows.append({
            "nn_tanimoto": round(float(sim), 4),
            "nn_index": int(idx),
            "in_domain": in_domain,
            "advice": ("within the training distribution" if in_domain else
                       f"nearest training molecule is only {sim:.2f} similar; "
                       "the model's own confidence does not detect this, so the "
                       "prediction should not be trusted"),
        })
    return rows
