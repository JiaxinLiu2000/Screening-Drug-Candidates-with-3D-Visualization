"""Scoring layer behind the API: load once, predict many.

Everything here is a thin wrapper over `tox21lib`; no model logic lives in the
web layer. Three rules shape the design:

1. **Row order is preserved.** A molecule that will not parse keeps its row with
   ``error`` set, so the caller can line results up with its input. Dropping
   failures and returning a shorter list is how a user silently reads results
   against the wrong molecules.
2. **Applicability is not optional.** `results/abstention.json` showed the
   models are confidently wrong outside their training distribution and that
   their own confidence does not detect it, so every row carries a
   nearest-neighbour distance and a verdict.
3. **Calibrated probabilities where they exist.** The toxicity head emits
   logits; `results/tox_calibration.csv` shows isotonic regression cuts expected
   calibration error from 0.0653 to 0.0252, so the API reports the calibrated
   value as the primary number and keeps the raw one for comparison.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tox21lib import applicability, model_io, paths, properties  # noqa: E402

MODELS_DIR = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"
REFERENCE_PATH = MODELS_DIR / "applicability_reference.npz"


class ServiceError(RuntimeError):
    """A request the service refuses, with a reason fit to show a user."""


@dataclass
class Bundle:
    """Everything loaded once at startup."""

    registry: dict
    references: dict[str, applicability.Reference]
    toxicity: Any = None
    affinity: Any = None
    multitask: Any = None
    calibrators: dict[str, Any] = field(default_factory=dict)
    endpoints: list[str] = field(default_factory=list)
    targets: list[dict] = field(default_factory=list)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


_BUNDLE: Bundle | None = None


def _fit_calibrators(endpoints: list[str]) -> dict[str, Any]:
    """Isotonic calibrators per endpoint, fitted on the committed validation fold.

    Returns an empty mapping when the raw Tox21 table is absent, which is the
    normal state of a fresh clone: the API then reports uncalibrated scores and
    says so, rather than inventing a calibration.
    """
    if not paths.CLEAN_PARQUET.exists():
        return {}
    try:
        import pandas as pd
        from sklearn.isotonic import IsotonicRegression

        from tox21lib import ablation_data
    except Exception:
        return {}

    clean = pd.read_parquet(paths.CLEAN_PARQUET)
    subsets = ablation_data.tox21_subsets("scaffold", 0)
    mask = subsets == "valid"
    smiles = clean.loc[mask, "canonical_smiles"].tolist()
    if not smiles:
        return {}

    bundle = get_bundle()
    frame = bundle.toxicity.predict_frame(smiles)
    out: dict[str, Any] = {}
    for j, endpoint in enumerate(endpoints):
        labels = clean.loc[mask, endpoint].to_numpy(dtype=float)
        scores = frame[endpoint].to_numpy(dtype=float)
        observed = ~np.isnan(labels) & ~np.isnan(scores)
        if observed.sum() < 20 or len(np.unique(labels[observed])) < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(scores[observed], labels[observed])
        out[endpoint] = iso
    return out


def get_bundle() -> Bundle:
    """Load models, protein embeddings and the applicability reference once."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    if not REGISTRY_PATH.exists():
        raise ServiceError(
            f"{REGISTRY_PATH.relative_to(ROOT)} missing. Run "
            "`python src/38_export_models.py` to build the shipped models.")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    # Every artefact failure below is re-raised as ServiceError. The loaders
    # raise ModelContractError and ApplicabilityError, which are siblings of
    # ServiceError rather than subclasses, so without this the endpoints -
    # which catch ServiceError - returned a bare HTTP 500 and threw away the
    # one message that tells the operator what to run. The messages are already
    # actionable; the only thing missing was letting them reach the caller.
    try:
        toxicity = model_io.load_model("toxicity_gin")
        affinity = model_io.load_model("affinity_frozen_mlp")
        multitask = model_io.load_model("multitask_gin")
        references = applicability.load_reference(REFERENCE_PATH)
        embeddings = model_io.load_protein_embeddings()
    except (model_io.ModelContractError, applicability.ApplicabilityError) as exc:
        raise ServiceError(f"{type(exc).__name__}: {exc}") from exc
    except (OSError, KeyError, ValueError) as exc:
        # A truncated or hand-edited artefact lands here; the file name is the
        # part the operator needs, so it must not be swallowed either.
        raise ServiceError(
            f"could not load the shipped models ({type(exc).__name__}: {exc}). "
            "Re-run `python src/38_export_models.py`.") from exc

    targets = [{"target_id": accession, "dim": int(vec.shape[0])}
               for accession, vec in sorted(embeddings.items())]

    _BUNDLE = Bundle(
        registry=registry, references=references, toxicity=toxicity,
        affinity=affinity, multitask=multitask,
        endpoints=list(toxicity.endpoints), targets=targets,
    )
    # Calibration needs the loaded toxicity model, so it is fitted after the
    # bundle exists rather than inside the constructor.
    _BUNDLE.calibrators = _fit_calibrators(_BUNDLE.endpoints)
    return _BUNDLE


def list_targets() -> dict:
    bundle = get_bundle()
    return {
        "n_targets": len(bundle.targets),
        "embedding_model": bundle.registry["protein_embeddings"]["model"],
        "targets": bundle.targets,
    }


def list_models() -> dict:
    """The registry plus whether each entry actually loaded."""
    bundle = get_bundle()
    loaded = {"toxicity_gin": bundle.toxicity is not None,
              "affinity_frozen_mlp": bundle.affinity is not None,
              "multitask_gin": bundle.multitask is not None}
    models = []
    for name, entry in bundle.registry["models"].items():
        models.append({**entry, "name": name, "loaded": loaded.get(name, False)})
    return {
        "format_version": bundle.registry["format_version"],
        "models": models,
        "protein_embeddings": bundle.registry["protein_embeddings"],
        "applicability_reference": bundle.registry.get("applicability_reference"),
        "calibration": {
            "available": bool(bundle.calibrators),
            "method": "isotonic regression on the scaffold validation fold",
            "n_endpoints_calibrated": len(bundle.calibrators),
            "note": ("raw Tox21 labels absent, so scores are uncalibrated"
                     if not bundle.calibrators else
                     "expected calibration error falls 0.0653 -> 0.0252 "
                     "(results/tox_calibration.csv)"),
        },
    }


def screen(smiles: list[str], target_id: str | None,
           threshold: float = applicability.DEFAULT_DOMAIN_THRESHOLD) -> dict:
    """Score a batch of molecules; one output row per input row, in order."""
    bundle = get_bundle()
    if not smiles:
        raise ServiceError("no molecules supplied")
    known = {t["target_id"] for t in bundle.targets}
    if target_id is not None and target_id not in known:
        raise ServiceError(
            f"unknown target {target_id!r}. This build ships embeddings for "
            f"{len(known)} proteins; a target outside that set cannot be scored "
            "and is not approximated by a similar one.")

    with bundle._lock:
        tox_frame = bundle.toxicity.predict_frame(smiles)
        affinity = (bundle.affinity.predict(smiles, [target_id] * len(smiles))
                    if target_id else None)

    tox_domain = applicability.assess(smiles, bundle.references["toxicity"],
                                      threshold=threshold)
    aff_domain = applicability.assess(smiles, bundle.references["affinity"],
                                      threshold=threshold)
    descriptors = _descriptors(smiles)

    rows = []
    for i, smi in enumerate(smiles):
        raw = {ep: _finite(tox_frame.iloc[i][ep]) for ep in bundle.endpoints}
        parsed = any(v is not None for v in raw.values())
        calibrated = {
            ep: (round(float(bundle.calibrators[ep].predict([raw[ep]])[0]), 4)
                 if ep in bundle.calibrators and raw[ep] is not None else None)
            for ep in bundle.endpoints
        }
        primary = calibrated if bundle.calibrators else raw
        risk = [v for v in primary.values() if v is not None]
        rows.append({
            "input_index": i,
            "smiles": smi,
            "error": None if parsed else "could not be parsed as a molecule",
            "toxicity_raw": raw,
            "toxicity_calibrated": calibrated,
            "toxicity_risk_max": round(max(risk), 4) if risk else None,
            "affinity_kiba_score": (_finite(affinity[i]) if affinity is not None
                                    else None),
            "applicability_toxicity": tox_domain[i],
            "applicability_affinity": aff_domain[i] if target_id else None,
            "descriptors": descriptors[i],
        })

    in_domain = sum(1 for r in rows if r["applicability_toxicity"]["in_domain"])
    return {
        "n_input": len(smiles),
        "n_scored": sum(1 for r in rows if r["error"] is None),
        "n_in_domain": in_domain,
        "target_id": target_id,
        "domain_threshold": threshold,
        # Quoting only the 0.0-17.2 range misleads: 75% of real KIBA labels sit
        # within +-0.6 of the median and the interquartile range is 0.75, so
        # predictions that all look "the same" are in fact tracking the data.
        # The spread is the number a reader needs, not the extremes.
        "affinity_unit": ("KIBA score; NOT pKd. Half the measured values lie in "
                          "11.20-11.95 and 75% within +-0.6 of the 11.50 median, "
                          "so differences of ~0.5 are large for this target "
                          "variable"
                          if target_id else None),
        "toxicity_primary": "calibrated" if bundle.calibrators else "raw",
        "endpoints": bundle.endpoints,
        "rows": rows,
    }


def _finite(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(out) else round(out, 4)


#: Descriptor columns the interface shows. `properties.PROPERTIES` is the
#: project's single definition of each one; QED and the heavy-atom count are
#: added here because only the interface needs them (point size and a sanity
#: column), and `descriptor_table` cannot be reused directly because it raises
#: on an unparseable SMILES while this layer must keep the row.
_EXTRA_DESCRIPTORS = ("qed", "heavy_atoms")


def descriptor_columns() -> list[str]:
    return list(properties.PROPERTIES) + list(_EXTRA_DESCRIPTORS)


def _descriptors(smiles: list[str]) -> list[dict]:
    """RDKit descriptors, computed rather than predicted.

    Kept separate in the payload so the interface can mark them as exact: a user
    must be able to tell which columns are measurements of the structure and
    which are model output.
    """
    from rdkit import Chem
    from rdkit.Chem import QED

    columns = descriptor_columns()
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            out.append({c: None for c in columns})
            continue
        row = {name: round(float(fn(mol)), 4)
               for name, fn in properties.PROPERTIES.items()}
        row["qed"] = round(float(QED.qed(mol)), 4)
        row["heavy_atoms"] = int(mol.GetNumHeavyAtoms())
        out.append(row)
    return out


def molecule_detail(smiles: str) -> dict:
    """2D depiction, per-atom attribution and the nearest training molecules."""
    from rdkit import Chem

    bundle = get_bundle()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ServiceError(f"could not parse {smiles!r}")

    detail: dict = {
        "smiles": Chem.MolToSmiles(mol),
        "n_atoms": mol.GetNumAtoms(),
        "svg": _svg(mol),
        "structural_alerts": _alerts(smiles),
    }

    with bundle._lock:
        try:
            from tox21lib import attribution

            result = attribution.attribute(bundle.toxicity.model, smiles,
                                           device=bundle.toxicity.device)
            # `attribute` names this key "attribution": baseline minus masked,
            # shape (n_atoms, n_endpoints).
            drops = np.asarray(result["attribution"], dtype=float)
            detail["attribution"] = {
                "endpoints": bundle.endpoints,
                "per_atom": [[round(float(v), 5) for v in row] for row in drops],
                "max_per_atom": [round(float(v), 5) for v in drops.max(axis=1)],
                "baseline_probs": [round(float(v), 5)
                                   for v in np.asarray(result["baseline_probs"],
                                                       dtype=float)],
                "method": "single-atom masking; positive means removing the "
                          "atom lowered the predicted probability",
            }
        except Exception as exc:
            detail["attribution"] = {"error": f"{type(exc).__name__}: {exc}"}

    nn = applicability.nearest_tanimoto([smiles], bundle.references["toxicity"])
    detail["nearest_training_neighbour"] = {
        "tanimoto": _finite(nn["nn_tanimoto"][0]),
        "reference_index": int(nn["nn_index"][0]),
        "reference_split": bundle.references["toxicity"].split,
        "reference_size": len(bundle.references["toxicity"]),
        "note": "the reference ships as packed fingerprints, so the neighbour's "
                "structure is not recoverable from it",
    }
    return detail


def _svg(mol) -> str:
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(380, 280)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _alerts(smiles: str) -> dict:
    try:
        from tox21lib import attribution

        return {k: v for k, v in attribution.alert_atoms(smiles).items() if v}
    except Exception:
        return {}


def health() -> dict:
    try:
        bundle = get_bundle()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "models": len(bundle.registry["models"]),
        "targets": len(bundle.targets),
        "endpoints": len(bundle.endpoints),
        "calibrated_endpoints": len(bundle.calibrators),
        "applicability_reference_sizes": {
            name: len(ref) for name, ref in bundle.references.items()},
    }
