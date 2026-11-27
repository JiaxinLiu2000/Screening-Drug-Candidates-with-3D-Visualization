"""FastAPI surface for the screening tool.

Deliberately small: five endpoints, an in-process job store, no queue and no
database. The project's own result is that the interesting part of this tool is
which predictions it refuses to stand behind, not how many workers it can span,
so the complexity budget goes to the scoring contract instead of the plumbing.

Jobs exist because a batch of a few hundred molecules takes long enough that a
browser request would look frozen, not because the work is distributed. They
live in a dictionary and are lost on restart, which is stated in the response
rather than hidden.

    python -m uvicorn app.backend.main:app --port 8000 --reload
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.backend import service  # noqa: E402

#: A batch larger than this is refused outright rather than silently truncated.
MAX_MOLECULES = 2000

app = FastAPI(
    title="Dual-objective screening",
    version="1.0.0",
    description=(
        "Scores molecules for binding affinity against a chosen protein and for "
        "twelve Tox21 toxicity endpoints, and reports how far each molecule sits "
        "from the training distribution."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=2)


class ScreenRequest(BaseModel):
    smiles: list[str] = Field(..., min_length=1,
                              description="one SMILES per entry; order is kept")
    target_id: str | None = Field(
        None, description="UniProt accession; omit to score toxicity only")
    domain_threshold: float = Field(
        0.4, ge=0.0, le=1.0,
        description="nearest-neighbour Tanimoto below which a row is flagged "
                    "out of domain")


@app.get("/api/health")
def health() -> dict:
    return service.health()


@app.get("/api/targets")
def targets() -> dict:
    try:
        return service.list_targets()
    except service.ServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/models")
def models() -> dict:
    try:
        return service.list_models()
    except service.ServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _run_job(job_id: str, request: ScreenRequest) -> None:
    started = time.time()
    try:
        result = service.screen(request.smiles, request.target_id,
                               threshold=request.domain_threshold)
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="done", result=result,
                                 seconds=round(time.time() - started, 2))
    except Exception as exc:
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="error",
                                 error=f"{type(exc).__name__}: {exc}",
                                 seconds=round(time.time() - started, 2))


@app.post("/api/screen", status_code=202)
def screen(request: ScreenRequest) -> dict:
    cleaned = [s.strip() for s in request.smiles if s and s.strip()]
    if not cleaned:
        raise HTTPException(status_code=422,
                            detail="no non-empty SMILES supplied")
    if len(cleaned) > MAX_MOLECULES:
        raise HTTPException(
            status_code=413,
            detail=f"{len(cleaned)} molecules exceeds the {MAX_MOLECULES} limit. "
                   "The batch is refused rather than truncated, so you can "
                   "split it yourself and keep track of which rows ran.")
    try:
        service.get_bundle()
    except service.ServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {"job_id": job_id, "status": "running",
                         "n_input": len(cleaned),
                         "submitted_at": time.time()}
    _POOL.submit(_run_job, job_id,
                 request.model_copy(update={"smiles": cleaned}))
    return {"job_id": job_id, "status": "running", "n_input": len(cleaned)}


@app.get("/api/screen/{job_id}")
def screen_status(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"no job {job_id}. Jobs are held in memory and are lost when "
                   "the server restarts.")
    return job


@app.get("/api/molecule")
def molecule(smiles: str) -> dict:
    try:
        return service.molecule_detail(smiles)
    except service.ServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/molecule/svg")
def molecule_svg(smiles: str) -> Response:
    try:
        detail = service.molecule_detail(smiles)
    except service.ServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(content=detail["svg"], media_type="image/svg+xml")


@app.get("/api/structures")
def structures() -> dict:
    """Co-crystal structures available for the 3D view.

    Reads the receptor directory produced by the docking work; each entry has a
    prepared receptor PDB, the crystal ligand, and a pocket definition.
    """
    receptors = ROOT / "receptors"
    if not receptors.exists():
        return {"n_structures": 0, "structures": []}
    out = []
    for pocket in sorted(receptors.glob("*_pocket.json")):
        pdb_id = pocket.name.split("_")[0]
        receptor = receptors / f"{pdb_id}_receptor.pdb"
        ligand = receptors / f"{pdb_id}_ligand.sdf"
        if receptor.exists() and ligand.exists():
            out.append({"pdb_id": pdb_id,
                        "receptor": f"/api/structures/{pdb_id}/receptor",
                        "ligand": f"/api/structures/{pdb_id}/ligand"})
    return {"n_structures": len(out), "structures": out}


def _structure_file(pdb_id: str, kind: str) -> Path:
    suffix = {"receptor": "_receptor.pdb", "ligand": "_ligand.sdf"}[kind]
    path = ROOT / "receptors" / f"{pdb_id}{suffix}"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found")
    return path


@app.get("/api/structures/{pdb_id}/receptor")
def structure_receptor(pdb_id: str) -> Response:
    return Response(content=_structure_file(pdb_id, "receptor").read_text(),
                    media_type="chemical/x-pdb")


@app.get("/api/structures/{pdb_id}/ligand")
def structure_ligand(pdb_id: str) -> Response:
    return Response(content=_structure_file(pdb_id, "ligand").read_text(),
                    media_type="chemical/x-mdl-sdfile")
