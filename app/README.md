# Screening application

A two-process application: a FastAPI service that loads the committed models,
and a Vite/React interface that talks to it.

## What it does

Given a list of SMILES strings and a target protein, it returns for every
molecule:

- a predicted binding affinity against the chosen protein (KIBA score),
- twelve Tox21 endpoint probabilities, isotonic-calibrated when the raw Tox21
  labels are available locally,
- the nearest-neighbour Tanimoto similarity to the training set, and a verdict
  on whether the prediction should be trusted at all,
- RDKit descriptors, which are computed rather than predicted and are marked as
  such in the payload.

Molecules that fail to parse keep their row with `error` set. The row order of
the response always matches the order of the request, so a caller can line
results up against its own input.

## Why applicability is a first-class column

`results/abstention.json` records that the model's own confidence is useless as
a guide to when it is wrong: abstaining on the molecules whose predicted
probability is closest to 0.5 scored PR-AUC 0.2079, which is *worse* than
abstaining at random (0.3983). Abstaining on the molecules furthest from the
training set scored 0.4904, the best of the four signals tested.

So the interface sorts out-of-domain rows to the bottom under every sort order,
strikes their SMILES through, and draws them as hollow rings in the scatter
plot. An interface that put this in a tooltip would eventually present a
confidently wrong answer as an ordinary one.

## Setup

The backend needs the committed models. From the repository root:

```bash
python src/38_export_models.py        # writes models/ if it is not there yet
python -m uvicorn app.backend.main:app --port 8000
```

In a second terminal:

```bash
cd app/frontend
npm install
npm run dev                            # http://localhost:5173
```

`VITE_API_BASE` overrides the API origin; it defaults to
`http://127.0.0.1:8000`.

### What a fresh clone has and does not have

| Needed for | Shipped in the repository? |
|---|---|
| Toxicity, affinity and multi-task predictions | yes — `models/*.pt` |
| Protein embeddings for 229 targets | yes — `models/protein_embeddings.npz` |
| Applicability-domain reference | yes — `models/applicability_reference.npz` |
| 3D co-crystal structures | yes — `receptors/` |
| Isotonic calibration of toxicity scores | **no** — needs the raw Tox21 table |

Without the raw Tox21 table the API reports uncalibrated scores and says so in
`GET /api/models` and in the interface's status bar, rather than presenting raw
scores as if they were calibrated probabilities. Run
`python src/00_fetch_data.py` to download the data and enable calibration.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | models loaded, reference sizes |
| GET | `/api/targets` | the 229 proteins with embeddings |
| GET | `/api/models` | registry, sha256 per model, calibration status |
| POST | `/api/screen` | submit a batch, returns `job_id` |
| GET | `/api/screen/{job_id}` | poll; `status` is `running`, `done` or `error` |
| GET | `/api/molecule?smiles=` | 2D SVG, atom attribution, nearest neighbour |
| GET | `/api/structures` | co-crystal structures available for the 3D view |

Jobs are held in a dictionary and are lost when the server restarts; the 404
response says so instead of pretending the job never existed. Batches larger
than 2000 molecules are refused rather than truncated, so the caller keeps
track of which rows ran.

## Deliberate omissions

- **No queue, no database.** The design sketch had Redis-RQ and PostgreSQL.
  Neither earns its place here: inference is a few seconds for a few hundred
  molecules, and persistence would add operational surface without changing any
  answer the tool gives.
- **No live docking.** The 3D panel shows a reference co-crystal — receptor
  plus its own crystal ligand — not a docked pose of the queried molecule.
  Docking takes tens of seconds per molecule, so it runs offline through
  `src/41_docking_scaleup.py`. The panel says this on screen; implying the
  displayed pose belongs to the query would be the easiest way for it to
  mislead.
- **Plotly replaced by hand-written SVG.** The only interactions needed are
  hover and rectangle selection, and a dependency-free plot was worth more than
  a configurable one.

## Units

The affinity number is a **KIBA score** (median 11.50, interquartile range
11.20–11.95; 75% of measured values lie within ±0.6 of the median),
not a pKd and not a binding constant. The API labels it on every response and
the interface repeats the label on the axis, because a reader who assumes pKd
will be wrong by orders of magnitude.
