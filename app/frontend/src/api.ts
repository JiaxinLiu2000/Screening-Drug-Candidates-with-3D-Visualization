/**
 * Typed client for the screening API.
 *
 * `applicability` is required on every row rather than optional. The project's
 * abstention study found the models are confidently wrong outside their
 * training distribution, so an interface that can forget to render that field
 * is an interface that will eventually present a confident wrong answer as a
 * plain one.
 */

// Optional chaining so the module can also be imported outside Vite (tests,
// scripts), where import.meta.env does not exist and a bare access throws.
const BASE = import.meta.env?.VITE_API_BASE ?? "http://127.0.0.1:8000";

export interface Applicability {
  nn_tanimoto: number | null;
  nn_index: number | null;
  in_domain: boolean;
  advice: string;
}

export interface Descriptors {
  mol_weight: number | null;
  clogp: number | null;
  tpsa: number | null;
  ring_count: number | null;
  aromatic_ring_count: number | null;
  rotatable_bonds: number | null;
  fraction_csp3: number | null;
  qed: number | null;
  heavy_atoms: number | null;
}

export interface ScreenRow {
  input_index: number;
  smiles: string;
  error: string | null;
  toxicity_raw: Record<string, number | null>;
  toxicity_calibrated: Record<string, number | null>;
  toxicity_risk_max: number | null;
  affinity_kiba_score: number | null;
  applicability_toxicity: Applicability;
  applicability_affinity: Applicability | null;
  descriptors: Descriptors;
}

export interface ScreenResult {
  n_input: number;
  n_scored: number;
  n_in_domain: number;
  target_id: string | null;
  domain_threshold: number;
  affinity_unit: string | null;
  toxicity_primary: "calibrated" | "raw";
  endpoints: string[];
  rows: ScreenRow[];
}

export interface Job {
  job_id: string;
  status: "running" | "done" | "error";
  n_input: number;
  seconds?: number;
  error?: string;
  result?: ScreenResult;
}

export interface TargetInfo {
  target_id: string;
  dim: number;
}

export interface ModelEntry {
  name: string;
  file: string;
  task: string;
  description: string;
  sha256: string;
  bytes: number;
  loaded: boolean;
  metrics: Record<string, unknown>;
}

export interface ModelsResponse {
  format_version: number;
  models: ModelEntry[];
  protein_embeddings: Record<string, unknown>;
  applicability_reference: Record<string, unknown> | null;
  calibration: {
    available: boolean;
    method: string;
    n_endpoints_calibrated: number;
    note: string;
  };
}

export interface MoleculeDetail {
  smiles: string;
  n_atoms: number;
  svg: string;
  structural_alerts: Record<string, number[]>;
  attribution:
    | {
        endpoints: string[];
        per_atom: number[][];
        max_per_atom: number[];
        method: string;
      }
    | { error: string };
  nearest_training_neighbour: {
    tanimoto: number | null;
    reference_index: number;
    reference_split: string;
    reference_size: number;
    note: string;
  };
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? `request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const getHealth = () => get<Record<string, unknown>>("/api/health");
export const getTargets = () =>
  get<{ n_targets: number; embedding_model: string; targets: TargetInfo[] }>(
    "/api/targets",
  );
export const getModels = () => get<ModelsResponse>("/api/models");
export const getMolecule = (smiles: string) =>
  get<MoleculeDetail>(`/api/molecule?smiles=${encodeURIComponent(smiles)}`);
export const getStructures = () =>
  get<{ n_structures: number; structures: { pdb_id: string; receptor: string; ligand: string }[] }>(
    "/api/structures",
  );
export const structureUrl = (path: string) => `${BASE}${path}`;

export async function submitScreen(
  smiles: string[],
  targetId: string | null,
  domainThreshold: number,
): Promise<string> {
  const response = await fetch(`${BASE}/api/screen`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      smiles,
      target_id: targetId,
      domain_threshold: domainThreshold,
    }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? `submit failed: ${response.status}`);
  }
  const body = (await response.json()) as { job_id: string };
  return body.job_id;
}

export const getJob = (jobId: string) => get<Job>(`/api/screen/${jobId}`);

function delay(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

/**
 * Poll until the job leaves the running state.
 *
 * Bounded on purpose. An unbounded loop keeps polling after the component that
 * started it unmounts, and a job wedged in `running` would poll until the tab
 * is closed. `signal` covers the first case, `timeoutMs` the second; hitting
 * either throws rather than returning a half-finished job, so the caller
 * cannot mistake a timeout for an empty result.
 */
export async function waitForJob(
  jobId: string,
  onTick?: (job: Job) => void,
  intervalMs = 700,
  options: { signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<Job> {
  const { signal, timeoutMs = 10 * 60 * 1000 } = options;
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (signal?.aborted) throw new Error("polling aborted");
    const job = await getJob(jobId);
    onTick?.(job);
    if (job.status !== "running") return job;
    if (Date.now() >= deadline) {
      throw new Error(
        `job ${jobId} was still running after ` +
          `${(timeoutMs / 1000).toFixed(timeoutMs < 10_000 ? 1 : 0)}s; ` +
          "stopped polling. The server may still be working - re-check with " +
          `GET /api/screen/${jobId}.`,
      );
    }
    await delay(intervalMs);
  }
}
