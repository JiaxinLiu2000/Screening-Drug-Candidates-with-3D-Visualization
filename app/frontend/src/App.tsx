/**
 * Screening workbench.
 *
 * The column order is the argument this project is making: applicability sits
 * next to the predictions, not in a tooltip. Rows outside the training
 * distribution are struck through and sorted last, because the abstention study
 * showed the model's own confidence does not detect those cases at all.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  getModels,
  getTargets,
  submitScreen,
  waitForJob,
  type ModelsResponse,
  type ScreenResult,
  type ScreenRow,
  type TargetInfo,
} from "./api";
import { ScatterPlot } from "./ScatterPlot";
import { MoleculeDetail } from "./MoleculeDetail";
import "./App.css";


// Eight Tox21-like molecules plus three the models should refuse: ethanol is
// far too small to resemble the training set, the dipeptide is the wrong class
// of chemistry entirely, and the last string does not parse. Without them a
// demo would only ever show the happy path, which is the opposite of the point
// this interface is making.
const EXAMPLE = [
  "CC(=O)Oc1ccccc1C(=O)O",
  "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
  "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
  "C1=CC=C(C=C1)[N+](=O)[O-]",
  "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
  "CCN(CC)CCNC(=O)c1ccc(N)cc1",
  "O=C(Nc1ccc(Cl)cc1)Nc1ccccc1",
  "CC1=C(C(=O)Nc2ccccc2)SC(=N1)N",
  "CCO",
  "NC(C)C(=O)NC(CC(=O)O)C(=O)O",
  "this-is-not-a-molecule",
].join("\n");

type SortKey = "input" | "affinity" | "toxicity" | "domain";

export default function App() {
  const [smilesText, setSmilesText] = useState(EXAMPLE);
  const [targets, setTargets] = useState<TargetInfo[]>([]);
  const [targetId, setTargetId] = useState<string>("");
  const [threshold, setThreshold] = useState(0.4);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [result, setResult] = useState<ScreenResult | null>(null);
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [focus, setFocus] = useState<ScreenRow | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("input");

  // One controller per in-flight poll. Aborting on unmount stops the request
  // loop instead of leaving it to update state on a component that is gone;
  // aborting on resubmit stops an earlier batch from overwriting a later one.
  const pollRef = useRef<AbortController | null>(null);
  useEffect(() => () => pollRef.current?.abort(), []);

  useEffect(() => {
    getTargets()
      .then((r) => {
        setTargets(r.targets);
        if (r.targets.length > 0) setTargetId(r.targets[0].target_id);
      })
      .catch((e: Error) => setError(e.message));
    getModels()
      .then(setModels)
      .catch((e: Error) => setError(e.message));
  }, []);

  const run = async () => {
    const smiles = smilesText
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    if (smiles.length === 0) {
      setError("enter at least one SMILES string");
      return;
    }
    pollRef.current?.abort();
    const controller = new AbortController();
    pollRef.current = controller;
    setBusy(true);
    setError(null);
    setResult(null);
    setFocus(null);
    setSelected(new Set());
    try {
      const jobId = await submitScreen(smiles, targetId || null, threshold);
      setStatus(`job ${jobId} running on ${smiles.length} molecules…`);
      const job = await waitForJob(jobId, undefined, 700, {
        signal: controller.signal,
      });
      if (job.status === "error") throw new Error(job.error ?? "job failed");
      setResult(job.result ?? null);
      setStatus(`job ${jobId} finished in ${job.seconds ?? "?"}s`);
    } catch (e) {
      if (controller.signal.aborted) return;   // superseded or unmounted
      setError((e as Error).message);
      setStatus("");
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  };

  const rows = useMemo(() => {
    if (!result) return [];
    const copy = [...result.rows];
    const score = (r: ScreenRow) => {
      switch (sortKey) {
        case "affinity":
          return -(r.affinity_kiba_score ?? -Infinity);
        case "toxicity":
          return r.toxicity_risk_max ?? Infinity;
        case "domain":
          return -(r.applicability_toxicity.nn_tanimoto ?? -1);
        default:
          return r.input_index;
      }
    };
    // Out-of-domain rows sink to the bottom under every sort order: ranking a
    // molecule the model cannot score into first place is the specific failure
    // this interface exists to prevent.
    copy.sort((a, b) => {
      const da = a.applicability_toxicity.in_domain ? 0 : 1;
      const db = b.applicability_toxicity.in_domain ? 0 : 1;
      if (da !== db) return da - db;
      return score(a) - score(b);
    });
    return copy;
  }, [result, sortKey]);

  return (
    <div className="shell">
      <header>
        <h1>Dual-objective screening</h1>
        <p className="muted">
          Binding affinity and twelve Tox21 endpoints, with the distance from
          the training distribution reported for every molecule.
        </p>
      </header>

      <section className="controls">
        <label className="field">
          <span>Molecules (one SMILES per line)</span>
          <textarea
            value={smilesText}
            onChange={(e) => setSmilesText(e.target.value)}
            rows={9}
            spellCheck={false}
          />
        </label>

        <div className="side">
          <label className="field">
            <span>Target protein ({targets.length} with embeddings)</span>
            <select value={targetId} onChange={(e) => setTargetId(e.target.value)}>
              <option value="">— toxicity only —</option>
              {targets.map((t) => (
                <option key={t.target_id} value={t.target_id}>
                  {t.target_id}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>
              Applicability threshold (nearest-neighbour Tanimoto): {threshold.toFixed(2)}
            </span>
            <input
              type="range"
              min={0}
              max={0.9}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
          </label>

          <button onClick={run} disabled={busy}>
            {busy ? "scoring…" : "Screen"}
          </button>
          {status && <p className="muted small">{status}</p>}
          {error && <p className="error">{error}</p>}
        </div>
      </section>

      {models && <ModelBar models={models} />}

      {result && (
        <>
          <section className="summary">
            <Stat label="molecules" value={result.n_input} />
            <Stat label="scored" value={result.n_scored} />
            <Stat
              label="in domain"
              value={`${result.n_in_domain} / ${result.n_scored}`}
              warn={result.n_in_domain < result.n_scored}
            />
            <Stat
              label="toxicity scores"
              value={result.toxicity_primary}
            />
            {result.affinity_unit && (
              <Stat label="affinity unit" value={result.affinity_unit} wide />
            )}
          </section>

          <section className="plot-row">
            <ScatterPlot
              rows={result.rows}
              hasAffinity={Boolean(result.target_id)}
              selected={selected}
              onSelect={setSelected}
              onHover={(r) => r && setFocus(r)}
            />
            <MoleculeDetail row={focus} endpoints={result.endpoints} />
          </section>

          <section className="table-wrap">
            <div className="table-head">
              <h2>Results</h2>
              <label className="inline">
                sort by{" "}
                <select
                  value={sortKey}
                  onChange={(e) => setSortKey(e.target.value as SortKey)}
                >
                  <option value="input">input order</option>
                  <option value="affinity">strongest binding</option>
                  <option value="toxicity">lowest toxicity risk</option>
                  <option value="domain">closest to training set</option>
                </select>
              </label>
              {selected.size > 0 && (
                <span className="muted small">
                  {selected.size} selected in plot
                </span>
              )}
            </div>
            <ResultTable
              rows={rows}
              hasAffinity={Boolean(result.target_id)}
              selected={selected}
              focus={focus}
              onFocus={setFocus}
            />
          </section>
        </>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  warn,
  wide,
}: {
  label: string;
  value: string | number;
  warn?: boolean;
  wide?: boolean;
}) {
  return (
    <div className={`stat${wide ? " wide" : ""}${warn ? " warn" : ""}`}>
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

function ModelBar({ models }: { models: ModelsResponse }) {
  return (
    <section className="modelbar">
      {models.models.map((m) => (
        <span key={m.name} className={m.loaded ? "chip ok" : "chip bad"}>
          {m.name} · {m.task} · {(m.bytes / 1e6).toFixed(1)} MB ·{" "}
          {m.sha256.slice(0, 8)}
        </span>
      ))}
      <span className={models.calibration.available ? "chip ok" : "chip warn"}>
        calibration: {models.calibration.available ? "isotonic" : "unavailable"}
      </span>
    </section>
  );
}

function ResultTable({
  rows,
  hasAffinity,
  selected,
  focus,
  onFocus,
}: {
  rows: ScreenRow[];
  hasAffinity: boolean;
  selected: Set<number>;
  focus: ScreenRow | null;
  onFocus: (row: ScreenRow) => void;
}) {
  return (
    <table className="results">
      <thead>
        <tr>
          <th>#</th>
          <th>SMILES</th>
          {hasAffinity && <th>affinity (KIBA)</th>}
          <th>tox risk</th>
          <th>nn Tanimoto</th>
          <th>applicability</th>
          <th>QED</th>
          <th>MW</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const ad = row.applicability_toxicity;
          const classes = [
            row.error ? "failed" : "",
            ad.in_domain ? "" : "out-domain",
            selected.has(row.input_index) ? "selected" : "",
            focus?.input_index === row.input_index ? "focused" : "",
          ]
            .filter(Boolean)
            .join(" ");
          return (
            <tr
              key={row.input_index}
              className={classes}
              onClick={() => onFocus(row)}
            >
              <td className="num">{row.input_index}</td>
              <td className="smiles-cell">
                {row.error ? (
                  <span className="error small">
                    {row.smiles} — {row.error}
                  </span>
                ) : (
                  row.smiles
                )}
              </td>
              {hasAffinity && (
                <td className="num">
                  {row.affinity_kiba_score?.toFixed(2) ?? "—"}
                </td>
              )}
              <td className="num">{row.toxicity_risk_max?.toFixed(3) ?? "—"}</td>
              <td className="num">{ad.nn_tanimoto?.toFixed(3) ?? "—"}</td>
              <td className={ad.in_domain ? "ok small" : "warn small"}>
                {ad.in_domain ? "in domain" : "do not trust"}
              </td>
              <td className="num">{row.descriptors.qed?.toFixed(2) ?? "—"}</td>
              <td className="num">{row.descriptors.mol_weight?.toFixed(0) ?? "—"}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
