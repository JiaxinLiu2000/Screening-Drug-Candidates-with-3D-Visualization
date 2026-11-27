/**
 * Detail panel: 2D depiction with atom attribution, and the 3D pocket view.
 *
 * The nearest training neighbour is shown next to the prediction on purpose, so
 * a user can judge the extrapolation distance themselves rather than taking the
 * in-domain flag on faith.
 */

import { useEffect, useRef, useState } from "react";
import { createViewer } from "3dmol";
import {
  getMolecule,
  getStructures,
  structureUrl,
  type MoleculeDetail as Detail,
  type ScreenRow,
} from "./api";

interface Props {
  row: ScreenRow | null;
  endpoints: string[];
}

export function MoleculeDetail({ row, endpoints }: Props) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!row || row.error) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setBusy(true);
    setError(null);
    getMolecule(row.smiles)
      .then((d) => {
        if (!cancelled) setDetail(d);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, [row]);

  if (!row) {
    return (
      <aside className="detail detail-empty">
        Select a row to see its structure, per-atom attribution and distance to
        the training set.
      </aside>
    );
  }

  const applicability = row.applicability_toxicity;

  return (
    <aside className="detail">
      <h2>Molecule detail</h2>
      <code className="smiles">{row.smiles}</code>

      {busy && <p className="muted">rendering…</p>}
      {error && <p className="error">{error}</p>}

      {detail && (
        <>
          <div
            className="depiction"
            // The SVG comes from our own RDKit call on localhost, not from user
            // input, and RDKit emits markup rather than a parsable structure.
            dangerouslySetInnerHTML={{ __html: detail.svg }}
          />

          <h3>Distance to the training set</h3>
          <p className={applicability.in_domain ? "ok" : "warn"}>
            Nearest training molecule: Tanimoto{" "}
            <strong>
              {applicability.nn_tanimoto === null
                ? "n/a"
                : applicability.nn_tanimoto.toFixed(3)}
            </strong>{" "}
            over {detail.nearest_training_neighbour.reference_size} molecules in
            the {detail.nearest_training_neighbour.reference_split} training
            fold.
          </p>
          <p className="muted small">{applicability.advice}</p>

          <h3>Atom attribution</h3>
          {"error" in detail.attribution ? (
            <p className="error small">{detail.attribution.error}</p>
          ) : (
            <AttributionTable
              perAtom={detail.attribution.max_per_atom}
              method={detail.attribution.method}
              alerts={detail.structural_alerts}
            />
          )}

          <h3>Toxicity endpoints</h3>
          <EndpointTable row={row} endpoints={endpoints} />

          <PocketView />
        </>
      )}
    </aside>
  );
}

function AttributionTable({
  perAtom,
  method,
  alerts,
}: {
  perAtom: number[];
  method: string;
  alerts: Record<string, number[]>;
}) {
  const ranked = perAtom
    .map((value, atom) => ({ atom, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 5);
  const alertAtoms = new Set(Object.values(alerts).flat());
  const max = Math.max(...perAtom.map(Math.abs), 1e-9);

  return (
    <>
      <table className="mini">
        <thead>
          <tr>
            <th>atom</th>
            <th>largest drop</th>
            <th />
            <th>alert</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map(({ atom, value }) => (
            <tr key={atom}>
              <td>{atom}</td>
              <td className="num">{value.toFixed(4)}</td>
              <td>
                <span
                  className="bar"
                  style={{ width: `${(Math.abs(value) / max) * 60}px` }}
                />
              </td>
              <td>{alertAtoms.has(atom) ? "yes" : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">{method}</p>
      {Object.keys(alerts).length > 0 && (
        <p className="muted small">
          Structural alerts matched: {Object.keys(alerts).join(", ")}
        </p>
      )}
      <p className="muted small">
        Single-atom masking understates groups: removing one atom of a
        three-atom epoxide leaves a near-identical environment. Treat these as a
        hint, not an explanation.
      </p>
    </>
  );
}

function EndpointTable({ row, endpoints }: { row: ScreenRow; endpoints: string[] }) {
  return (
    <table className="mini">
      <thead>
        <tr>
          <th>endpoint</th>
          <th>calibrated</th>
          <th>raw</th>
        </tr>
      </thead>
      <tbody>
        {endpoints.map((endpoint) => {
          const calibrated = row.toxicity_calibrated[endpoint];
          const raw = row.toxicity_raw[endpoint];
          return (
            <tr key={endpoint}>
              <td>{endpoint}</td>
              <td className="num">
                {calibrated === null || calibrated === undefined
                  ? "—"
                  : calibrated.toFixed(3)}
              </td>
              <td className="num muted">
                {raw === null || raw === undefined ? "—" : raw.toFixed(3)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/**
 * 3Dmol.js view of a co-crystal structure.
 *
 * The docked pose of the *queried* molecule is not shown, because docking takes
 * tens of seconds per molecule and is not wired into the request path. What is
 * shown is the reference co-crystal: receptor plus its crystal ligand. Claiming
 * otherwise would be the easiest way for this panel to mislead.
 */
function PocketView() {
  const host = useRef<HTMLDivElement>(null);
  const [structures, setStructures] = useState<{ pdb_id: string; receptor: string; ligand: string }[]>(
    [],
  );
  const [chosen, setChosen] = useState<string>("");
  const [note, setNote] = useState<string>("");

  useEffect(() => {
    getStructures()
      .then((r) => {
        setStructures(r.structures);
        if (r.structures.length > 0) setChosen(r.structures[0].pdb_id);
        if (r.structures.length === 0) setNote("no prepared receptors found");
      })
      .catch((e: Error) => setNote(e.message));
  }, []);

  useEffect(() => {
    if (!chosen || !host.current) return;
    const entry = structures.find((s) => s.pdb_id === chosen);
    if (!entry) return;
    let viewer: { clear: () => void } | null = null;
    let cancelled = false;

    (async () => {
      const [receptor, ligand] = await Promise.all([
        fetch(structureUrl(entry.receptor)).then((r) => r.text()),
        fetch(structureUrl(entry.ligand)).then((r) => r.text()),
      ]);
      if (cancelled || !host.current) return;
      host.current.innerHTML = "";
      const v = createViewer(host.current, { backgroundColor: "white" });
      viewer = v;
      v.addModel(receptor, "pdb");
      v.setStyle({}, { cartoon: { color: "spectrum" } });
      v.addModel(ligand, "sdf");
      v.setStyle({ model: 1 }, { stick: { radius: 0.18, colorscheme: "greenCarbon" } });
      v.zoomTo({ model: 1 });
      v.render();
    })().catch((e: Error) => setNote(e.message));

    return () => {
      cancelled = true;
      viewer?.clear();
    };
  }, [chosen, structures]);

  return (
    <>
      <h3>Binding site (reference co-crystal)</h3>
      {structures.length > 0 && (
        <select value={chosen} onChange={(e) => setChosen(e.target.value)}>
          {structures.map((s) => (
            <option key={s.pdb_id} value={s.pdb_id}>
              {s.pdb_id}
            </option>
          ))}
        </select>
      )}
      <div ref={host} className="viewer3d" />
      {note && <p className="muted small">{note}</p>}
      <p className="muted small">
        Receptor plus its own crystal ligand. The queried molecule is not docked
        here: docking runs offline and takes tens of seconds per molecule.
      </p>
    </>
  );
}
