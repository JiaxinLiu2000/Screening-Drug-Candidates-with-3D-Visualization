"""Task 5 - EGFR structures 1M17 and 4HJO: report, clean, pocket, viewer.

Writes receptors/<ID>_receptor.pdb, receptors/<ID>_ligand.sdf,
receptors/<ID>_pocket.json, results/egfr_structures.json and a
self-contained figures/viewer.html (3Dmol.js from cdnjs, pinned).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import paths, structure  # noqa: E402

PDB_IDS = ("1M17", "4HJO")
THREEDMOL_VERSION = "2.0.4"
THREEDMOL_URL = (
    f"https://cdnjs.cloudflare.com/ajax/libs/3Dmol/{THREEDMOL_VERSION}/3Dmol-min.js"
)


def process(pdb_id: str) -> dict:
    raw = structure.fetch_pdb(pdb_id, paths.DATA_RAW)
    st = structure.read_structure(raw)

    chains = structure.protein_chains(st)
    het_counts = structure.het_atom_counts(st)
    groups = structure.het_groups(st)
    ligand = structure.pick_ligand(st)
    chain_name = structure.main_protein_chain(st, ligand)

    print(f"\n=== {pdb_id} ===")
    print(f"resolution           : {st.resolution} A")
    print(f"all chains in model 1: {[c.name for c in st[0]]}")
    print(f"protein chains       : {[(c, n) for c, n in chains]} (name, n_residues)")
    print("HETATM residue codes with atom counts:")
    for code, n in het_counts.items():
        n_inst = sum(1 for g in groups if g.resname == code)
        tag = "  <-- drug-like ligand" if code == ligand.resname else (
            "  (excluded: solvent/buffer/ion)" if code in structure.NON_LIGAND_HET else ""
        )
        print(f"   {code:<5} atoms={n:<5} instances={n_inst}{tag}")
    print(f"drug-like ligand     : {ligand.resname} "
          f"(chain {ligand.chain}, resi {ligand.seqid}, {ligand.n_atoms} atoms)")
    print(f"kept protein chain   : {chain_name}")

    receptor_path, n_res = structure.write_clean_receptor(
        st, chain_name, paths.RECEPTORS / f"{pdb_id}_receptor.pdb"
    )
    ligand_info = structure.write_ligand_sdf(
        raw,
        ligand,
        paths.RECEPTORS / f"{pdb_id}_ligand.sdf",
        ccd_cache=paths.DATA_RAW / "ccd",
    )
    ligand_path = ligand_info["path"]
    pocket = structure.build_pocket(pdb_id, st, ligand, chain_name)
    pocket_path = structure.write_pocket_json(
        pocket, paths.RECEPTORS / f"{pdb_id}_pocket.json"
    )

    header = structure.header_info(raw)
    missing = structure.missing_residues(raw, chain_name)
    pocket_seqids = {r["seqid"] for r in pocket.contact_residues}
    quality = structure.pocket_quality(st, chain_name, ligand, pocket_seqids, missing)
    conformation = structure.alpha_c_salt_bridge(st, chain_name)

    print(f"cleaned receptor     : {receptor_path.name} ({n_res} residues, protein only)")
    print(f"ligand SDF           : {ligand_path.name} -> {ligand_info['smiles']}")
    print(f"ligand InChIKey      : {ligand_info['inchikey']} "
          f"(bond orders from {ligand_info['bond_orders_from']}, "
          f"{ligand_info['n_aromatic_bonds']} aromatic bonds)")
    print(f"pocket center        : {pocket.center}")
    print(f"pocket box (min/max) : {pocket.box_min} / {pocket.box_max} "
          f"(ligand bbox + 5 A per side, size {pocket.box_size})")
    print(f"residues within 5 A  : {len(pocket.contact_residues)} -> "
          + ", ".join(f"{r['resname']}{r['seqid']}" for r in pocket.contact_residues))
    print(f"deposited TITLE      : {header['title']}")
    print(f"primary citation     : {header['citation_title']} [{header['citation_ref']}]")
    print(f"conformation         : Lys{conformation['lys_seqid']}-Glu{conformation['glu_seqid']} "
          f"= {conformation['distance_angstrom']} A -> {conformation['state']}")
    print(f"pocket quality       : unobserved residues in pocket "
          f"{quality['missing_in_pocket'] or 'none'}; "
          f"alternate-conformer atoms in pocket {quality['altloc_atoms_in_pocket']}; "
          f"mean B pocket {quality['mean_b_pocket']}, ligand {quality['mean_b_ligand']}")
    print(f"internal chain gaps  : {quality['n_internal_gap_residues']} residues "
          f"{quality['internal_gap_residues'] or ''}")
    print(f"wrote                : {pocket_path.name}")

    return {
        "pdb_id": pdb_id,
        "resolution": st.resolution,
        "chains_in_model": [c.name for c in st[0]],
        "protein_chains": [{"chain": c, "n_residues": n} for c, n in chains],
        "het_atom_counts": het_counts,
        "het_instances": [
            {"resname": g.resname, "chain": g.chain, "seqid": g.seqid, "n_atoms": g.n_atoms}
            for g in groups
            if g.resname != "HOH"
        ],
        "n_water_molecules": sum(1 for g in groups if g.resname == "HOH"),
        "ligand": {
            "resname": ligand.resname,
            "chain": ligand.chain,
            "seqid": ligand.seqid,
            "n_atoms": ligand.n_atoms,
            "smiles": ligand_info["smiles"],
            "inchikey": ligand_info["inchikey"],
            "n_heavy_atoms": ligand_info["n_heavy_atoms"],
            "bond_orders_from": ligand_info["bond_orders_from"],
        },
        "kept_chain": chain_name,
        "kept_chain_n_residues": n_res,
        "pocket_center": pocket.center,
        "pocket_box_min": pocket.box_min,
        "pocket_box_max": pocket.box_max,
        "n_contact_residues": len(pocket.contact_residues),
        "contact_residues": [f"{r['resname']}{r['seqid']}" for r in pocket.contact_residues],
        "header": header,
        "conformation": conformation,
        "pocket_quality": quality,
        "missing_residues": missing,
        "files": {
            "receptor": f"receptors/{receptor_path.name}",
            "ligand_sdf": f"receptors/{ligand_path.name}",
            "pocket_json": f"receptors/{pocket_path.name}",
        },
    }


def recommend(reports: list[dict]) -> dict:
    """Pick the receptor for the docking workflow from measured structural facts.

    Resolution alone cannot settle 2.60 vs 2.75 A, and the pocket-local quality
    metrics favour the runner-up, so neither decides it. The ranking is:

      1. no unobserved (REMARK 465) residue inside the 5 A pocket - hard gate,
         a gap in the binding site cannot be docked into (both entries pass);
      2. conformational state matching the screening objective - we screen
         ATP-competitive ligands, so the active alphaC-in state is preferred.
         This is what actually decides the choice;
      3. crystallographic resolution, as a minor tiebreak only.

    Pocket-local disorder (alternate conformers, B-factors) and construct length
    are reported as caveats, not used to rank.
    """
    def key(r: dict) -> tuple:
        return (
            len(r["pocket_quality"]["missing_in_pocket"]),
            0 if r["conformation"]["state"].startswith("active") else 1,
            r["resolution"],
        )

    ranked = sorted(reports, key=key)
    best, other = ranked[0], ranked[1]
    bq, oq = best["pocket_quality"], other["pocket_quality"]
    bspan, ospan = bq["modelled_residue_span"], oq["modelled_residue_span"]

    reason = (
        f"{best['pdb_id']} and {other['pdb_id']} both carry the same ATP-site ligand "
        f"({best['ligand']['resname']}, erlotinib, InChIKey "
        f"{best['ligand']['inchikey']}) in a single clean chain, and neither has an "
        f"unobserved residue inside the 5 A pocket, so both pass the hard gate and the "
        f"pocket definitions are directly comparable. The choice is decided by "
        f"conformational state. {other['pdb_id']}'s deposition says so in its title "
        f"(\"{other['header']['title']}\"); {best['pdb_id']}'s title does not state a "
        f"state, so both were established from the coordinates via the "
        f"beta3-Lys{best['conformation']['lys_seqid']}/alphaC-"
        f"Glu{best['conformation']['glu_seqid']} salt bridge: "
        f"{best['conformation']['distance_angstrom']} A in {best['pdb_id']} "
        f"({best['conformation']['state']}) versus "
        f"{other['conformation']['distance_angstrom']} A in {other['pdb_id']} "
        f"({other['conformation']['state']}); erlotinib binding both states is the "
        f"subject of {other['header']['citation_ref']}. We screen ATP-competitive "
        f"chemotypes against the catalytically relevant active site, so the active "
        f"alphaC-in state is the right target, with resolution "
        f"({best['resolution']} vs {other['resolution']} A) a minor tiebreak in the "
        f"same direction. "
        f"Two caveats to state in the proposal. First, {other['pdb_id']}'s pocket is "
        f"the better-ordered one: mean B {oq['mean_b_pocket']} vs "
        f"{bq['mean_b_pocket']} over pocket atoms, ligand B {oq['mean_b_ligand']} vs "
        f"{bq['mean_b_ligand']}, and {oq['altloc_atoms_in_pocket']} vs "
        f"{bq['altloc_atoms_in_pocket']} alternate-conformer atoms in the site. "
        f"Second, {best['pdb_id']} models more residues "
        f"({best['kept_chain_n_residues']} vs {other['kept_chain_n_residues']}) only "
        f"because its construct is longer (span {bspan[0]}-{bspan[1]} vs "
        f"{ospan[0]}-{ospan[1]}); it actually has more residues in internal chain gaps "
        f"({bq['n_internal_gap_residues']} vs {oq['n_internal_gap_residues']}), none of "
        f"them in the pocket. {other['pdb_id']} is therefore kept as the inactive-state "
        f"counter-structure for a cross-docking sanity check rather than discarded."
    )
    return {
        "recommended": best["pdb_id"],
        "runner_up": other["pdb_id"],
        "reason": reason,
        "criteria": [
            "no unobserved residue inside the 5 A pocket (hard gate, both pass)",
            "active alphaC-in conformation for ATP-competitive screening (decisive)",
            "resolution (minor tiebreak)",
        ],
        "evidence": {
            r["pdb_id"]: {
                "deposited_title": r["header"]["title"],
                "citation": f"{r['header']['citation_title']} [{r['header']['citation_ref']}]",
                "salt_bridge_angstrom": r["conformation"]["distance_angstrom"],
                "state": r["conformation"]["state"],
                "resolution": r["resolution"],
                "missing_in_pocket": r["pocket_quality"]["missing_in_pocket"],
                "altloc_atoms_in_pocket": r["pocket_quality"]["altloc_atoms_in_pocket"],
                "mean_b_pocket": r["pocket_quality"]["mean_b_pocket"],
                "mean_b_ligand": r["pocket_quality"]["mean_b_ligand"],
            }
            for r in reports
        },
    }


def build_viewer(reports: list[dict], recommendation: dict) -> Path:
    """Self-contained 3Dmol.js page; all coordinates inlined (works over file://)."""
    payload = {}
    for rep in reports:
        pdb_id = rep["pdb_id"]
        payload[pdb_id] = {
            "receptor": (paths.RECEPTORS / f"{pdb_id}_receptor.pdb").read_text(encoding="utf-8"),
            "ligand": (paths.RECEPTORS / f"{pdb_id}_ligand.sdf").read_text(encoding="utf-8"),
            "pocket": json.loads(
                (paths.RECEPTORS / f"{pdb_id}_pocket.json").read_text(encoding="utf-8")
            ),
            "meta": {
                "resolution": rep["resolution"],
                "chain": rep["kept_chain"],
                "n_residues": rep["kept_chain_n_residues"],
                "ligand_resname": rep["ligand"]["resname"],
            },
        }

    data_js = json.dumps(payload).replace("</", "<\\/")
    default_id = recommendation["recommended"]
    buttons_html = []
    for r in reports:
        pid = r["pdb_id"]
        is_default = pid == default_id
        cls = ' class="active"' if is_default else ""
        label = pid + (" (recommended)" if is_default else "")
        buttons_html.append(
            f'      <button type="button" data-pdb="{pid}"{cls}>{label}</button>'
        )
    buttons = "\n".join(buttons_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EGFR kinase domain - cleaned receptor, erlotinib, 5 A pocket</title>
<script src="{THREEDMOL_URL}" crossorigin="anonymous"></script>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 16px; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  #viewer {{ position: relative; width: 960px; height: 640px;
             border: 1px solid #8884; border-radius: 6px; margin-top: 10px; }}
  .bar {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
  button {{ font: inherit; padding: 4px 10px; border-radius: 4px;
            border: 1px solid #8886; background: transparent; cursor: pointer; }}
  button.active {{ background: #2d7ff933; border-color: #2d7ff9; font-weight: 600; }}
  #info {{ margin-top: 8px; font-size: 13px; opacity: .85; max-width: 960px; }}
  code {{ font-size: 12px; }}
</style>
</head>
<body>
<h1>EGFR kinase domain &mdash; cartoon receptor, erlotinib sticks, 5 &Aring; pocket residues</h1>
<div class="bar">
{buttons}
  <span style="margin-left:12px">
    <label><input type="radio" name="view" value="whole" checked> whole domain</label>
    <label><input type="radio" name="view" value="pocket"> zoom pocket</label>
  </span>
  <span style="margin-left:12px">
    <label><input type="checkbox" id="show-labels"> residue labels</label>
    <label><input type="checkbox" id="show-box"> pocket box</label>
    <label><input type="checkbox" id="show-surface"> pocket surface</label>
  </span>
</div>
<div id="viewer"></div>
<div id="info"></div>
<script>
const DATA = {data_js};
const viewer = $3Dmol.createViewer(document.getElementById("viewer"),
                                   {{ backgroundColor: "white" }});
let current = "{default_id}";

function render(id) {{
  current = id;
  const d = DATA[id];
  viewer.clear();
  viewer.addModel(d.receptor, "pdb");
  viewer.setStyle({{ model: 0 }}, {{ cartoon: {{ color: "spectrum", opacity: 0.9 }} }});

  const resi = d.pocket.contact_residues.map(r => r.seqid);
  viewer.addStyle({{ model: 0, resi: resi }},
                  {{ stick: {{ radius: 0.16, colorscheme: "yellowCarbon" }} }});
  if (document.getElementById("show-labels").checked) {{
    viewer.addResLabels({{ model: 0, resi: resi, atom: "CA" }},
                        {{ font: "sans-serif", fontSize: 10, showBackground: true,
                           backgroundOpacity: 0.55, fontColor: "black" }});
  }}

  viewer.addModel(d.ligand, "sdf");
  viewer.setStyle({{ model: 1 }},
                  {{ stick: {{ radius: 0.22, colorscheme: "cyanCarbon" }} }});

  if (document.getElementById("show-surface").checked) {{
    viewer.addSurface($3Dmol.SurfaceType.VDW, {{ opacity: 0.65, color: "lightgrey" }},
                      {{ model: 0, resi: resi }});
  }}
  if (document.getElementById("show-box").checked) {{
    const lo = d.pocket.box_min, hi = d.pocket.box_max;
    viewer.addBox({{ center: {{ x: (lo[0]+hi[0])/2, y: (lo[1]+hi[1])/2, z: (lo[2]+hi[2])/2 }},
                    dimensions: {{ w: hi[0]-lo[0], h: hi[1]-lo[1], d: hi[2]-lo[2] }},
                    color: "magenta", opacity: 0.18, wireframe: false }});
  }}

  const view = document.querySelector('input[name="view"]:checked').value;
  if (view === "pocket") {{
    viewer.zoomTo({{ model: 1 }});
    viewer.zoom(0.5);
  }} else {{
    viewer.zoomTo();
  }}
  viewer.render();

  const p = d.pocket, m = d.meta;
  document.getElementById("info").innerHTML =
    `<b>${{id}}</b> &middot; ${{m.resolution}} &Aring; &middot; chain ${{m.chain}} `
    + `(${{m.n_residues}} residues, waters/buffers/other chains stripped) &middot; `
    + `ligand <b>${{m.ligand_resname}}</b> &middot; `
    + `pocket center <code>[${{p.center.join(", ")}}]</code> &middot; `
    + `box <code>[${{p.box_min.join(", ")}}]</code> &rarr; `
    + `<code>[${{p.box_max.join(", ")}}]</code> (ligand bbox + ${{p.padding_angstrom}} &Aring;) `
    + `&middot; ${{p.contact_residues.length}} residues within `
    + `${{p.contact_cutoff_angstrom}} &Aring; (yellow sticks).`;
}}

document.querySelectorAll("button[data-pdb]").forEach(b => {{
  b.addEventListener("click", () => {{
    document.querySelectorAll("button[data-pdb]").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    render(b.dataset.pdb);
  }});
}});
document.querySelectorAll('#show-box, #show-surface, #show-labels, input[name="view"]')
  .forEach(el => el.addEventListener("change", () => render(current)));

render(current);
</script>
</body>
</html>
"""
    out = paths.FIGURES / "viewer.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    paths.ensure_dirs()
    reports = [process(pdb_id) for pdb_id in PDB_IDS]
    recommendation = recommend(reports)

    out = paths.RESULTS / "egfr_structures.json"
    out.write_text(
        json.dumps({"structures": reports, "recommendation": recommendation}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    viewer = build_viewer(reports, recommendation)

    print(f"\nRecommended PDB: {recommendation['recommended']}")
    print(f"Reason: {recommendation['reason']}")
    print(f"\nWrote {out}")
    print(f"Wrote {viewer}  ({viewer.stat().st_size / 1024:.0f} KB, self-contained)")
    print("Open it with:  start figures\\viewer.html      (Windows cmd)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
