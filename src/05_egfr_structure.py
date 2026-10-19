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
    ligand_path = structure.write_ligand_sdf(
        raw, ligand, paths.RECEPTORS / f"{pdb_id}_ligand.sdf"
    )
    pocket = structure.build_pocket(pdb_id, st, ligand, chain_name)
    pocket_path = structure.write_pocket_json(
        pocket, paths.RECEPTORS / f"{pdb_id}_pocket.json"
    )

    print(f"cleaned receptor     : {receptor_path.name} ({n_res} residues, protein only)")
    print(f"ligand SDF           : {ligand_path.name}")
    print(f"pocket center        : {pocket.center}")
    print(f"pocket box (min/max) : {pocket.box_min} / {pocket.box_max} "
          f"(ligand bbox + 5 A per side, size {pocket.box_size})")
    print(f"residues within 5 A  : {len(pocket.contact_residues)} -> "
          + ", ".join(f"{r['resname']}{r['seqid']}" for r in pocket.contact_residues))
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
        },
        "kept_chain": chain_name,
        "kept_chain_n_residues": n_res,
        "pocket_center": pocket.center,
        "pocket_box_min": pocket.box_min,
        "pocket_box_max": pocket.box_max,
        "n_contact_residues": len(pocket.contact_residues),
        "contact_residues": [f"{r['resname']}{r['seqid']}" for r in pocket.contact_residues],
        "files": {
            "receptor": f"receptors/{receptor_path.name}",
            "ligand_sdf": f"receptors/{ligand_path.name}",
            "pocket_json": f"receptors/{pocket_path.name}",
        },
    }


def recommend(reports: list[dict]) -> dict:
    """Pick the structure to build the docking workflow on, from measured values.

    Criteria, in order: crystallographic resolution (lower is better), then the
    number of modelled residues in the kept chain (more complete pocket).
    """
    ranked = sorted(
        reports, key=lambda r: (r["resolution"], -r["kept_chain_n_residues"])
    )
    best, other = ranked[0], ranked[1]
    reason = (
        f"{best['pdb_id']} is the higher-resolution structure "
        f"({best['resolution']} A vs {other['resolution']} A for {other['pdb_id']}) "
        f"and models {best['kept_chain_n_residues']} residues in the kept chain "
        f"(vs {other['kept_chain_n_residues']}); both carry the same ATP-site ligand "
        f"{best['ligand']['resname']} (erlotinib), so the pocket definition is "
        f"directly comparable and the sharper electron density gives more reliable "
        f"side-chain positions for docking."
    )
    return {"recommended": best["pdb_id"], "runner_up": other["pdb_id"], "reason": reason}


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
