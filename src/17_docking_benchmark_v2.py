"""Phase 3 Task 2 - EGFR self-docking benchmark, stratified by alphaC conformation.

Every entry is self-contained for self-docking: its own cognate receptor, its own
pocket box from its own cognate ligand, and its own crystallographic ligand pose.

Pipeline: RCSB query (UniProt P00533, X-ray, resolution <= 2.8 A; the RCSB Search
API has no conformational-state attribute, so state can only be measured from
coordinates) -> the Phase 2 per-entry quality filters (reused from
src/13_docking_benchmark.py) -> the Lys/Glu salt bridge measured on EVERY passing
entry -> one entry per distinct ligand InChIKey -> final set balanced toward
equal active/inactive counts. Writes receptors/benchmark_v2/ and
results/docking_benchmark_v2.{md,json}. No docking is run here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util
import json
import shutil
import tempfile

import numpy as np
from rdkit import Chem

from tox21lib import featurize, paths, rcsb, structure

# Reuse (not reimplement) the Phase 2 per-entry quality filters.
_spec = importlib.util.spec_from_file_location(
    "docking_benchmark_v1", Path(__file__).resolve().parent / "13_docking_benchmark.py")
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)

ACCESSION = _v1.ACCESSION
RESOLUTION_BOUND = 2.8  # no relaxation: same bound as Phase 2
MIN_HEAVY, MAX_HEAVY = _v1.MIN_HEAVY, _v1.MAX_HEAVY
POCKET_CUTOFF = _v1.POCKET_CUTOFF
BOX_PADDING = 5.0
MAX_SET = 20
SIM_THRESHOLD = 0.7
SALT_BRIDGE_CUTOFF = 4.0  # encoded in structure.alpha_c_salt_bridge
OUT_DIR = paths.RECEPTORS / "benchmark_v2"
CCD_CACHE = paths.DATA_RAW / "ccd"
QUERY_CACHE = paths.DATA_RAW / f"rcsb_{ACCESSION}_xray_le{RESOLUTION_BOUND}.json"


def state_of(conf: dict) -> str:
    return conf["state"].split(" ")[0]


def screen(ids: list[str], tmp: Path) -> tuple[list[dict], dict[str, str]]:
    """Quality filters + salt-bridge measurement for every candidate."""
    log: dict[str, str] = {}
    passing = []
    for pdb_id in ids:
        r = _v1.evaluate(pdb_id, tmp)
        if not r["ok"]:
            log[pdb_id] = r["reason"]
            continue
        if r["resolution"] > RESOLUTION_BOUND:
            log[pdb_id] = f"resolution {r['resolution']} A in file > bound {RESOLUTION_BOUND} A"
            continue
        r["conformation"] = structure.alpha_c_salt_bridge_any_numbering(r["st"], r["chain"])
        r["state"] = state_of(r["conformation"])
        passing.append(r)
    return passing, log


def pick(passing: list[dict], log: dict[str, str]) -> tuple[list[dict], dict]:
    """Distinct chemistry, then balance active/inactive within MAX_SET."""
    for r in passing:
        if r["state"] == "undetermined":
            log[r["pdb_id"]] = ("passed quality filters but conformational state undetermined: "
                                f"{r['conformation']['reason']}")
    determined = [r for r in passing if r["state"] != "undetermined"]

    # One entry per InChIKey: best resolution, then PDB ID (Phase 2 rule, state-blind).
    groups: dict[str, list[dict]] = {}
    for r in determined:
        groups.setdefault(r["inchikey"], []).append(r)
    reps = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda r: (r["resolution"], r["pdb_id"]))
        rep = members[0]
        reps.append(rep)
        for r in members[1:]:
            log[r["pdb_id"]] = (f"duplicate ligand chemistry: {r['ligand'].resname} "
                                f"({r['inchikey']}, {r['state']}) already represented by "
                                f"{rep['pdb_id']} ({rep['resolution']} A, {rep['state']})")

    rank = lambda r: (r["resolution"], r["pdb_id"])  # noqa: E731
    active = sorted((r for r in reps if r["state"] == "active"), key=rank)
    inactive = sorted((r for r in reps if r["state"] == "inactive"), key=rank)
    n_active = min(len(active), MAX_SET // 2)
    n_inactive = min(len(inactive), MAX_SET - n_active)
    if n_inactive < MAX_SET // 2:  # not enough inactives: give the slack back to actives
        n_active = min(len(active), MAX_SET - n_inactive)
    for stratum, chosen_n in ((active, n_active), (inactive, n_inactive)):
        for r in stratum[chosen_n:]:
            log[r["pdb_id"]] = (f"passed all filters ({r['state']}) but outside the "
                                f"{MAX_SET}-structure cap after active/inactive balancing "
                                "(ranked by resolution, then PDB ID, within state)")
    chosen = sorted(active[:n_active] + inactive[:n_inactive], key=lambda r: r["pdb_id"])
    population = {
        "entries_passing_quality_filters": len(passing),
        "entries_by_state": {s: sum(r["state"] == s for r in passing)
                             for s in ("active", "inactive", "undetermined")},
        "distinct_ligands_by_state": {"active": len(active), "inactive": len(inactive)},
        "active_entries": sorted(r["pdb_id"] for r in passing if r["state"] == "active"),
        "distinct_active_ligand_entries": [r["pdb_id"] for r in active],
    }
    return chosen, population


def write_entry(r: dict) -> dict:
    pdb_id, st, ligand, chain = r["pdb_id"], r["st"], r["ligand"], r["chain"]
    _, n_res = structure.write_clean_receptor(st, chain, OUT_DIR / f"{pdb_id}_receptor.pdb")
    info = structure.write_ligand_sdf(r["raw"], ligand, OUT_DIR / f"{pdb_id}_ligand.sdf",
                                      ccd_cache=CCD_CACHE)
    pocket = structure.build_pocket(pdb_id, st, ligand, chain, padding=BOX_PADDING,
                                    contact_cutoff=POCKET_CUTOFF)
    quality = structure.pocket_quality(st, chain, ligand, {c["seqid"] for c in r["contacts"]},
                                       r["missing"])
    rec = {
        "pdb_id": pdb_id,
        "use": "self-docking: dock this ligand into this receptor, score RMSD vs this SDF",
        "resolution_angstrom": r["resolution"],
        "ligand_code": ligand.resname,
        "ligand_instance": ligand.label,
        "ligand_smiles": info["smiles"],
        "ligand_inchikey": info["inchikey"],
        "ligand_heavy_atoms": info["n_heavy_atoms"],
        "conformation": r["conformation"],
        "state": r["state"],
        "pocket_source": f"cognate ligand {ligand.label} of {pdb_id}",
        "pocket_center": pocket.center,
        "pocket_box_min": pocket.box_min,
        "pocket_box_max": pocket.box_max,
        "pocket_box_size": pocket.box_size,
        "pocket_box_padding_angstrom": pocket.padding_angstrom,
        "mean_pocket_b_factor": quality["mean_b_pocket"],
        "mean_ligand_b_factor": quality["mean_b_ligand"],
        "altloc_atoms_in_pocket": quality["altloc_atoms_in_pocket"],
        "n_residues_within_5A": len(pocket.contact_residues),
        "contact_residues": [f"{c['resname']}{c['seqid']}" for c in pocket.contact_residues],
        "kept_chain": chain,
        "kept_chain_residues": n_res,
        "title": structure.header_info(r["raw"])["title"],
        "files": {"receptor": f"receptors/benchmark_v2/{pdb_id}_receptor.pdb",
                  "ligand": f"receptors/benchmark_v2/{pdb_id}_ligand.sdf",
                  "record": f"receptors/benchmark_v2/{pdb_id}.json"},
    }
    (OUT_DIR / f"{pdb_id}.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")

    mol = Chem.MolFromMolFile(str(OUT_DIR / f"{pdb_id}_ligand.sdf"))
    assert mol is not None, f"{pdb_id}: SDF does not parse"
    assert mol.GetNumHeavyAtoms() == r["n_heavy"] == info["n_heavy_atoms"], f"{pdb_id}: heavy"
    assert Chem.MolToInchiKey(mol) == info["inchikey"] == r["inchikey"], f"{pdb_id}: InChIKey"
    rec["sdf_check"] = {"parses": True, "heavy_atoms_match": True, "inchikey_match": True}
    return rec


def similarity(recs: list[dict]) -> dict:
    fps = featurize.featurize_smiles([r["ligand_smiles"] for r in recs]).astype(bool)
    pairs = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            inter = np.logical_and(fps[i], fps[j]).sum()
            union = np.logical_or(fps[i], fps[j]).sum()
            pairs.append((float(inter / union), recs[i]["pdb_id"], recs[j]["pdb_id"]))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    sims = np.array([p[0] for p in pairs])
    return {
        "fingerprint": f"Morgan r={featurize.MORGAN_RADIUS}, {featurize.MORGAN_N_BITS} bits",
        "n_pairs": len(pairs),
        "min": round(float(sims.min()), 3),
        "median": round(float(np.median(sims)), 3),
        "mean": round(float(sims.mean()), 3),
        "max": round(float(sims.max()), 3),
        "most_similar_pair": {"pdb_ids": [pairs[0][1], pairs[0][2]],
                              "tanimoto": round(pairs[0][0], 3)},
        "threshold": SIM_THRESHOLD,
        "n_pairs_above_threshold": int((sims > SIM_THRESHOLD).sum()),
        "pairs_above_threshold": [{"pdb_ids": [a, b], "tanimoto": round(s, 3)}
                                  for s, a, b in pairs if s > SIM_THRESHOLD],
    }


def reason_category(reason: str) -> str:
    for prefix, label in (
        ("PDB-format", "PDB-format file unavailable"),
        ("0 drug-like", "no drug-like ligand instance (15-60 heavy atoms)"),
        ("ligand ", "ligand not in contact with kept chain"),
        ("unobserved", "unobserved residue in/adjacent to pocket"),
        ("CCD", "CCD template does not map"),
        ("SDF", "SDF heavy-atom mismatch"),
        ("resolution", "resolution above bound"),
        ("duplicate", "duplicate ligand chemistry (InChIKey)"),
        ("passed quality", "conformational state undetermined"),
        ("passed all", f"beyond {MAX_SET}-structure cap after balancing"),
    ):
        if reason.startswith(prefix):
            return label
    if "drug-like" in reason:
        return "more than one drug-like ligand instance"
    raise ValueError(f"uncategorised rejection reason: {reason}")


def write_markdown(s: dict, recs: list[dict], out: Path) -> None:
    sim, pop, cc = s["similarity"], s["queried_population"], s["conformation_counts"]
    lines = [
        "# EGFR self-docking benchmark (v2)",
        "",
        "Generated by `src/17_docking_benchmark_v2.py`. No docking is run here.",
        "",
        "## Protocol note - two different uses of docking",
        "",
        "1. **Engine pose accuracy (self-docking; this set).** Each ligand is docked back into "
        "its *own* cognate receptor (`receptors/benchmark_v2/<PDB>_receptor.pdb`), inside a box "
        "built from its *own* crystallographic ligand (`pocket_center` / `pocket_box_*` in "
        "`<PDB>.json`, ligand bbox + 5 A per side), and the predicted pose is scored by heavy-atom "
        "RMSD against its *own* crystal pose (`<PDB>_ligand.sdf`). This is the Vina vs DiffDock "
        "comparison. Because receptor and ligand come from the same crystal, the receptor is "
        "already in the conformation that ligand selected, so RMSD measures the engine, not "
        "conformational tolerance. Results are reported overall and per alphaC state.",
        "2. **Screening (1M17).** Novel molecules are docked into the single screening receptor "
        "1M17 (active, erlotinib-bound; `receptors/` Phase 1 artefacts from "
        "`src/05_egfr_structure.py`). There is no crystal pose, so no RMSD; this use produces "
        "scores/rankings only. Neither the benchmark_v2 receptors nor their boxes are used for "
        "screening, and 1M17 is not used for pose-accuracy evaluation.",
        "",
        "Phase 2's set (`receptors/benchmark/`, 17/20 inactive) remains on disk unchanged; "
        "docking its ligands into 1M17 would have conflated engine error with "
        "active-vs-inactive receptor mismatch, which this set avoids.",
        "",
        "## Selection",
        "",
        f"- RCSB query: UniProt {ACCESSION}, X-ray, `resolution_combined <= {RESOLUTION_BOUND}` "
        f"(cached payload + IDs in `{QUERY_CACHE.relative_to(paths.ROOT).as_posix()}`): "
        f"**{s['n_queried']}** entries. RCSB exposes no conformational-state attribute, so every "
        "entry was screened and the salt bridge measured from coordinates on each one that "
        "passed the quality filters.",
        "- Quality filters (identical to Phase 2): exactly one non-polymer ligand with "
        f"{MIN_HEAVY}-{MAX_HEAVY} heavy atoms; ligand contacts the kept chain; no REMARK 465 "
        "residue in or sequence-adjacent to the 5 A pocket; CCD template maps onto the crystal "
        "copy; then one entry per ligand InChIKey.",
        f"- Filter relaxation: **{s['relaxation']}**",
        "",
        "### Size of the active-state population",
        "",
        f"- Entries passing all quality filters: **{pop['entries_passing_quality_filters']}** "
        f"(active {pop['entries_by_state']['active']}, inactive "
        f"{pop['entries_by_state']['inactive']}, undetermined "
        f"{pop['entries_by_state']['undetermined']}).",
        f"- Distinct ligand chemistries: active **{pop['distinct_ligands_by_state']['active']}**, "
        f"inactive **{pop['distinct_ligands_by_state']['inactive']}**.",
        f"- Active entries: {', '.join(pop['active_entries']) or 'none'}.",
        "- One entry per InChIKey (best resolution, then PDB ID; state-blind), then the "
        f"{MAX_SET // 2} best-resolution entries per state (resolution, then PDB ID); if a "
        "state had fewer, its slack would go to the other state.",
        "",
        f"**Final set: {s['n_selected']} structures - active {cc['active']}, inactive "
        f"{cc['inactive']}.** {s['balance_statement']}",
        "",
        "Per-state site quality (confound for any per-state RMSD comparison): "
        + "; ".join(
            f"{st} median resolution "
            f"{np.median([r['resolution_angstrom'] for r in recs if r['state'] == st]):.2f} A, "
            f"median pocket B "
            f"{np.median([r['mean_pocket_b_factor'] for r in recs if r['state'] == st]):.1f}"
            for st in ("active", "inactive")) + ".",
        "",
        "Rejections by reason (every one of the queried entries is either selected or logged "
        "in `results/docking_benchmark_v2.json` -> `rejection_log`):",
        "",
        "| Reason | Entries |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in s["rejection_reason_counts"].items()]
    lines += [
        "",
        "## Structures",
        "",
        "Salt bridge: min beta3-Lys NZ to alphaC-Glu OE1/OE2 distance, <= 4.0 A = active; "
        "numbering chosen per entry by the Lys/Glu/hinge-Met triple "
        "(mature 721/738/769 or precursor 745/762/793).",
        "",
        "| PDB | Res. (A) | Ligand | Heavy | SMILES | State | Lys-Glu (A) | Numbering | "
        "Mean pocket B | Altloc atoms in site | Residues <= 5 A |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        c = r["conformation"]
        lines.append(
            f"| {r['pdb_id']} | {r['resolution_angstrom']:.2f} | {r['ligand_code']} | "
            f"{r['ligand_heavy_atoms']} | `{r['ligand_smiles']}` | {r['state']} | "
            f"{c['distance_angstrom']:.2f} | {c['numbering']} (K{c['lys_seqid']}/E{c['glu_seqid']}) | "
            f"{r['mean_pocket_b_factor']} | {r['altloc_atoms_in_pocket']} | "
            f"{r['n_residues_within_5A']} |")
    lines += [
        "",
        "## Ligand diversity (final set)",
        "",
        f"Pairwise Tanimoto, {sim['fingerprint']} (`tox21lib.featurize`), {sim['n_pairs']} pairs:",
        "",
        "| Statistic | Value |",
        "|---|---|",
        f"| min | {sim['min']:.3f} |",
        f"| median | {sim['median']:.3f} |",
        f"| mean | {sim['mean']:.3f} |",
        f"| max | {sim['max']:.3f} |",
        f"| pairs > {sim['threshold']} | {sim['n_pairs_above_threshold']} |",
        f"| most similar pair | {' / '.join(sim['most_similar_pair']['pdb_ids'])} "
        f"({sim['most_similar_pair']['tanimoto']:.3f}) |",
        "",
        "All InChIKeys are distinct. The Morgan fingerprint is chirality-blind, so a pair at "
        "Tanimoto 1.000 with distinct InChIKeys is a pair of stereoisomers (pairs at 1.000: "
        + (", ".join(" / ".join(p["pdb_ids"]) for p in sim["pairs_above_threshold"]
                     if p["tanimoto"] == 1.0) or "none") + ").",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths.ensure_dirs()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    query = rcsb.uniprot_xray_query(ACCESSION, RESOLUTION_BOUND)
    ids = rcsb.cached_entry_ids(query, QUERY_CACHE)
    with tempfile.TemporaryDirectory() as tmpdir:
        passing, log = screen(ids, Path(tmpdir))
    chosen, population = pick(passing, log)
    recs = [write_entry(r) for r in chosen]
    log = dict(sorted(log.items()))

    cc = {"active": sum(r["state"] == "active" for r in recs),
          "inactive": sum(r["state"] == "inactive" for r in recs)}
    n_act_pop = population["distinct_ligands_by_state"]["active"]
    assert len({r["ligand_inchikey"] for r in recs}) == len(recs)
    n_pop = population["distinct_ligands_by_state"]
    if cc["active"] == cc["inactive"]:
        balance = (f"The set is balanced. The active-state population is not small: "
                   f"{n_act_pop} distinct active-state ligand chemistries (and "
                   f"{n_pop['inactive']} inactive) pass the quality filters, so stratification "
                   f"is supported at {cc['active']} per state; the {MAX_SET}-structure cap, not "
                   "availability, limits each stratum.")
    else:
        short = min(("active", "inactive"), key=lambda k: n_pop[k])
        balance = (f"The set is not balanced: only {n_pop[short]} distinct {short}-state ligand "
                   "chemistries pass the quality filters in the queried population, so all of "
                   f"them are included. LIMITATION: per-state pose-accuracy estimates rest on "
                   f"{cc['active']} active vs {cc['inactive']} inactive structures; treat the "
                   f"{short}-stratum result as descriptive, not as a powered comparison.")
    reason_counts: dict[str, int] = {}
    for reason in log.values():
        k = reason_category(reason)
        reason_counts[k] = reason_counts.get(k, 0) + 1

    summary = {
        "accession": ACCESSION,
        "resolution_bound": RESOLUTION_BOUND,
        "relaxation": "no filter relaxed (same bounds as Phase 2)",
        "query_cache": QUERY_CACHE.relative_to(paths.ROOT).as_posix(),
        "n_queried": len(ids),
        "n_selected": len(recs),
        "n_rejected": len(log),
        "queried_population": population,
        "conformation_counts": cc,
        "balance_statement": balance,
        "salt_bridge_cutoff_angstrom": SALT_BRIDGE_CUTOFF,
        "box_padding_angstrom": BOX_PADDING,
        "similarity": similarity(recs),
        "rejection_reason_counts": dict(sorted(reason_counts.items(),
                                               key=lambda kv: (-kv[1], kv[0]))),
        "selected": [r["pdb_id"] for r in recs],
        "structures": recs,
        "rejection_log": log,
        "protocol": {
            "self_docking": "receptors/benchmark_v2/<PDB>_receptor.pdb + box from <PDB>.json; "
                            "RMSD vs <PDB>_ligand.sdf; Vina vs DiffDock pose accuracy",
            "screening": "novel molecules docked into 1M17 (src/05_egfr_structure.py artefacts); "
                         "no RMSD, ranking only",
        },
    }
    assert set(summary["selected"]).isdisjoint(log)
    assert set(summary["selected"]) | set(log) == set(ids) and len(recs) + len(log) == len(ids)
    (paths.RESULTS / "docking_benchmark_v2.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, recs, paths.RESULTS / "docking_benchmark_v2.md")
    print(f"queried {len(ids)}; population {population['entries_by_state']} "
          f"distinct {population['distinct_ligands_by_state']}; selected {len(recs)} {cc}")
    print("similarity", {k: v for k, v in summary["similarity"].items()
                         if k != "pairs_above_threshold"})
    print("reasons", summary["rejection_reason_counts"])


if __name__ == "__main__":
    main()
