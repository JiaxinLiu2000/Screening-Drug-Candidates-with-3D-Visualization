"""Phase 2 Task 5 - EGFR co-crystal reference set for Vina vs DiffDock pose accuracy.

RCSB query (UniProt P00533, X-ray, resolution bound) -> per-entry filters with a
recorded rejection reason -> distinct ligand chemistry by InChIKey (best
resolution wins) -> 15-20 structures written to receptors/benchmark/.
Writes results/docking_benchmark_set.{md,json}. No docking is run here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import shutil
import tempfile

import numpy as np
from rdkit import Chem

from tox21lib import featurize, paths, rcsb, structure

ACCESSION = "P00533"
RESOLUTION_STEPS = (2.8, 2.9, 3.0)  # only the resolution bound may be relaxed
MIN_HEAVY, MAX_HEAVY = 15, 60
POCKET_CUTOFF = 5.0
MIN_SET, MAX_SET = 15, 20
SIM_THRESHOLD = 0.7
OUT_DIR = paths.RECEPTORS / "benchmark"
CCD_CACHE = paths.DATA_RAW / "ccd"


def evaluate(pdb_id: str, tmp: Path) -> dict:
    """Apply the per-entry filters. Returns {'ok': True, ...} or {'ok': False, 'reason'}."""
    try:
        raw = structure.fetch_pdb(pdb_id, paths.DATA_RAW)
    except Exception as exc:  # noqa: BLE001 - large entries have no legacy PDB file
        return {"ok": False, "reason": f"PDB-format file unavailable ({type(exc).__name__}: {exc})"}
    st = structure.read_structure(raw)
    resolution = round(float(st.resolution), 2)

    ligs = structure.nonpolymer_ligands(st)
    qualifying = [g for g in ligs if MIN_HEAVY <= g.n_heavy_atoms <= MAX_HEAVY]
    if len(qualifying) != 1:
        seen = ", ".join(f"{g.label}({g.n_heavy_atoms} heavy)" for g in ligs) or "none"
        return {"ok": False, "reason": f"{len(qualifying)} drug-like ligand instances with "
                f"{MIN_HEAVY}-{MAX_HEAVY} heavy atoms (non-excluded HET groups: {seen})"}
    ligand = qualifying[0]
    n_heavy = ligand.n_heavy_atoms

    chain = structure.main_protein_chain(st, ligand)
    coords = structure.ligand_coords(st, ligand)
    contacts = structure.contact_residues(st, chain, coords, POCKET_CUTOFF)
    if not contacts:
        return {"ok": False, "reason": f"ligand {ligand.label} has no residue of kept "
                f"chain {chain} within {POCKET_CUTOFF} A"}
    missing = structure.missing_residues(raw, chain)
    gap = sorted({r["seqid"] for r in contacts} & set(missing))
    # contact_residues only sees modelled residues, so also test REMARK 465 residues
    # whose sequence neighbours line the pocket (an unobserved residue's atoms have
    # no coordinates; adjacency to a contact residue is the observable proxy).
    contact_ids = {r["seqid"] for r in contacts}
    gap_adjacent = sorted(m for m in missing if (m - 1) in contact_ids or (m + 1) in contact_ids)
    if gap or gap_adjacent:
        return {"ok": False, "reason": f"unobserved (REMARK 465) residue(s) "
                f"{sorted(set(gap) | set(gap_adjacent))} in/adjacent to the 5 A pocket of "
                f"{ligand.label}"}

    try:
        info = structure.write_ligand_sdf(raw, ligand, tmp / f"{pdb_id}.sdf", ccd_cache=CCD_CACHE)
    except Exception as exc:  # noqa: BLE001 - template mapping or CCD download failure
        return {"ok": False, "reason": f"CCD template does not map onto {ligand.label}: {exc}"}
    if info["n_heavy_atoms"] != n_heavy:
        return {"ok": False, "reason": f"SDF heavy atoms {info['n_heavy_atoms']} != crystal "
                f"{n_heavy} for {ligand.label}"}
    return {"ok": True, "pdb_id": pdb_id, "raw": raw, "st": st, "resolution": resolution,
            "ligand": ligand, "chain": chain, "contacts": contacts, "missing": missing,
            "smiles": info["smiles"], "inchikey": info["inchikey"], "n_heavy": n_heavy}


def select(bound: float, cache: dict, tmp: Path) -> tuple[list[str], list[dict], dict]:
    query = rcsb.uniprot_xray_query(ACCESSION, bound)
    ids = rcsb.cached_entry_ids(query, paths.DATA_RAW / f"rcsb_{ACCESSION}_xray_le{bound}.json")
    log: dict[str, str] = {}
    survivors = []
    for pdb_id in ids:
        if pdb_id not in cache:
            cache[pdb_id] = evaluate(pdb_id, tmp)
        r = cache[pdb_id]
        if r["ok"] and r["resolution"] > bound:
            log[pdb_id] = f"resolution {r['resolution']} A in file > bound {bound} A"
        elif r["ok"]:
            survivors.append(r)
        else:
            log[pdb_id] = r["reason"]
    survivors.sort(key=lambda r: (r["resolution"], r["pdb_id"]))
    distinct, rep = [], {}
    for r in survivors:
        if r["inchikey"] in rep:
            log[r["pdb_id"]] = (f"duplicate ligand chemistry: {r['ligand'].resname} "
                                f"({r['inchikey']}) already represented by {rep[r['inchikey']]}")
        else:
            rep[r["inchikey"]] = f"{r['pdb_id']} ({r['resolution']} A)"
            distinct.append(r)
    for r in distinct[MAX_SET:]:
        log[r["pdb_id"]] = (f"passed all filters but outside the {MAX_SET}-structure cap "
                            f"(ranked by resolution, then PDB ID)")
    return ids, distinct[:MAX_SET], dict(sorted(log.items()))


def write_entry(r: dict) -> dict:
    pdb_id, st, ligand, chain = r["pdb_id"], r["st"], r["ligand"], r["chain"]
    _, n_res = structure.write_clean_receptor(st, chain, OUT_DIR / f"{pdb_id}_receptor.pdb")
    info = structure.write_ligand_sdf(r["raw"], ligand, OUT_DIR / f"{pdb_id}_ligand.sdf",
                                      ccd_cache=CCD_CACHE)
    pocket = structure.build_pocket(pdb_id, st, ligand, chain, padding=5.0,
                                    contact_cutoff=POCKET_CUTOFF)
    quality = structure.pocket_quality(st, chain, ligand, {c["seqid"] for c in r["contacts"]},
                                       r["missing"])
    conf = structure.alpha_c_salt_bridge_any_numbering(st, chain)
    rec = {
        "pdb_id": pdb_id,
        "resolution_angstrom": r["resolution"],
        "ligand_code": ligand.resname,
        "ligand_instance": ligand.label,
        "ligand_smiles": info["smiles"],
        "ligand_inchikey": info["inchikey"],
        "ligand_heavy_atoms": info["n_heavy_atoms"],
        "pocket_center": pocket.center,
        "pocket_box_min": pocket.box_min,
        "pocket_box_max": pocket.box_max,
        "pocket_box_size": pocket.box_size,
        "pocket_box_padding_angstrom": pocket.padding_angstrom,
        "mean_pocket_b_factor": quality["mean_b_pocket"],
        "n_residues_within_5A": len(pocket.contact_residues),
        "contact_residues": [f"{c['resname']}{c['seqid']}" for c in pocket.contact_residues],
        "kept_chain": chain,
        "kept_chain_residues": n_res,
        "title": structure.header_info(r["raw"])["title"],
        "conformation": conf,
        "files": {"receptor": f"receptors/benchmark/{pdb_id}_receptor.pdb",
                  "ligand": f"receptors/benchmark/{pdb_id}_ligand.sdf"},
    }
    (OUT_DIR / f"{pdb_id}.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")

    # Acceptance check: the written SDF round-trips with the recorded chemistry.
    mol = Chem.MolFromMolFile(str(OUT_DIR / f"{pdb_id}_ligand.sdf"))
    assert mol is not None, f"{pdb_id}: SDF does not parse"
    assert mol.GetNumHeavyAtoms() == r["n_heavy"], f"{pdb_id}: heavy-atom mismatch"
    assert Chem.MolToInchiKey(mol) == info["inchikey"] == r["inchikey"], f"{pdb_id}: InChIKey"
    return rec


def similarity(recs: list[dict]) -> dict:
    fps = featurize.featurize_smiles([r["ligand_smiles"] for r in recs]).astype(bool)
    pairs = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            inter = np.logical_and(fps[i], fps[j]).sum()
            union = np.logical_or(fps[i], fps[j]).sum()
            pairs.append((float(inter / union), recs[i]["pdb_id"], recs[j]["pdb_id"]))
    sims = np.array([p[0] for p in pairs])
    top = max(pairs, key=lambda p: (p[0], [-ord(c) for c in p[1] + p[2]]))
    return {
        "fingerprint": f"Morgan r={featurize.MORGAN_RADIUS}, {featurize.MORGAN_N_BITS} bits",
        "n_pairs": len(pairs),
        "min": round(float(sims.min()), 3),
        "median": round(float(np.median(sims)), 3),
        "max": round(float(sims.max()), 3),
        "mean": round(float(sims.mean()), 3),
        "most_similar_pair": {"pdb_ids": [top[1], top[2]], "tanimoto": round(top[0], 3)},
        "threshold": SIM_THRESHOLD,
        "n_pairs_above_threshold": int((sims > SIM_THRESHOLD).sum()),
        "pairs_above_threshold": [
            {"pdb_ids": [a, b], "tanimoto": round(s, 3)}
            for s, a, b in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])) if s > SIM_THRESHOLD
        ],
    }


def write_markdown(summary: dict, recs: list[dict], out: Path) -> None:
    sim = summary["similarity"]
    lines = [
        "# EGFR co-crystal docking benchmark set",
        "",
        f"Generated by `src/13_docking_benchmark.py`. RCSB Search API query: UniProt "
        f"{ACCESSION}, X-ray diffraction, `rcsb_entry_info.resolution_combined <= "
        f"{summary['resolution_bound_used']}`.",
        "",
        "## Selection",
        "",
        f"- Entries returned by the query at the initial bound "
        f"({RESOLUTION_STEPS[0]} A): **{summary['query_counts'][str(RESOLUTION_STEPS[0])]}**",
        f"- Resolution bound used: **{summary['resolution_bound_used']} A** - "
        f"{summary['relaxation']}",
        f"- Entries evaluated at the bound used: {summary['n_queried']}; selected "
        f"{summary['n_selected']}; rejected {summary['n_rejected']} (full log in "
        f"`results/docking_benchmark_set.json`).",
        "- Filters, in order: exactly one non-polymer drug-like ligand instance "
        f"({MIN_HEAVY}-{MAX_HEAVY} heavy atoms, `NON_LIGAND_HET` excluded); ligand contacts "
        "the kept chain; no REMARK 465 residue within (or sequence-adjacent to) the 5 A "
        "pocket; CCD template maps onto the crystal copy; distinct ligand InChIKey "
        "(best resolution, then PDB ID, kept); at most 20 structures by resolution.",
        "",
        "Rejections by reason:",
        "",
        "| Reason | Entries |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in summary["rejection_reason_counts"].items()]
    lines += [
        "",
        "## Structures",
        "",
        "| PDB | Res. (A) | Ligand | Heavy atoms | SMILES | Conformation | Lys-Glu (A) | "
        "Numbering | Mean pocket B | Residues <= 5 A |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        c = r["conformation"]
        state = c["state"].split(" ")[0]
        dist = "-" if c["distance_angstrom"] is None else f"{c['distance_angstrom']:.2f}"
        num = (f"{c['numbering']} (K{c['lys_seqid']}/E{c['glu_seqid']})"
               if c["numbering"] else c.get("reason", "-"))
        lines.append(
            f"| {r['pdb_id']} | {r['resolution_angstrom']:.2f} | {r['ligand_code']} | "
            f"{r['ligand_heavy_atoms']} | `{r['ligand_smiles']}` | {state} | {dist} | {num} | "
            f"{r['mean_pocket_b_factor']} | {r['n_residues_within_5A']} |"
        )
    cc = summary["conformation_counts"]
    lines += [
        "",
        "## Conformation",
        "",
        "Beta3-Lys NZ to alphaC-Glu OE1/OE2 minimum distance, <= 4.0 A = active (alphaC-in), "
        "same threshold as the 1M17 call. Numbering chosen per entry by requiring Lys, Glu "
        "and hinge Met (mature 721/738/769, precursor 745/762/793) in the kept chain.",
        "",
        f"- Active: **{cc['active']}**",
        f"- Inactive: **{cc['inactive']}**",
        f"- Undetermined: **{cc['undetermined']}**",
        "",
        "## Ligand diversity",
        "",
        f"Pairwise Tanimoto similarity, {sim['fingerprint']} (`tox21lib.featurize`), "
        f"{sim['n_pairs']} pairs:",
        "",
        "| Statistic | Value |",
        "|---|---|",
        f"| min | {sim['min']:.3f} |",
        f"| median | {sim['median']:.3f} |",
        f"| mean | {sim['mean']:.3f} |",
        f"| max | {sim['max']:.3f} |",
        f"| most similar pair | {' / '.join(sim['most_similar_pair']['pdb_ids'])} "
        f"({sim['most_similar_pair']['tanimoto']:.3f}) |",
        f"| pairs > {sim['threshold']} | {sim['n_pairs_above_threshold']} |",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths.ensure_dirs()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    cache: dict[str, dict] = {}
    counts: dict[str, int] = {}
    passed: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for bound in RESOLUTION_STEPS:
            ids, chosen, log = select(bound, cache, Path(tmpdir))
            counts[str(bound)] = len(ids)
            passed[str(bound)] = len(chosen)
            print(f"bound {bound} A: query returned {len(ids)} entries, "
                  f"{len(chosen)} distinct-ligand structures pass")
            if len(chosen) >= MIN_SET:
                break
        else:
            raise SystemExit(f"only {len(chosen)} structures at {RESOLUTION_STEPS[-1]} A; "
                             f"need {MIN_SET} - stopping without padding")

    relaxation = ("no relaxation needed" if bound == RESOLUTION_STEPS[0] else
                  f"resolution bound relaxed from {RESOLUTION_STEPS[0]} A to {bound} A because "
                  f"fewer than {MIN_SET} distinct-ligand structures passed at tighter bounds "
                  f"(passing per bound: {passed}); no other filter was relaxed")
    recs = [write_entry(r) for r in chosen]

    categories = (
        ("PDB-format", "PDB-format file unavailable"),
        ("0 drug-like", "no drug-like ligand instance (15-60 heavy atoms)"),
        ("drug-like", "more than one drug-like ligand instance"),
        ("ligand ", "ligand not in contact with kept chain"),
        ("unobserved", "unobserved residue in/adjacent to pocket"),
        ("CCD", "CCD template does not map"),
        ("SDF", "SDF heavy-atom mismatch"),
        ("resolution", "resolution above bound"),
        ("duplicate", "duplicate ligand chemistry (InChIKey)"),
        ("passed", f"beyond {MAX_SET}-structure cap"),
    )
    reason_counts: dict[str, int] = {}
    for reason in log.values():
        key = next(label for prefix, label in categories
                   if reason.startswith(prefix) or (prefix == "drug-like" and "drug-like" in reason))
        reason_counts[key] = reason_counts.get(key, 0) + 1
    for r in recs:
        c = r["conformation"]
        print(f"{r['pdb_id']} {r['resolution_angstrom']:.2f} {r['ligand_code']:>4} "
              f"{r['ligand_heavy_atoms']:>2} {c['state'].split(' ')[0]:>12} "
              f"{c['distance_angstrom']} {c['numbering']}")

    conf_counts = {"active": 0, "inactive": 0, "undetermined": 0}
    for r in recs:
        conf_counts[r["conformation"]["state"].split(" ")[0]] += 1

    summary = {
        "accession": ACCESSION,
        "query_counts": counts,
        "resolution_bound_used": bound,
        "relaxation": relaxation,
        "n_queried": len(ids),
        "n_selected": len(recs),
        "n_rejected": len(log),
        "conformation_counts": conf_counts,
        "similarity": similarity(recs),
        "rejection_reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "selected": [r["pdb_id"] for r in recs],
        "structures": recs,
        "rejection_log": log,
    }
    assert set(summary["selected"]) | set(log) == set(ids) and len(recs) + len(log) == len(ids)
    (paths.RESULTS / "docking_benchmark_set.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, recs, paths.RESULTS / "docking_benchmark_set.md")
    print(f"selected {len(recs)}; conformations {conf_counts}; "
          f"similarity {summary['similarity']}")
    print("reasons", summary["rejection_reason_counts"])


if __name__ == "__main__":
    main()
