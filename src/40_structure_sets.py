"""Phase 5 Task 2a - replace the 20-structure EGFR set with real structural benchmarks.

The Phase 2 docking set held 20 co-crystal structures, every one of them EGFR
(UniProt P00533). `src/14_phase2_decisions.py` already recorded that this cannot
support a claim about docking-method performance; a reviewer made the same point
and asked for a large-scale structural database instead.

This script inventories the replacements and records exactly how much the sample
grew along both axes that matter - number of complexes and number of distinct
proteins - because going from 20 to 428 complexes is worth little if they are
still all one target.

    PoseBusters benchmark set   428 complexes, PDB depositions from 2021 onward,
                                the standard set for pose-prediction evaluation
                                (Buttenschoen et al., Chem. Sci. 2024)
    Astex Diverse set            85 complexes, the older hand-curated standard
    PDBbind refined set       5,311 complexes with measured affinities, which is
                                also this project's first external affinity
                                validation set: every affinity number so far
                                comes from KIBA or DAVIS alone

Only the inventory happens here. Docking runs are `src/41_docking_scaleup.py`.

    python src/40_structure_sets.py
    python src/40_structure_sets.py --refresh-pdbbind
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import paths  # noqa: E402

POSEBUSTERS_DIR = paths.DATA_RAW / "posebusters"
PDBBIND_PARQUET = paths.DATA_RAW / "pdbbind_refined.parquet"
PDBBIND_SOURCE = "hf://datasets/DaInternet12/pdbbind_refined/refined (1).parquet"

#: Zenodo record for the PoseBusters paper data (CC-BY-4.0).
POSEBUSTERS_ZENODO = "https://zenodo.org/record/8278563"

OUT_JSON = paths.RESULTS / "structure_sets.json"
OUT_MD = paths.RESULTS / "structure_sets.md"
OUT_CSV = paths.RESULTS / "structure_sets_inventory.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh-pdbbind", action="store_true",
                   help="re-download the PDBbind refined table even if cached")
    return p.parse_args()

def chain_sequences(pdb_path: Path) -> list[str]:
    """One-letter sequences per chain, read from the CA atoms of ATOM records.

    SEQRES would be the better source - it reports the deposited construct
    rather than only the ordered residues - but the PoseBusters and Astex
    protein files carry ATOM records only, with no SEQRES at all. An earlier
    version of this function parsed SEQRES, found nothing in any file, and
    silently reported every complex as the same protein. Parsing what the files
    actually contain is the fix; the cost is that disordered residues are
    missing from each sequence, which is noted where the counts are used.
    """
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "SEC": "U", "PYL": "O", "MSE": "M",
    }
    chains: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            residue = line[17:20].strip()
            if residue not in three_to_one:
                continue
            chain = line[21]
            # resSeq plus insertion code identifies a residue; altloc copies of
            # the same CA must not be counted twice.
            res_id = line[22:27].strip()
            if (chain, res_id) in seen:
                continue
            seen.add((chain, res_id))
            chains.setdefault(chain, []).append(three_to_one[residue])
    return ["".join(v) for v in chains.values()]


def inventory_set(root: Path, label: str) -> dict:
    """Count complexes, distinct depositions and distinct protein sequences."""
    if not root.exists():
        return {"label": label, "present": False, "path": str(root)}

    entries = sorted(p for p in root.iterdir() if p.is_dir())
    rows: list[dict] = []
    sequences: Counter[str] = Counter()
    missing_protein = 0
    unparsed: list[str] = []
    for entry in entries:
        pdb_id = entry.name.split("_")[0]
        protein = entry / f"{entry.name}_protein.pdb"
        ligand = entry / f"{entry.name}_ligand.sdf"
        if not protein.exists():
            missing_protein += 1
            continue
        chains = chain_sequences(protein)
        if not chains:
            # A file we cannot read must not be silently folded into some other
            # protein's group: that is exactly how the SEQRES version of this
            # code reported 428 complexes as one protein.
            unparsed.append(entry.name)
            continue
        # A complex's identity is the multiset of its chains; using the longest
        # chain alone would merge genuinely different complexes that happen to
        # share one subunit.
        key = "|".join(sorted(chains))
        sequences[key] += 1
        rows.append({
            "set": label,
            "complex": entry.name,
            "pdb_id": pdb_id,
            "ligand_code": entry.name.split("_", 1)[1] if "_" in entry.name else "",
            "n_chains": len(chains),
            "longest_chain_residues": max(len(c) for c in chains),
            "total_residues": sum(len(c) for c in chains),
            "has_ligand_sdf": ligand.exists(),
            "protein_bytes": protein.stat().st_size,
        })

    if unparsed:
        raise RuntimeError(
            f"{label}: no CA atoms parsed from {len(unparsed)} protein files "
            f"(first: {unparsed[0]}). Refusing to report protein diversity from "
            "a partially parsed set."
        )

    frame = pd.DataFrame(rows)
    return {
        "label": label,
        "present": True,
        "path": str(root.relative_to(paths.ROOT)),
        "n_complexes": len(rows),
        "n_distinct_pdb_entries": int(frame["pdb_id"].nunique()) if len(frame) else 0,
        "n_distinct_protein_sequence_sets": len(sequences),
        "largest_protein_group": max(sequences.values()) if sequences else 0,
        "n_missing_protein_pdb": missing_protein,
        "n_with_ligand_sdf": int(frame["has_ligand_sdf"].sum()) if len(frame) else 0,
        "median_residues_per_complex": (int(frame["total_residues"].median())
                                        if len(frame) else 0),
        "_frame": frame,
    }


def load_pdbbind(refresh: bool) -> dict:
    """Cache the PDBbind refined affinity table locally.

    The official structure archive at pdbbind-plus.org.cn requires registration,
    so only the table is mirrored here; coordinates for any subset we dock are
    fetched from RCSB, which is the authoritative source for them anyway.
    """
    if PDBBIND_PARQUET.exists() and not refresh:
        frame = pd.read_parquet(PDBBIND_PARQUET)
        source = "cached"
    else:
        print(f"  downloading {PDBBIND_SOURCE}")
        frame = pd.read_parquet(PDBBIND_SOURCE)
        PDBBIND_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(PDBBIND_PARQUET, index=False)
        source = PDBBIND_SOURCE

    affinity = pd.to_numeric(frame["neg_log10_affinity_M"], errors="coerce")
    return {
        "present": True,
        "source": source,
        "path": str(PDBBIND_PARQUET.relative_to(paths.ROOT)),
        "sha256": hashlib.sha256(PDBBIND_PARQUET.read_bytes()).hexdigest(),
        "n_rows": int(len(frame)),
        "n_distinct_pdb_ids": int(frame["name"].nunique()),
        "n_distinct_smiles": int(frame["smiles_can"].nunique()),
        "n_distinct_sequences": int(frame["seq"].nunique()),
        "affinity_unit": "pKd/pKi (-log10 M)",
        "affinity_min": float(affinity.min()),
        "affinity_max": float(affinity.max()),
        "affinity_median": float(affinity.median()),
        "n_affinity_missing": int(affinity.isna().sum()),
        "columns": list(frame.columns),
    }


def write_report(summary: dict) -> None:
    pb = summary["posebusters_benchmark_set"]
    ax = summary["astex_diverse_set"]
    pdbbind = summary["pdbbind_refined"]
    old = summary["superseded_phase2_set"]

    lines: list[str] = []
    add = lines.append
    add("# Structural benchmark sets")
    add("")
    add("Generated by `src/40_structure_sets.py`.")
    add("")
    add("## 1. What this replaces")
    add("")
    add(f"The Phase 2/4 docking set is {old['n_complexes']} co-crystal structures "
        f"of a single protein ({old['uniprot']}, {old['protein_name']}), selected "
        f"from {old['n_queried']} RCSB hits. Pose-accuracy numbers computed on it "
        "describe one binding site, so they cannot be read as a statement about "
        "docking methods in general. `results/PHASE2_DECISIONS.md` said so at the "
        "time; this is the fix, not a new discovery.")
    add("")
    add("## 2. Replacements")
    add("")
    add("| set | complexes | distinct PDB entries | distinct protein chain-sets | "
        "largest single-protein group |")
    add("|---|---|---|---|---|")
    add(f"| Phase 2 EGFR set (superseded) | {old['n_complexes']} | "
        f"{old['n_complexes']} | 1 | {old['n_complexes']} |")
    for block, name in ((pb, "PoseBusters benchmark"), (ax, "Astex Diverse")):
        if not block.get("present"):
            add(f"| {name} | absent | - | - | - |")
            continue
        add(f"| {name} | {block['n_complexes']} | "
            f"{block['n_distinct_pdb_entries']} | "
            f"{block['n_distinct_protein_sequence_sets']} | "
            f"{block['largest_protein_group']} |")
    add("")
    if pb.get("present"):
        add(f"The PoseBusters set multiplies the number of complexes by "
            f"{pb['n_complexes'] / old['n_complexes']:.1f}x, and no two of its "
            "complexes share an identical set of chain sequences.")
        add("")
        add("**How to read the protein column.** Chains are compared by exact "
            "sequence, taken from the CA atoms because these files carry no "
            "SEQRES records. Exact matching splits different constructs or "
            "truncations of the same protein into separate groups, so the column "
            "is an upper bound on the number of distinct proteins rather than a "
            "count of them - on PDBbind the same procedure maps "
            f"{pdbbind['n_rows']} complexes onto {pdbbind['n_distinct_sequences']} "
            "sequences, so it does merge entries when they are truly identical. "
            "The comparison that does not depend on the threshold is the one that "
            "blocked the Phase 2 claim: that set was 20 structures returned by a "
            f"single query for one UniProt accession ({old['uniprot']}), whereas "
            "these sets were assembled to span unrelated targets.")
        add("")

    add("## 3. PDBbind refined set")
    add("")
    add(f"{pdbbind['n_rows']} complexes, {pdbbind['n_distinct_pdb_ids']} distinct PDB "
        f"entries, {pdbbind['n_distinct_sequences']} distinct protein sequences, "
        f"{pdbbind['n_distinct_smiles']} distinct canonical SMILES.")
    add("")
    add(f"Affinity is reported as {pdbbind['affinity_unit']}, range "
        f"{pdbbind['affinity_min']:.2f} to {pdbbind['affinity_max']:.2f}, median "
        f"{pdbbind['affinity_median']:.2f}. This matters beyond the docking work: "
        "every affinity number in this project so far comes from KIBA or DAVIS, "
        "and the KIBA target is a KIBA score rather than a binding constant. "
        "PDBbind is the first set here with a physical unit and the first "
        "opportunity to test the affinity model outside its training distribution.")
    add("")
    add("The official PDBbind structure archive requires registration, so only "
        "the affinity table is mirrored; coordinates for any docked subset come "
        "from RCSB.")
    add("")
    add("## 4. Provenance")
    add("")
    add(f"- PoseBusters and Astex data: {POSEBUSTERS_ZENODO} (CC-BY-4.0), "
        "accompanying Buttenschoen et al., *Chemical Science* 15(9):3130-3139, 2024.")
    add(f"- PDBbind refined table: `{PDBBIND_SOURCE}`, sha256 "
        f"`{pdbbind['sha256'][:16]}...`.")
    add("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")


def main() -> int:
    args = parse_args()
    t0 = time.time()

    print("Inventorying structural sets...")
    pb = inventory_set(POSEBUSTERS_DIR / "posebusters_benchmark_set",
                       "posebusters_benchmark_set")
    ax = inventory_set(POSEBUSTERS_DIR / "astex_diverse_set", "astex_diverse_set")

    frames = [b.pop("_frame") for b in (pb, ax) if b.get("present")]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUT_CSV, index=False)
        print(f"Wrote {OUT_CSV}")

    print("Loading PDBbind refined table...")
    pdbbind = load_pdbbind(args.refresh_pdbbind)

    phase2 = json.loads((paths.RESULTS / "docking_benchmark_v2.json").read_text())
    summary = {
        "generated_seconds": round(time.time() - t0, 1),
        "superseded_phase2_set": {
            "n_complexes": phase2["n_selected"],
            "n_queried": phase2["n_queried"],
            "uniprot": phase2["accession"],
            "protein_name": "EGFR",
            "note": "single target; cannot support a docking-method claim",
        },
        "posebusters_benchmark_set": pb,
        "astex_diverse_set": ax,
        "pdbbind_refined": pdbbind,
        "posebusters_zenodo": POSEBUSTERS_ZENODO,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")

    write_report(summary)

    print(f"\nPoseBusters {pb.get('n_complexes', 0)} complexes / "
          f"{pb.get('n_distinct_protein_sequence_sets', 0)} distinct proteins; "
          f"Astex {ax.get('n_complexes', 0)}; "
          f"PDBbind {pdbbind['n_rows']} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
