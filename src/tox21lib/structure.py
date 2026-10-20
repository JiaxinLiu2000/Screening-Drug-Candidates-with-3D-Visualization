"""PDB fetching, receptor cleaning and pocket geometry (gemmi + RDKit).

Used for the EGFR receptors (1M17, 4HJO) that feed the 3Dmol.js viewer.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import gemmi
import numpy as np
import requests

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
CCD_SDF_URL = "https://files.rcsb.org/ligands/download/{resname}_ideal.sdf"

#: HETATM codes that are solvent, buffer, cryoprotectant, ion or crystallization
#: additive - never the drug-like ligand of interest.
NON_LIGAND_HET: frozenset[str] = frozenset(
    {
        "HOH", "DOD", "WAT",                                    # water
        "SO4", "PO4", "ACT", "ACY", "FMT", "CIT", "TRS", "EPE", # buffers
        "MES", "BME", "DTT", "TLA", "MLA", "MPD", "IMD", "AZI",
        "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "2PE", "DMS", # cryo/solvent
        "EOH", "IPA", "MOH", "TBU", "BU3", "P6G", "7PE",
        "NA", "K", "MG", "CA", "ZN", "MN", "FE", "FE2", "CL",   # ions
        "BR", "IOD", "CD", "NI", "CO", "CU", "CS", "RB", "SR",
        "NH4", "NO3", "CO3", "F", "FLC",
        "UNX", "UNL",                                           # unknown
    }
)

#: Minimum heavy atoms for a HETATM group to qualify as "drug-like".
MIN_LIGAND_HEAVY_ATOMS = 10

#: Altloc codes belonging to the single ligand copy we keep (blank, or "A").
#: Everything derived from ligand atoms - heavy-atom counts, pocket centre, box,
#: contact residues, the SDF - uses exactly this selection, so a disordered
#: ligand is never mixed across mutually exclusive conformers.
PRIMARY_ALTLOCS = frozenset({"", " ", "\x00", "A"})


@dataclass(frozen=True)
class HetGroup:
    """One HETATM residue instance in a structure.

    ``n_atoms`` is the raw record count, which includes hydrogens and every
    alternate conformer. ``n_heavy_atoms`` is the chemically meaningful count:
    the primary conformer's heavy atoms, i.e. exactly the atom set written to the
    ligand SDF. Size filters MUST use ``n_heavy_atoms`` - 8SC7's D0D ligand has
    41 records but 31 heavy atoms because three conformers are deposited.
    """

    resname: str
    chain: str
    seqid: int
    n_atoms: int
    n_heavy_atoms: int

    @property
    def label(self) -> str:
        return f"{self.resname}/{self.chain}{self.seqid}"


@dataclass(frozen=True)
class Pocket:
    """Ligand-derived pocket definition written to ``receptors/*_pocket.json``."""

    pdb_id: str
    ligand_resname: str
    ligand_chain: str
    ligand_seqid: int
    center: list[float]
    box_min: list[float]
    box_max: list[float]
    box_size: list[float]
    padding_angstrom: float
    contact_residues: list[dict]
    contact_cutoff_angstrom: float


def fetch_pdb(pdb_id: str, dest_dir: Path, *, timeout: int = 60) -> Path:
    """Download ``<pdb_id>.pdb`` from RCSB into ``dest_dir`` (cached)."""
    pdb_id = pdb_id.upper()
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{pdb_id}.pdb"
    if path.exists() and path.stat().st_size > 0:
        return path
    resp = requests.get(RCSB_PDB_URL.format(pdb_id=pdb_id), timeout=timeout)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def read_structure(path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    return st


def het_groups(st: gemmi.Structure) -> list[HetGroup]:
    """Every HETATM residue instance in model 1, with raw and heavy atom counts."""
    groups: list[HetGroup] = []
    for chain in st[0]:
        for res in chain:
            if res.het_flag != "H":
                continue
            n_heavy = sum(
                1
                for a in res
                if a.altloc in PRIMARY_ALTLOCS and a.element.name != "H"
            )
            groups.append(
                HetGroup(res.name, chain.name, res.seqid.num, len(res), n_heavy)
            )
    return groups


def het_atom_counts(st: gemmi.Structure) -> dict[str, int]:
    """Total atom count per HETATM residue code (summed over instances)."""
    counter: Counter[str] = Counter()
    for g in het_groups(st):
        counter[g.resname] += g.n_atoms
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def protein_chains(st: gemmi.Structure) -> list[tuple[str, int]]:
    """(chain name, residue count) for chains containing amino acids."""
    out = []
    for chain in st[0]:
        n = sum(1 for r in chain if r.het_flag == "A" and gemmi.find_tabulated_residue(r.name)
                and gemmi.find_tabulated_residue(r.name).is_amino_acid())
        if n:
            out.append((chain.name, n))
    return out


def pick_ligand(st: gemmi.Structure) -> HetGroup:
    """Pick the drug-like ligand: largest eligible HETATM group by heavy atoms.

    Selection and the size threshold both use the primary conformer's heavy-atom
    count, so a ligand deposited with several alternate conformers is not treated
    as larger than it chemically is.
    """
    candidates = [
        g
        for g in het_groups(st)
        if g.resname not in NON_LIGAND_HET and g.n_heavy_atoms >= MIN_LIGAND_HEAVY_ATOMS
    ]
    if not candidates:
        raise RuntimeError(f"no drug-like HETATM group found in {st.name}")
    return max(candidates, key=lambda g: (g.n_heavy_atoms, g.chain, -g.seqid))


def main_protein_chain(st: gemmi.Structure, ligand: HetGroup) -> str:
    """Chain to keep: the ligand's own chain if it is a protein chain, else the largest."""
    chains = protein_chains(st)
    if not chains:
        raise RuntimeError("structure contains no protein chain")
    names = {c for c, _ in chains}
    if ligand.chain in names:
        return ligand.chain
    return max(chains, key=lambda c: c[1])[0]


def write_clean_receptor(
    st: gemmi.Structure, chain_name: str, out_path: Path
) -> tuple[Path, int]:
    """Write a PDB with only the protein atoms of ``chain_name``.

    Waters, buffers, cryoprotectants, ions, the ligand and all other chains are
    removed. Returns the path and the number of residues written.
    """
    clean = gemmi.Structure()
    clean.name = st.name
    clean.spacegroup_hm = st.spacegroup_hm
    clean.cell = st.cell
    clean.resolution = st.resolution
    model = gemmi.Model("1")
    chain = gemmi.Chain(chain_name)
    n_res = 0
    for res in st[0][chain_name]:
        if res.het_flag != "A":
            continue
        info = gemmi.find_tabulated_residue(res.name)
        if info is None or not info.is_amino_acid():
            continue
        chain.add_residue(res)
        n_res += 1
    model.add_chain(chain)
    clean.add_model(model)
    clean.setup_entities()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.write_pdb(str(out_path))
    return out_path, n_res



def ligand_coords(st: gemmi.Structure, ligand: HetGroup) -> np.ndarray:
    """(n_atoms, 3) heavy-atom coordinates of ONE copy of the ligand instance.

    Disordered ligands are deposited as several mutually exclusive conformers
    sharing a residue (altloc A/B/C). Averaging across them would place the
    pocket centre between conformers and inflate the box - 8SC7's D0D moved the
    centre by 1.4 A - so only the primary copy counts, matching exactly the atom
    set that ``write_ligand_sdf`` writes. Hydrogens are excluded for the same
    reason: the SDF is heavy-atom only.
    """
    for res in st[0][ligand.chain]:
        if res.name == ligand.resname and res.seqid.num == ligand.seqid:
            xyz = [
                [a.pos.x, a.pos.y, a.pos.z]
                for a in res
                if a.altloc in PRIMARY_ALTLOCS and a.element.name != "H"
            ]
            if not xyz:
                raise RuntimeError(f"ligand {ligand.label} has no primary-conformer atoms")
            return np.array(xyz)
    raise RuntimeError(f"ligand {ligand.label} not found")


def fetch_ccd_template(resname: str, cache_dir: Path, *, timeout: int = 60) -> Path:
    """Download the RCSB Chemical Component Dictionary ideal SDF for a ligand code."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{resname}_ideal.sdf"
    if path.exists() and path.stat().st_size > 0:
        return path
    resp = requests.get(CCD_SDF_URL.format(resname=resname), timeout=timeout)
    resp.raise_for_status()
    path.write_text(resp.text, encoding="utf-8")
    return path


def ccd_template(resname: str, cache_dir: Path):
    """Heavy-atom reference molecule for a ligand code, with correct bond orders."""
    from rdkit import Chem

    path = fetch_ccd_template(resname, cache_dir)
    mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=True)
    if mol is None:
        raise RuntimeError(f"RDKit could not parse the CCD template for {resname}")
    return mol


def write_ligand_sdf(
    pdb_path: Path, ligand: HetGroup, out_path: Path, *, ccd_cache: Path
) -> dict:
    """Extract the ligand to SDF with crystallographic coordinates and real bonds.

    PDB files carry no bond orders, and RDKit's proximity bonding returns every
    bond as a single bond - which silently destroys aromatic rings and triple
    bonds. The connectivity is therefore taken from the RCSB Chemical Component
    Dictionary entry for the ligand code via ``AssignBondOrdersFromTemplate``,
    and the function fails loudly if the crystal copy does not match the
    reference (missing atoms, altloc duplicates), rather than writing a
    chemically wrong file.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromPDBFile(
        str(pdb_path), removeHs=False, sanitize=False, proximityBonding=True
    )
    if mol is None:
        raise RuntimeError(f"RDKit could not read {pdb_path}")

    keep = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if (info := atom.GetPDBResidueInfo()) is not None
        and info.GetResidueName().strip() == ligand.resname
        and info.GetChainId().strip() == ligand.chain
        and info.GetResidueNumber() == ligand.seqid
        and info.GetAltLoc().strip() in ("", "A")
    ]
    if not keep:
        raise RuntimeError(f"ligand {ligand.label} not found by RDKit in {pdb_path}")

    emol = Chem.RWMol(mol)
    for idx in sorted(set(range(mol.GetNumAtoms())) - set(keep), reverse=True):
        emol.RemoveAtom(idx)
    frag = emol.GetMol()
    Chem.SanitizeMol(
        frag,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
    )
    frag = Chem.RemoveHs(frag, sanitize=False)

    template = ccd_template(ligand.resname, ccd_cache)
    try:
        frag = AllChem.AssignBondOrdersFromTemplate(template, frag)
    except Exception as exc:  # noqa: BLE001 - RDKit raises ValueError/RuntimeError
        raise RuntimeError(
            f"could not map the CCD template for {ligand.resname} onto the crystal "
            f"copy in {pdb_path.name} ({frag.GetNumAtoms()} heavy atoms vs "
            f"{template.GetNumAtoms()} in the reference): {exc}"
        ) from exc

    Chem.SanitizeMol(frag)
    frag.SetProp("_Name", f"{pdb_path.stem}_{ligand.resname}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(out_path)) as writer:
        writer.write(frag)

    flat = Chem.Mol(frag)
    flat.RemoveAllConformers()
    return {
        "path": out_path,
        "smiles": Chem.MolToSmiles(flat),
        "inchikey": Chem.MolToInchiKey(flat),
        "n_heavy_atoms": frag.GetNumHeavyAtoms(),
        "n_aromatic_bonds": sum(1 for b in frag.GetBonds() if b.GetIsAromatic()),
        "bond_orders_from": f"RCSB CCD {ligand.resname}_ideal.sdf",
    }


def header_info(pdb_path: Path) -> dict:
    """TITLE and primary-citation lines straight out of the deposited PDB header.

    The conformational state of an EGFR structure is frequently stated in the
    deposition itself; reading it beats relying on recollection.
    """
    title, jrnl_title, jrnl_ref = [], [], ""
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("TITLE"):
            title.append(line[10:].strip())
        elif line.startswith("JRNL        TITL"):
            jrnl_title.append(line[19:].strip())
        elif line.startswith("JRNL        REF ") and not jrnl_ref:
            jrnl_ref = " ".join(line[19:].split())
        elif line.startswith("ATOM"):
            break

    def tidy(parts: list[str]) -> str:
        joined = " ".join(" ".join(parts).split())
        # PDB wraps TITLE across 80-column records, so "4-" + "ANILINO..." comes
        # back as "4- ANILINO...". Rejoin a hyphen that a line break split.
        return re.sub(r"(\w)- ([A-Z0-9])", r"\1-\2", joined)

    return {
        "title": tidy(title),
        "citation_title": tidy(jrnl_title),
        "citation_ref": jrnl_ref,
    }


def missing_residues(pdb_path: Path, chain_name: str) -> list[int]:
    """Residue numbers listed as unobserved in REMARK 465 for ``chain_name``."""
    pattern = re.compile(r"^REMARK 465\s+(\w{1,3})\s+" + re.escape(chain_name) + r"\s+(-?\d+)")
    out = []
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.match(line)
        if m:
            out.append(int(m.group(2)))
        elif line.startswith("ATOM"):
            break
    return sorted(out)


def pocket_quality(
    st: gemmi.Structure,
    chain_name: str,
    ligand: HetGroup,
    pocket_seqids: set[int],
    missing: list[int],
) -> dict:
    """Local quality of the binding site: gaps, disorder and B-factors.

    Resolution is a whole-crystal statistic; what matters for docking is whether
    the atoms lining *this* pocket are actually resolved and single-conformer.
    """
    b_pocket, altloc_atoms = [], 0
    for res in st[0][chain_name]:
        if res.het_flag != "A" or res.seqid.num not in pocket_seqids:
            continue
        for atom in res:
            b_pocket.append(atom.b_iso)
            if atom.altloc not in ("", "\0", " "):
                altloc_atoms += 1

    b_ligand = []
    for res in st[0][ligand.chain]:
        if res.name == ligand.resname and res.seqid.num == ligand.seqid:
            b_ligand = [a.b_iso for a in res]

    modelled = [r.seqid.num for r in st[0][chain_name] if r.het_flag == "A"]
    lo, hi = min(modelled), max(modelled)
    internal_gaps = [m for m in missing if lo < m < hi]

    return {
        "n_pocket_residues": len(pocket_seqids),
        "missing_in_pocket": sorted(pocket_seqids & set(missing)),
        "internal_gap_residues": internal_gaps,
        "n_internal_gap_residues": len(internal_gaps),
        "modelled_residue_span": [lo, hi],
        "altloc_atoms_in_pocket": altloc_atoms,
        "mean_b_pocket": round(float(np.mean(b_pocket)), 1) if b_pocket else None,
        "mean_b_ligand": round(float(np.mean(b_ligand)), 1) if b_ligand else None,
    }


def alpha_c_salt_bridge(
    st: gemmi.Structure,
    chain_name: str,
    *,
    lys_seqid: int = 721,
    glu_seqid: int = 738,
) -> dict:
    """Beta3-lysine / alphaC-glutamate salt-bridge distance -> alphaC-in vs alphaC-out.

    Defaults are the EGFR kinase-domain numbering (Lys721, Glu738). An intact
    bridge (<= 4 A) is the active, alphaC-in state; a broken one is the inactive
    alphaC-out state. This is measured from coordinates, not assumed.
    """

    def atoms(seqid: int, names: set[str]) -> list[np.ndarray]:
        for res in st[0][chain_name]:
            if res.seqid.num == seqid and res.het_flag == "A":
                return [
                    np.array([a.pos.x, a.pos.y, a.pos.z]) for a in res if a.name in names
                ]
        return []

    lys = atoms(lys_seqid, {"NZ"})
    glu = atoms(glu_seqid, {"OE1", "OE2"})
    if not lys or not glu:
        return {"distance_angstrom": None, "state": "undetermined",
                "lys_seqid": lys_seqid, "glu_seqid": glu_seqid}
    d = min(float(np.linalg.norm(a - b)) for a in lys for b in glu)
    return {
        "distance_angstrom": round(d, 2),
        "state": "active (alphaC-in, salt bridge intact)" if d <= 4.0
        else "inactive (alphaC-out, salt bridge broken)",
        "lys_seqid": lys_seqid,
        "glu_seqid": glu_seqid,
    }


def contact_residues(
    st: gemmi.Structure, chain_name: str, coords: np.ndarray, cutoff: float = 5.0
) -> list[dict]:
    """Protein residues of ``chain_name`` with any atom within ``cutoff`` of the ligand."""
    hits: list[dict] = []
    for res in st[0][chain_name]:
        if res.het_flag != "A":
            continue
        xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res])
        if xyz.size == 0:
            continue
        d = np.sqrt(((xyz[:, None, :] - coords[None, :, :]) ** 2).sum(-1)).min()
        if d <= cutoff:
            hits.append(
                {"resname": res.name, "seqid": res.seqid.num, "min_dist": round(float(d), 2)}
            )
    return sorted(hits, key=lambda r: r["seqid"])


def build_pocket(
    pdb_id: str,
    st: gemmi.Structure,
    ligand: HetGroup,
    chain_name: str,
    *,
    padding: float = 5.0,
    contact_cutoff: float = 5.0,
) -> Pocket:
    """Pocket center (ligand centroid) and box (ligand bbox + ``padding`` per side)."""
    coords = ligand_coords(st, ligand)
    center = coords.mean(axis=0)
    box_min = coords.min(axis=0) - padding
    box_max = coords.max(axis=0) + padding
    return Pocket(
        pdb_id=pdb_id,
        ligand_resname=ligand.resname,
        ligand_chain=ligand.chain,
        ligand_seqid=ligand.seqid,
        center=[round(float(v), 3) for v in center],
        box_min=[round(float(v), 3) for v in box_min],
        box_max=[round(float(v), 3) for v in box_max],
        box_size=[round(float(v), 3) for v in (box_max - box_min)],
        padding_angstrom=padding,
        contact_residues=contact_residues(st, chain_name, coords, contact_cutoff),
        contact_cutoff_angstrom=contact_cutoff,
    )


def write_pocket_json(pocket: Pocket, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(pocket), indent=2) + "\n", encoding="utf-8")
    return out_path


def nonpolymer_ligands(st: gemmi.Structure) -> list[HetGroup]:
    """Non-polymer HETATM instances not in ``NON_LIGAND_HET``.

    Modified residues inside a polymer (e.g. PTR, CME) are skipped: they are part
    of the protein chain, not a bound ligand. Each returned group carries its
    ``n_heavy_atoms`` (primary conformer, no H/D), which is what size filters use.
    """
    out = []
    for chain in st[0]:
        for res in chain:
            if res.het_flag != "H" or res.name in NON_LIGAND_HET:
                continue
            if res.entity_type == gemmi.EntityType.Polymer:
                continue
            n_heavy = sum(
                1 for a in res
                if a.altloc in PRIMARY_ALTLOCS and a.element.name not in ("H", "D")
            )
            out.append(
                HetGroup(res.name, chain.name, res.seqid.num, len(res), n_heavy)
            )
    return out


def alpha_c_salt_bridge_any_numbering(st: gemmi.Structure, chain_name: str) -> dict:
    """``alpha_c_salt_bridge`` under mature (721/738) then precursor (745/762) numbering.

    Uses the numbering whose Lys, Glu and hinge Met (Met769 / Met793) are all
    present with the expected residue names - the hinge check guards against a
    coincidental Lys/Glu pair under the wrong numbering. Reports ``undetermined``
    with the reason if no numbering matches, or if both do (ambiguous).
    """
    present = {r.seqid.num: r.name for r in st[0][chain_name] if r.het_flag == "A"}
    tried, matches = [], []
    for label, lys, glu, met in (("mature", 721, 738, 769), ("precursor", 745, 762, 793)):
        tried.append(f"{label} {lys}={present.get(lys, 'absent')}/"
                     f"{glu}={present.get(glu, 'absent')}/{met}={present.get(met, 'absent')}")
        if (present.get(lys), present.get(glu), present.get(met)) == ("LYS", "GLU", "MET"):
            matches.append((label, lys, glu))
    if len(matches) != 1:
        why = "no" if not matches else "ambiguous:"
        return {"distance_angstrom": None, "state": "undetermined", "numbering": None,
                "lys_seqid": None, "glu_seqid": None,
                "reason": f"{why} Lys/Glu/hinge-Met triple matched: " + "; ".join(tried)}
    label, lys, glu = matches[0]
    res = alpha_c_salt_bridge(st, chain_name, lys_seqid=lys, glu_seqid=glu)
    res["numbering"] = label
    if res["state"] == "undetermined":
        res["reason"] = f"Lys{lys}/Glu{glu} present but NZ or OE1/OE2 atoms unmodelled"
    return res
