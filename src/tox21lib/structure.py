"""PDB fetching, receptor cleaning and pocket geometry (gemmi + RDKit).

Used for the EGFR receptors (1M17, 4HJO) that feed the 3Dmol.js viewer.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import gemmi
import numpy as np
import requests

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

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


@dataclass(frozen=True)
class HetGroup:
    """One HETATM residue instance in a structure."""

    resname: str
    chain: str
    seqid: int
    n_atoms: int

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
    """Every HETATM residue instance in model 1, with its atom count."""
    groups: list[HetGroup] = []
    for chain in st[0]:
        for res in chain:
            if res.het_flag != "H":
                continue
            groups.append(
                HetGroup(res.name, chain.name, res.seqid.num, len(res))
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
    """Pick the drug-like ligand: largest HETATM group not in the exclusion list."""
    candidates = [
        g
        for g in het_groups(st)
        if g.resname not in NON_LIGAND_HET and g.n_atoms >= MIN_LIGAND_HEAVY_ATOMS
    ]
    if not candidates:
        raise RuntimeError(f"no drug-like HETATM group found in {st.name}")
    return max(candidates, key=lambda g: (g.n_atoms, g.chain, -g.seqid))


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
    """(n_atoms, 3) coordinates of the selected ligand instance (heavy + H)."""
    for res in st[0][ligand.chain]:
        if res.name == ligand.resname and res.seqid.num == ligand.seqid:
            return np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res])
    raise RuntimeError(f"ligand {ligand.label} not found")


def write_ligand_sdf(
    pdb_path: Path, ligand: HetGroup, out_path: Path
) -> Path:
    """Extract the ligand to SDF via RDKit, keeping experimental coordinates.

    Bond orders are perceived by RDKit from PDB connectivity/geometry; this is
    adequate for visualization, which is the ligand file's only use in v1.
    """
    from rdkit import Chem

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
    frag.SetProp("_Name", f"{Path(pdb_path).stem}_{ligand.resname}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(out_path)) as writer:
        writer.write(frag)
    return out_path


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
