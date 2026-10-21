"""AutoDock Vina self-docking helpers (Phase 4 D1).

Vina is run as the official pre-built Windows binary (no Python bindings); if the
binary is missing or a run fails, an exception is raised - there is NO fallback
scoring. Receptors are prepared with Meeko ``mk_prepare_receptor``; ligands with
Meeko ``MoleculePreparation`` from a fresh RDKit ETKDG conformer (so the starting
geometry is not the crystal pose). RMSD is RDKit ``rdMolAlign.CalcRMS``: heavy-atom,
symmetry-corrected over all substructure matches, and *without* re-alignment (the
in-place variant of ``GetBestRMS``; GetBestRMS would superimpose the pose onto the
crystal ligand first, which hides docking placement errors).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from . import paths

VINA_VERSION = "1.2.7"
VINA_URL = (
    "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/"
    f"v{VINA_VERSION}/vina_{VINA_VERSION}_win.exe"
)
VINA_SHA256 = "e0c4b2715e0c1a74f6e92d0f3be0328ac97542eafbc111e6b1efad897a73cce5"
DOCK_DIR = paths.DATA_PROCESSED / "docking"
VINA_EXE = DOCK_DIR / "bin" / f"vina_{VINA_VERSION}_win.exe"
MK_RECEPTOR = Path(sys.executable).parent / "mk_prepare_receptor.exe"
EXHAUSTIVENESS = 8
NUM_MODES = 9
SUCCESS_RMSD = 2.0
LIGAND_EMBED_SEED = 0

_RESULT_RE = re.compile(r"^REMARK VINA RESULT:\s+(-?\d+\.\d+)", re.M)


def vina_version() -> str:
    out = subprocess.run([str(VINA_EXE), "--version"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def vina_seed(seed: int) -> int:
    """Vina treats ``--seed 0`` as 'draw a random seed', so project seed s maps to s + 1."""
    return seed + 1


def prepare_receptor(pdb: Path, out_base: Path) -> dict:
    """Meeko receptor PDBQT.

    ``--default_altloc A`` matches ``structure.PRIMARY_ALTLOCS``; ``-a`` drops residues that
    fail Meeko template matching (typically side chains truncated in the deposited model).
    Dropped residues are parsed from Meeko's log and reported.
    """
    proc = subprocess.run(
        [str(MK_RECEPTOR), "--read_pdb", str(pdb), "-o", str(out_base), "-p", "-a",
         "--default_altloc", "A"],
        capture_output=True, text=True,
    )
    pdbqt = out_base.with_suffix(".pdbqt")
    log = proc.stdout + proc.stderr
    if proc.returncode != 0 or not pdbqt.exists():
        raise RuntimeError(f"mk_prepare_receptor failed for {pdb.name}: {log[-2000:]}")
    m = re.search(r"Template matching failed for: \[([^\]]*)\]", log)
    deleted = re.findall(r"'([^']+)'", m.group(1)) if m else []
    return {"pdbqt": pdbqt, "log": log, "deleted_residues": deleted}


def load_crystal_ligand(sdf: Path) -> Chem.Mol:
    mol = Chem.MolFromMolFile(str(sdf), removeHs=True)
    if mol is None:
        raise RuntimeError(f"RDKit could not read {sdf}")
    return mol


def prepare_ligand(crystal: Chem.Mol, out: Path) -> None:
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.AddHs(Chem.Mol(crystal))
    mol.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = LIGAND_EMBED_SEED
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError("ETKDG embedding failed")
    setups = MoleculePreparation().prepare(mol)
    text, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError(f"meeko ligand PDBQT failed: {err}")
    out.write_text(text)


def run_vina(receptor: Path, ligand: Path, center, size, seed: int, out: Path) -> str:
    if not VINA_EXE.exists():
        raise FileNotFoundError(f"Vina binary missing: {VINA_EXE} (download {VINA_URL})")
    cmd = [str(VINA_EXE), "--receptor", str(receptor), "--ligand", str(ligand),
           "--center_x", f"{center[0]}", "--center_y", f"{center[1]}", "--center_z", f"{center[2]}",
           "--size_x", f"{size[0]}", "--size_y", f"{size[1]}", "--size_z", f"{size[2]}",
           "--exhaustiveness", str(EXHAUSTIVENESS), "--num_modes", str(NUM_MODES),
           "--seed", str(seed), "--cpu", "1", "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"vina failed ({receptor.name}, seed {seed}): {proc.stdout[-1000:]}{proc.stderr[-1000:]}")
    return proc.stdout


def score_poses(out: Path, crystal: Chem.Mol) -> list[tuple[float, float]]:
    """(vina_score, symmetry-corrected in-place heavy-atom RMSD) for every output pose, in rank order."""
    from meeko import PDBQTMolecule, RDKitMolCreate

    text = out.read_text()
    scores = [float(s) for s in _RESULT_RE.findall(text)]
    pmol = PDBQTMolecule(text, is_dlg=False, skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pmol)
    if len(mols) != 1 or mols[0] is None:
        raise RuntimeError(f"could not rebuild RDKit molecule from {out.name}")
    docked = Chem.RemoveHs(mols[0])
    if docked.GetNumConformers() != len(scores):
        raise RuntimeError(f"{out.name}: {docked.GetNumConformers()} poses vs {len(scores)} scores")
    res = []
    for i, conf in enumerate(docked.GetConformers()):
        rms = rdMolAlign.CalcRMS(docked, crystal, prbId=conf.GetId())
        res.append((scores[i], rms))
    return res
