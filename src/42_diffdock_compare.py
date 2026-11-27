"""Phase 5 Task 3 - DiffDock against AutoDock Vina on the same complexes.

The proposal promised a Vina-versus-DiffDock comparison and it was never run;
`results/PHASE2_DECISIONS.md` recorded at the time that the 20-structure EGFR set
could not support such a claim anyway. With the benchmark now at 428 + 85
cross-target complexes, the comparison is finally meaningful.

DiffDock cannot share this project's environment: it pins Python 3.9.18,
torch 1.13.1+cu117 and torch-geometric 2.2.0, against 3.11.17 / 2.14.1+cu126 /
2.8.1 here. It therefore runs in the official container (`docker build -t
diffdock external/DiffDock`), which carries its own micromamba environment, and
this script only prepares inputs, invokes the container, and scores the output
with the *same* RMSD function Vina was scored with.

Two properties make the comparison fair rather than merely parallel:

* **Same complexes.** Only complexes Vina also scored are used, read from
  `results/docking_scaleup.csv`. A complex excluded there - because its pocket
  residues could not be parameterised - is excluded here too.
* **Same metric.** RMSD is `tox21lib.docking.score_poses`' symmetry-corrected,
  in-place heavy-atom RMSD, not DiffDock's own confidence score. The two methods
  report different quantities natively and comparing those would compare nothing.

What the comparison cannot answer: DiffDock is blind (it is given the whole
protein), while our Vina runs are self-docking (the box comes from the crystal
ligand of the same structure). Vina is therefore handed information DiffDock is
not. This is stated in the report rather than corrected, because correcting it
would mean re-running Vina blind, which is a different experiment.

    python src/42_diffdock_compare.py --smoke          # 4 complexes
    python src/42_diffdock_compare.py --limit 60
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import docking, paths  # noqa: E402

RDLogger.DisableLog("rdApp.*")

IMAGE = "diffdock"
SETS = {
    "posebusters": paths.DATA_RAW / "posebusters" / "posebusters_benchmark_set",
    "astex": paths.DATA_RAW / "posebusters" / "astex_diverse_set",
}
WORK = paths.DATA_PROCESSED / "docking" / "diffdock"
VINA_CSV = paths.RESULTS / "docking_scaleup.csv"

OUT_CSV = paths.RESULTS / "diffdock_compare.csv"
OUT_JSON = paths.RESULTS / "diffdock_compare.json"
OUT_MD = paths.RESULTS / "diffdock_compare.md"

#: DiffDock's default; kept so the published numbers are reproducible.
SAMPLES_PER_COMPLEX = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="first 4 complexes only")
    p.add_argument("--limit", type=int, default=0, help="cap complexes (0 = all)")
    p.add_argument("--batch-size", type=int, default=10)
    return p.parse_args()


def require_image() -> str:
    """Fail loudly if the container is missing; never fall back to a stand-in."""
    probe = subprocess.run(["docker", "image", "inspect", IMAGE],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        raise SystemExit(
            f"FAILED: docker image '{IMAGE}' not found. Build it first:\n"
            f"    docker build external/DiffDock -f external/DiffDock/Dockerfile "
            f"-t {IMAGE}\n"
            "There is deliberately no fallback: a 'comparison' against a "
            "substitute method would not be the comparison the proposal promised.")
    gpu = subprocess.run(
        ["docker", "run", "--rm", "--gpus", "all", IMAGE, "sh", "-c",
         "micromamba run -n diffdock python -c "
         "\"import torch;print(torch.cuda.is_available())\""],
        capture_output=True, text=True)
    if "True" not in gpu.stdout:
        raise SystemExit(
            "FAILED: the container cannot see the GPU. DiffDock on CPU is far "
            f"too slow for a benchmark.\n{gpu.stdout[-500:]}{gpu.stderr[-500:]}")
    return gpu.stdout.strip()


def complexes_vina_scored() -> pd.DataFrame:
    """The complexes Vina produced a pose for, so the two sets match exactly."""
    if not VINA_CSV.exists():
        raise SystemExit(
            f"FAILED: {VINA_CSV} missing; run src/41_docking_scaleup.py first")
    frame = pd.read_csv(VINA_CSV)
    scored = frame[frame["error"].isna() & frame["top_rmsd"].notna()]
    if scored.empty:
        raise SystemExit(f"FAILED: {VINA_CSV} holds no successfully docked rows")
    return scored[["set", "complex", "top_rmsd", "best_of_poses_rmsd",
                   "top_success", "best_of_success"]].copy()


def build_input_csv(rows: pd.DataFrame, dest: Path) -> Path:
    """DiffDock's batch input: one line per complex, paths inside the container."""
    records = []
    for _, row in rows.iterrows():
        name = row["complex"]
        entry = SETS[row["set"]] / name
        records.append({
            "complex_name": name,
            # /inputs is where the host's data/raw/posebusters is mounted.
            "protein_path": f"/inputs/{row['set']}/{name}/{name}_protein.pdb",
            "ligand_description": str(
                Chem.MolToSmiles(docking.load_crystal_ligand(
                    entry / f"{name}_ligand.sdf"))),
            "protein_sequence": "",
        })
    frame = pd.DataFrame(records)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    return dest


def run_diffdock(input_csv: Path, out_dir: Path, batch_size: int) -> str:
    """Invoke the container once for the whole batch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mounts = [
        "-v", f"{(paths.DATA_RAW / 'posebusters').resolve()}:/inputs:ro",
        "-v", f"{input_csv.parent.resolve()}:/batch",
        "-v", f"{out_dir.resolve()}:/out",
    ]
    cmd = [
        "docker", "run", "--rm", "--gpus", "all", *mounts, IMAGE, "sh", "-c",
        "micromamba run -n diffdock python -m inference "
        f"--protein_ligand_csv /batch/{input_csv.name} "
        "--out_dir /out "
        f"--samples_per_complex {SAMPLES_PER_COMPLEX} "
        f"--batch_size {batch_size}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"DiffDock container failed:\n{log[-3000:]}")
    return log


def score_diffdock(out_dir: Path, rows: pd.DataFrame) -> list[dict]:
    """RMSD of DiffDock's ranked poses, using the project's own RMSD function."""
    records = []
    for _, row in rows.iterrows():
        name = row["complex"]
        rec: dict = {"set": row["set"], "complex": name, "diffdock_error": None}
        try:
            folder = out_dir / name
            if not folder.is_dir():
                raise FileNotFoundError(f"no output folder for {name}")
            crystal = docking.load_crystal_ligand(
                SETS[row["set"]] / name / f"{name}_ligand.sdf")
            # DiffDock writes rank1.sdf, rank2.sdf ... plus *_confidence*.sdf
            # duplicates; the bare rankN files are the ranked set.
            poses = sorted(
                (p for p in folder.glob("rank*.sdf") if "confidence" not in p.name),
                key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
            if not poses:
                raise FileNotFoundError(f"no rank*.sdf in {folder}")
            rmsds = []
            for pose in poses:
                mol = Chem.MolFromMolFile(str(pose), removeHs=True)
                if mol is None:
                    continue
                from rdkit.Chem import rdMolAlign
                rmsds.append(float(rdMolAlign.CalcRMS(mol, crystal)))
            if not rmsds:
                raise RuntimeError("no pose could be read back")
            rec.update({
                "diffdock_n_poses": len(rmsds),
                "diffdock_top_rmsd": rmsds[0],
                "diffdock_best_of_poses_rmsd": min(rmsds),
                "diffdock_best_pose_rank": int(np.argmin(rmsds)) + 1,
                "diffdock_top_success": bool(rmsds[0] <= docking.SUCCESS_RMSD),
                "diffdock_best_of_success": bool(
                    min(rmsds) <= docking.SUCCESS_RMSD),
            })
        except Exception as exc:
            rec["diffdock_error"] = f"{type(exc).__name__}: {exc}"
        records.append(rec)
    return records


def write_report(df: pd.DataFrame, summary: dict) -> None:
    lines: list[str] = []
    add = lines.append
    add("# DiffDock against AutoDock Vina")
    add("")
    add("Generated by `src/42_diffdock_compare.py`. DiffDock runs in the "
        "official container (its environment is incompatible with this "
        "project's); both methods are scored with the same symmetry-corrected, "
        "in-place heavy-atom RMSD.")
    add("")
    add(f"Paired on the {summary['n_paired']} complexes both methods produced a "
        "pose for.")
    add("")
    add("| method | top-pose within 2 A | best-of-N within 2 A | median top RMSD |")
    add("|---|---|---|---|")
    add(f"| AutoDock Vina (self-docking) | {summary['vina_top_rate']:.1%} | "
        f"{summary['vina_best_of_rate']:.1%} | {summary['vina_median_top']:.2f} |")
    add(f"| DiffDock (blind) | {summary['diffdock_top_rate']:.1%} | "
        f"{summary['diffdock_best_of_rate']:.1%} | "
        f"{summary['diffdock_median_top']:.2f} |")
    add("")
    add("## How to read this")
    add("")
    add("The two columns are **not** an apples-to-apples ranking of the methods. "
        "Vina here is self-docking: its search box is built from the crystal "
        "ligand of the very structure being redocked, so it is told where the "
        "pocket is. DiffDock is given the whole protein and has to find the site "
        "itself. A Vina advantage in this table is partly that head start.")
    add("")
    add(f"Both agree on {summary['n_both_success']} complexes and both fail on "
        f"{summary['n_both_fail']}; Vina alone succeeds on "
        f"{summary['n_vina_only']}, DiffDock alone on {summary['n_diffdock_only']}.")
    add("")
    if summary["n_diffdock_failed"]:
        add(f"{summary['n_diffdock_failed']} complexes produced no readable "
            "DiffDock pose and are excluded from the paired rates above; they "
            "are kept in the CSV with the reason.")
        add("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")


def main() -> int:
    args = parse_args()
    version = require_image()
    print(f"container GPU check: {version}")

    rows = complexes_vina_scored()
    if args.smoke:
        rows = rows.head(4)
    elif args.limit:
        rows = rows.head(args.limit)
    print(f"{len(rows)} complexes (paired with the Vina run)")

    batch_dir = WORK / "batch"
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    input_csv = build_input_csv(rows, batch_dir / "complexes.csv")
    out_dir = WORK / "out"

    t0 = time.time()
    log = run_diffdock(input_csv, out_dir, args.batch_size)
    elapsed = time.time() - t0
    print(f"DiffDock took {elapsed:.0f}s for {len(rows)} complexes")
    (WORK / "diffdock_run.log").write_text(log[-20000:], encoding="utf-8")

    scored = pd.DataFrame(score_diffdock(out_dir, rows))
    merged = rows.merge(scored, on=["set", "complex"], how="left")
    merged.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(merged)} rows)")

    paired = merged[merged["diffdock_error"].isna()]
    summary = {
        "image": IMAGE,
        "samples_per_complex": SAMPLES_PER_COMPLEX,
        "success_rmsd_angstrom": docking.SUCCESS_RMSD,
        "seconds": round(elapsed, 1),
        "n_attempted": int(len(merged)),
        "n_paired": int(len(paired)),
        "n_diffdock_failed": int(merged["diffdock_error"].notna().sum()),
        "smoke": bool(args.smoke),
        "vina_mode": "self-docking (box from the crystal ligand)",
        "diffdock_mode": "blind (whole protein)",
    }
    if len(paired):
        summary.update({
            "vina_top_rate": float(paired["top_success"].mean()),
            "vina_best_of_rate": float(paired["best_of_success"].mean()),
            "vina_median_top": float(paired["top_rmsd"].median()),
            "diffdock_top_rate": float(paired["diffdock_top_success"].mean()),
            "diffdock_best_of_rate": float(paired["diffdock_best_of_success"].mean()),
            "diffdock_median_top": float(paired["diffdock_top_rmsd"].median()),
            "n_both_success": int((paired["top_success"]
                                   & paired["diffdock_top_success"]).sum()),
            "n_both_fail": int((~paired["top_success"]
                                & ~paired["diffdock_top_success"]).sum()),
            "n_vina_only": int((paired["top_success"]
                                & ~paired["diffdock_top_success"]).sum()),
            "n_diffdock_only": int((~paired["top_success"]
                                    & paired["diffdock_top_success"]).sum()),
        })
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")

    if len(paired):
        write_report(merged, summary)
        print(f"\nVina {summary['vina_top_rate']:.1%} vs "
              f"DiffDock {summary['diffdock_top_rate']:.1%} top-pose "
              f"(n={summary['n_paired']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
