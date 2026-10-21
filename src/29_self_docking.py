"""Phase 4 D1 - AutoDock Vina self-docking (redocking) over receptors/benchmark_v2.

For each of the 20 entries the cognate ligand is docked back into its own receptor
with Vina 1.2.7 (official Windows binary), project seeds 0-4 (passed to Vina as
--seed 1-5, because Vina treats --seed 0 as "pick a random seed"), exhaustiveness 8,
9 poses, --cpu 1 (single-threaded so each run is deterministic for its seed). The box is the
recorded pocket box (crystal ligand bounding box + 5 A per side). The top-ranked pose
is scored by symmetry-corrected in-place heavy-atom RMSD (RDKit rdMolAlign.CalcRMS)
against the crystallographic SDF.

    python src/29_self_docking.py            # prepare + dock (skips existing poses) + score
    python src/29_self_docking.py --redock   # force re-running Vina
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import docking, paths  # noqa: E402

BENCH = paths.RECEPTORS / "benchmark_v2"
OUT_CSV = paths.RESULTS / "self_docking.csv"
OUT_JSON = paths.RESULTS / "self_docking.json"
OUT_MD = paths.RESULTS / "self_docking.md"
N_WORKERS = 15


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def r3(x: float) -> float:
    return float(f"{x:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redock", action="store_true")
    args = ap.parse_args()

    if sha(docking.VINA_EXE) != docking.VINA_SHA256:
        raise RuntimeError("Vina binary checksum mismatch")
    vina_ver = docking.vina_version()
    rec_dir, lig_dir, pose_dir = (docking.DOCK_DIR / d for d in ("rec", "lig", "poses"))
    for d in (rec_dir, lig_dir, pose_dir):
        d.mkdir(parents=True, exist_ok=True)

    entries, prep_fail = [], []
    for rec_json in sorted(BENCH.glob("*.json")):
        rec = json.loads(rec_json.read_text())
        pdb = rec["pdb_id"]
        try:
            rprep = docking.prepare_receptor(BENCH / f"{pdb}_receptor.pdb", rec_dir / pdb)
            crystal = docking.load_crystal_ligand(BENCH / f"{pdb}_ligand.sdf")
            lig = lig_dir / f"{pdb}_ligand.pdbqt"
            docking.prepare_ligand(crystal, lig)
        except Exception as e:  # reported, entry excluded - never substituted
            prep_fail.append({"pdb_id": pdb, "error": str(e)})
            print(f"[prep FAIL] {pdb}: {e}")
            continue
        contacts = {int("".join(c for c in r if c.isdigit())) for r in rec["contact_residues"]}
        deleted_pocket = [d for d in rprep["deleted_residues"]
                          if d.split(":")[0] == rec["kept_chain"] and int(d.split(":")[1]) in contacts]
        size = [r3(hi - lo) for lo, hi in zip(rec["pocket_box_min"], rec["pocket_box_max"])]
        entries.append({"rec": rec, "pdb": pdb, "receptor": rprep["pdbqt"], "ligand": lig,
                        "crystal": crystal, "size": size, "deleted": rprep["deleted_residues"],
                        "deleted_pocket": deleted_pocket})
        print(f"[prep] {pdb}: deleted {len(rprep['deleted_residues'])} residues "
              f"({len(deleted_pocket)} in 5A contact set)")

    jobs = [(e, s) for e in entries for s in paths.SEEDS]

    def dock(job):
        e, s = job
        out = pose_dir / f"{e['pdb']}_vseed{docking.vina_seed(s)}.pdbqt"
        if args.redock or not out.exists():
            docking.run_vina(e["receptor"], e["ligand"], e["rec"]["pocket_center"], e["size"],
                             docking.vina_seed(s), out)
        return out

    with ThreadPoolExecutor(N_WORKERS) as ex:
        outs = list(ex.map(dock, jobs))

    rows = []
    for (e, s), out in zip(jobs, outs):
        poses = docking.score_poses(out, e["crystal"])
        rmsds = [p[1] for p in poses]
        rows.append({
            "pdb_id": e["pdb"], "state": e["rec"]["state"], "seed": s, "vina_seed": docking.vina_seed(s),
            "resolution_angstrom": e["rec"]["resolution_angstrom"],
            "mean_pocket_b_factor": e["rec"]["mean_pocket_b_factor"],
            "ligand_code": e["rec"]["ligand_code"], "ligand_heavy_atoms": e["rec"]["ligand_heavy_atoms"],
            "n_poses": len(poses),
            "top_vina_score": poses[0][0], "top_rmsd": r3(rmsds[0]),
            "best_of_poses_rmsd": r3(min(rmsds)), "best_pose_rank": int(np.argmin(rmsds)) + 1,
            "top_success": bool(rmsds[0] <= docking.SUCCESS_RMSD),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, lineterminator="\n")

    per = []
    for pdb, g in df.groupby("pdb_id", sort=True):
        e = next(x for x in entries if x["pdb"] == pdb)
        per.append({
            "pdb_id": pdb, "state": g.state.iloc[0], "ligand_code": g.ligand_code.iloc[0],
            "ligand_heavy_atoms": int(g.ligand_heavy_atoms.iloc[0]),
            "resolution_angstrom": float(g.resolution_angstrom.iloc[0]),
            "mean_pocket_b_factor": float(g.mean_pocket_b_factor.iloc[0]),
            "rmsd_mean": r3(g.top_rmsd.mean()), "rmsd_std": r3(g.top_rmsd.std(ddof=1)),
            "rmsd_min": r3(g.top_rmsd.min()), "rmsd_max": r3(g.top_rmsd.max()),
            "seed_spread": r3(g.top_rmsd.max() - g.top_rmsd.min()),
            "rmsd_seed0": r3(g[g.seed == 0].top_rmsd.iloc[0]),
            "success_seeds": int(g.top_success.sum()),
            "success_mean_rmsd": bool(g.top_rmsd.mean() <= docking.SUCCESS_RMSD),
            "vina_mean": r3(g.top_vina_score.mean()), "vina_std": r3(g.top_vina_score.std(ddof=1)),
            "best_of_poses_rmsd_mean": r3(g.best_of_poses_rmsd.mean()),
            "deleted_residues": e["deleted"], "deleted_pocket_residues": e["deleted_pocket"],
        })
    pe = pd.DataFrame(per)

    def block(sub_df, sub_pe):
        seed_rates = [float((sub_df[sub_df.seed == s].top_success).mean()) for s in paths.SEEDS]
        return {
            "n_entries": int(len(sub_pe)),
            "success_rate_per_seed": [r3(x) for x in seed_rates],
            "success_rate_mean_over_seeds": r3(np.mean(seed_rates)),
            "success_rate_std_over_seeds": r3(np.std(seed_rates, ddof=1)),
            "success_rate_all_runs": r3(sub_df.top_success.mean()),
            "entries_success_all_5_seeds": int((sub_pe.success_seeds == 5).sum()),
            "entries_success_any_seed": int((sub_pe.success_seeds >= 1).sum()),
            "median_top_rmsd_all_runs": r3(sub_df.top_rmsd.median()),
            "median_entry_mean_rmsd": r3(sub_pe.rmsd_mean.median()),
            "best_of_9_success_rate_all_runs": r3((sub_df.best_of_poses_rmsd <= docking.SUCCESS_RMSD).mean()),
            "mean_resolution_angstrom": r3(sub_pe.resolution_angstrom.mean()),
            "mean_pocket_b_factor": r3(sub_pe.mean_pocket_b_factor.mean()),
        }

    summary = {
        "task": "Phase 4 D1 self-docking (redocking)",
        "program": {"name": "AutoDock Vina", "version_string": vina_ver, "source": docking.VINA_URL,
                    "sha256": docking.VINA_SHA256, "how": "official pre-built Windows binary"},
        "receptor_prep": {"tool": "Meeko mk_prepare_receptor", "version": version("meeko"),
                          "flags": "--read_pdb -p -a --default_altloc A"},
        "ligand_prep": {"tool": "Meeko MoleculePreparation + PDBQTWriterLegacy", "version": version("meeko"),
                        "start_geometry": f"RDKit ETKDGv3 conformer, randomSeed={docking.LIGAND_EMBED_SEED}, "
                                          "hydrogens added to CCD bond-order SDF (neutral)"},
        "rmsd": {"function": "rdkit.Chem.rdMolAlign.CalcRMS", "rdkit_version": rdkit.__version__,
                 "definition": "heavy-atom, symmetry-corrected over substructure matches, no re-alignment"},
        "params": {"seeds": list(paths.SEEDS), "vina_seeds": [docking.vina_seed(s) for s in paths.SEEDS],
                   "vina_seed_note": "Vina --seed 0 means 'random seed' (verified: seed-0 runs were not "
                                     "reproducible across two runs), so project seed s is passed as --seed s+1",
                   "exhaustiveness": docking.EXHAUSTIVENESS,
                   "num_modes": docking.NUM_MODES, "cpu_per_run": 1,
                   "success_threshold_angstrom": docking.SUCCESS_RMSD,
                   "box": "pocket_center, size = pocket_box_max - pocket_box_min"},
        "n_entries_attempted": len(entries) + len(prep_fail), "prep_failures": prep_fail,
        "n_runs": int(len(df)),
        "overall": block(df, pe),
        "active": block(df[df.state == "active"], pe[pe.state == "active"]),
        "inactive": block(df[df.state == "inactive"], pe[pe.state == "inactive"]),
        "seed_spread": {"median": r3(pe.seed_spread.median()), "max": r3(pe.seed_spread.max()),
                        "entries_spread_gt_2A": pe[pe.seed_spread > 2.0].pdb_id.tolist(),
                        "entries_mixed_success": pe[(pe.success_seeds > 0) & (pe.success_seeds < 5)].pdb_id.tolist()},
        "threshold_sensitivity": {
            "entries_mean_rmsd_in_2.0_2.5A": pe[(pe.rmsd_mean > 2.0) & (pe.rmsd_mean <= 2.5)].pdb_id.tolist(),
            "success_rate_all_runs_at_2.5A": r3((df.top_rmsd <= 2.5).mean()),
            "success_rate_all_runs_at_3.0A": r3((df.top_rmsd <= 3.0).mean()),
        },
        "per_entry": per,
        "pose_sha256": {p.name: sha(p) for p in outs},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    write_md(summary, pe)
    print(json.dumps({k: summary[k] for k in ("overall", "active", "inactive", "seed_spread")}, indent=1))


def write_md(s: dict, pe: pd.DataFrame) -> None:
    o, a, i = s["overall"], s["active"], s["inactive"]
    L = ["# Phase 4 D1 - Self-docking (redocking) benchmark", "",
         f"Program: **{s['program']['version_string']}** ({s['program']['how']}, sha256 `{s['program']['sha256'][:16]}...`). "
         f"Receptors: {s['receptor_prep']['tool']} (meeko {s['receptor_prep']['version']}, `{s['receptor_prep']['flags']}`). "
         f"Ligands: {s['ligand_prep']['tool']} from {s['ligand_prep']['start_geometry']}. "
         f"Seeds 0-4 (Vina `--seed` 1-5: {s['params']['vina_seed_note']}), exhaustiveness 8, 9 poses, 1 CPU per run, box = recorded pocket box. "
         f"RMSD: `{s['rmsd']['function']}` ({s['rmsd']['definition']}; RDKit {s['rmsd']['rdkit_version']}). "
         "`GetBestRMS` was deliberately not used: it superimposes the pose on the crystal ligand first, which "
         "would erase placement errors; `CalcRMS` is the same symmetry-aware matching evaluated in place.", "",
         f"Entries attempted: {s['n_entries_attempted']}; preparation failures: {len(s['prep_failures'])}"
         + ("" if not s["prep_failures"] else " - " + "; ".join(f"{f['pdb_id']}: {f['error'][:200]}" for f in s["prep_failures"]))
         + f". Docking runs: {s['n_runs']}.", "",
         "## Overall (top-ranked pose, success = RMSD <= 2.0 A)", "",
         f"- Success rate per seed: {o['success_rate_per_seed']} -> mean {o['success_rate_mean_over_seeds']:.3f} +/- {o['success_rate_std_over_seeds']:.3f}",
         f"- Entries successful on all 5 seeds: {o['entries_success_all_5_seeds']}/{o['n_entries']}; on at least one seed: {o['entries_success_any_seed']}/{o['n_entries']}",
         f"- Median top-pose RMSD over all runs: {o['median_top_rmsd_all_runs']:.3f} A",
         f"- Best-of-9-poses success rate (sampling, not scoring): {o['best_of_9_success_rate_all_runs']:.3f}", "",
         "## Seed spread", "",
         f"Per-entry seed spread (max - min top-pose RMSD over 5 seeds): median {s['seed_spread']['median']:.3f} A, max {s['seed_spread']['max']:.3f} A. "
         f"Entries with spread > 2 A: {', '.join(s['seed_spread']['entries_spread_gt_2A']) or 'none'}. "
         f"Entries that succeed on some seeds but not others: {', '.join(s['seed_spread']['entries_mixed_success']) or 'none'}. "
         "A single-seed result for these entries is a coin flip; any between-entry or between-group RMSD difference "
         "smaller than the seed spread is not distinguishable from search noise.", "",
         "## Per entry", "",
         "| PDB | state | lig (HA) | res A | pocket B | RMSD mean +/- std | min-max | spread | success seeds | Vina kcal/mol | pocket residues dropped |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in pe.itertuples():
        L.append(f"| {r.pdb_id} | {r.state} | {r.ligand_code} ({r.ligand_heavy_atoms}) | {r.resolution_angstrom} | {r.mean_pocket_b_factor} | "
                 f"{r.rmsd_mean:.3f} +/- {r.rmsd_std:.3f} | {r.rmsd_min:.3f}-{r.rmsd_max:.3f} | {r.seed_spread:.3f} | {r.success_seeds}/5 | "
                 f"{r.vina_mean:.3f} +/- {r.vina_std:.3f} | {', '.join(r.deleted_pocket_residues) or '-'} |")
    L += ["", "## Conformation stratification", "",
          "| state | n | success rate (mean over seeds) | median top-pose RMSD | mean resolution A | mean pocket B |",
          "|---|---|---|---|---|---|"]
    for name, b in (("active", a), ("inactive", i)):
        L.append(f"| {name} | {b['n_entries']} | {b['success_rate_mean_over_seeds']:.3f} +/- {b['success_rate_std_over_seeds']:.3f} | "
                 f"{b['median_top_rmsd_all_runs']:.3f} | {b['mean_resolution_angstrom']:.3f} | {b['mean_pocket_b_factor']:.3f} |")
    t = s["threshold_sensitivity"]
    L += ["", "**Confound (documented in `results/docking_benchmark_v2.md`):** the active-state entries are systematically "
          "lower resolution with higher pocket B-factors than the inactive-state entries (that report: active median "
          "resolution 2.27 A / median pocket B 55.0; inactive 1.48 A / 19.1), so any per-state difference above is "
          "entangled with site quality (and ligand chemotype) and cannot be attributed to the alphaC conformation. With 10 "
          "entries per group, one entry moves a group success rate by 0.1.", "",
          "## Threshold sensitivity", "",
          f"{len(t['entries_mean_rmsd_in_2.0_2.5A'])} entries have a 5-seed mean top-pose RMSD in (2.0, 2.5] A "
          f"({', '.join(t['entries_mean_rmsd_in_2.0_2.5A'])}), so the 2.0 A success rate is threshold-sensitive: "
          f"success over all runs is {o['success_rate_all_runs']:.3f} at 2.0 A, {t['success_rate_all_runs_at_2.5A']:.3f} at 2.5 A "
          f"and {t['success_rate_all_runs_at_3.0A']:.3f} at 3.0 A.", "",
          "## What this does and does not establish", "",
          "- It establishes that this receptor/ligand preparation, box definition and Vina configuration can (or cannot, per entry) "
          "reproduce the crystallographic pose when the receptor is the ligand's own holo structure - the easiest possible docking task.",
          "- It does NOT establish cross-docking performance (ligand into a different receptor conformation), virtual-screening "
          "enrichment, or that Vina scores rank binding affinity. Vina scores here are reported only as program output; no "
          "affinity claim is made.",
          "- The box is derived from the known ligand, so the search space is biased toward the answer; apo or unknown-site docking would be harder.",
          "- Per-entry numbers are 5-seed samples; differences smaller than the reported seed spread are not real.", ""]
    OUT_MD.write_text("\n".join(L))


if __name__ == "__main__":
    main()
