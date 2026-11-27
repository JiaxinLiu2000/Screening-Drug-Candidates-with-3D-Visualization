"""Phase 5 Task 2b - redocking across 428 unrelated targets, with physical validity.

Phase 4 measured Vina self-docking on 20 EGFR co-crystals: 41.0% of top-ranked
poses within 2 A and 76.0% best-of-9. Those numbers describe one binding site.
This script runs the identical protocol over the PoseBusters benchmark set (428
complexes, no two sharing a chain-sequence set) and the Astex Diverse set (85),
so the success rate is measured across unrelated proteins instead of one.

Two things are added beyond scale:

* **Physical validity.** RMSD alone rewards a pose that is in roughly the right
  place while being chemically impossible. Every top-ranked pose is therefore
  put through the PoseBusters ``redock`` battery (bond lengths and angles, ring
  flatness, internal energy ratio, steric clash with protein, cofactors and
  waters, volume overlap, and RMSD). A pose counts as a success here only if it
  is both within 2 A and physically valid, which is a stricter and more honest
  definition than the Phase 4 one.
* **Per-target reporting.** Success is reported per complex, so the spread
  across targets is visible rather than averaged into a single rate.

Vina settings are deliberately unchanged from `src/29_self_docking.py`
(exhaustiveness 8, 9 modes, ``--cpu 1``, ligand rebuilt by ETKDG rather than
started from the crystal pose), because the point is to re-measure the same
protocol on a bigger sample, not to tune it.

    python src/41_docking_scaleup.py --smoke            # 6 complexes
    python src/41_docking_scaleup.py --set posebusters  # all 428
    python src/41_docking_scaleup.py --set all --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import docking, paths  # noqa: E402

RDLogger.DisableLog("rdApp.*")

SETS = {
    "posebusters": paths.DATA_RAW / "posebusters" / "posebusters_benchmark_set",
    "astex": paths.DATA_RAW / "posebusters" / "astex_diverse_set",
}
WORK = paths.DATA_PROCESSED / "docking" / "scaleup"
BOX_PADDING = 5.0          # same as the Phase 2 pocket box
DOCK_SEED = 0              # project seed; Vina sees vina_seed(0) == 1

OUT_CSV = paths.RESULTS / "docking_scaleup.csv"
OUT_JSON = paths.RESULTS / "docking_scaleup.json"
OUT_MD = paths.RESULTS / "docking_scaleup.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=("posebusters", "astex", "all"),
                   default="posebusters")
    p.add_argument("--workers", type=int, default=6,
                   help="parallel complexes; each Vina run stays --cpu 1")
    p.add_argument("--smoke", action="store_true", help="first 6 complexes only")
    p.add_argument("--limit", type=int, default=0, help="cap complexes (0 = all)")
    return p.parse_args()


def ligand_box(mol: Chem.Mol, padding: float = BOX_PADDING):
    """Centre and size of the crystal ligand's bounding box, padded per side.

    Matches the Phase 2 convention so the two benchmarks are comparable: the box
    is derived from the crystal ligand, which is the standard self-docking setup
    and is why these numbers are an upper bound on blind-docking performance.
    """
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i))
                       for i in range(mol.GetNumAtoms())])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    centre = (lo + hi) / 2.0
    size = (hi - lo) + 2.0 * padding
    return centre.tolist(), size.tolist()


def dock_one(entry: Path, set_name: str) -> dict:
    """Prepare, dock and validate one complex. Never raises; records the error."""
    name = entry.name
    row: dict = {"set": set_name, "complex": name,
                 "pdb_id": name.split("_")[0], "error": None}
    try:
        protein = entry / f"{name}_protein.pdb"
        ligand_sdf = entry / f"{name}_ligand.sdf"
        if not protein.exists() or not ligand_sdf.exists():
            raise FileNotFoundError("missing protein.pdb or ligand.sdf")

        crystal = docking.load_crystal_ligand(ligand_sdf)
        row["ligand_heavy_atoms"] = crystal.GetNumHeavyAtoms()
        centre, size = ligand_box(crystal)
        row["box_volume_A3"] = float(np.prod(size))

        work = WORK / set_name / name
        work.mkdir(parents=True, exist_ok=True)
        # Pass the crystal ligand so Meeko's `--delete_residues` fallback is
        # checked against the binding site: a complex whose pocket residues had
        # to be deleted is excluded rather than docked and pooled.
        rec = docking.prepare_receptor(protein, work / f"{name}_receptor",
                                       ligand=crystal)
        row["n_deleted_residues"] = len(rec["deleted_residues"])
        row["n_deleted_by_fallback"] = len(rec["deleted_by_fallback"])
        row["deleted_by_fallback"] = ";".join(rec["deleted_by_fallback"])
        row["pocket_residues_deleted"] = ";".join(rec["pocket_residues_deleted"])

        lig_pdbqt = work / f"{name}_ligand.pdbqt"
        docking.prepare_ligand(crystal, lig_pdbqt)

        out = work / f"{name}_vseed{docking.vina_seed(DOCK_SEED)}.pdbqt"
        # Resume: a pose file from an earlier run is reused, but only the Vina
        # search is skipped - receptor preparation above still runs, so the
        # pocket-deletion gate and the per-complex provenance are recomputed
        # rather than inherited from a run whose code may have differed. Vina
        # dominates the per-complex cost, so this recovers most of the time
        # without trusting a stale artefact for anything but coordinates.
        #
        # A leftover that parses but holds no poses must be redocked, not
        # reused: reusing it would raise "Vina produced no poses" on this run
        # and on every rerun afterwards, so the complex could never recover.
        # The stamp guards against reusing coordinates produced under different
        # search settings, which would silently mix two protocols in one table.
        stamp = work / f"{name}_vina_settings.json"
        expected = {"vina_version": docking.VINA_VERSION,
                    "exhaustiveness": docking.EXHAUSTIVENESS,
                    "num_modes": docking.NUM_MODES,
                    "seed": docking.vina_seed(DOCK_SEED),
                    "box_padding": BOX_PADDING}
        row["resumed"] = False
        poses: list[tuple[float, float]] = []
        if out.exists():
            # Compared on the settings keys only, so a stamp may carry extra
            # provenance (e.g. that it was backfilled) without failing the test.
            same_settings = stamp.exists() and all(
                json.loads(stamp.read_text()).get(k) == v
                for k, v in expected.items())
            if not same_settings:
                out.unlink(missing_ok=True)
            else:
                try:
                    poses = docking.score_poses(out, crystal)
                except Exception:
                    poses = []          # truncated or unparsable leftover
                if poses:
                    row["resumed"] = True
                else:
                    out.unlink(missing_ok=True)
        if not out.exists():
            docking.run_vina(rec["pdbqt"], lig_pdbqt, centre, size,
                             docking.vina_seed(DOCK_SEED), out)
            stamp.write_text(json.dumps(expected, indent=2) + "\n",
                             encoding="utf-8")
            poses = docking.score_poses(out, crystal)
        if not poses:
            raise RuntimeError("Vina produced no poses")
        scores = [s for s, _ in poses]
        rmsds = [r for _, r in poses]
        row.update({
            "n_poses": len(poses),
            "top_vina_score": scores[0],
            "top_rmsd": rmsds[0],
            "best_of_poses_rmsd": min(rmsds),
            "best_pose_rank": int(np.argmin(rmsds)) + 1,
            "top_success": bool(rmsds[0] <= docking.SUCCESS_RMSD),
            "best_of_success": bool(min(rmsds) <= docking.SUCCESS_RMSD),
            "pose_file": str(out.relative_to(paths.ROOT)),
        })
    except Exception as exc:  # recorded, never silently dropped
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback_tail"] = traceback.format_exc()[-400:]
    return row


#: The `redock` report's accuracy column. PoseBusters folds "within 2 A" into the
#: same table as the physical checks; keeping it out of the validity conjunction
#: is what lets the report say "accurate but physically invalid", which is the
#: failure mode the PoseBusters paper is actually about.
_RMSD_CHECK_PREFIX = "rmsd"


def run_posebusters(rows: list[dict]) -> pd.DataFrame:
    """PoseBusters ``redock`` battery on every top-ranked pose that exists.

    Run after docking rather than inside ``dock_one`` because PoseBusters loads
    its own RDKit state per call and is far cheaper in one batch.

    ``full_report=False`` is deliberate: it returns exactly the 28 canonical
    columns, so ``pb_valid`` is the conjunction of a fixed, documented check set
    rather than of whatever extra diagnostic columns a full report happens to
    include.
    """
    from posebusters import PoseBusters

    records = []
    buster = PoseBusters(config="redock")
    for row in rows:
        if row.get("error") or "pose_file" not in row:
            continue
        entry = SETS[row["set"] if row["set"] in SETS else "posebusters"] / row["complex"]
        pose = paths.ROOT / row["pose_file"]
        # PoseBusters needs a molecule file it can read; the Vina output is a
        # PDBQT, so the top pose is written back out as SDF next to it.
        sdf = pose.with_name(pose.stem + "_top.sdf")
        if not sdf.exists():
            try:
                from meeko import PDBQTMolecule, RDKitMolCreate

                pmol = PDBQTMolecule(pose.read_text(), is_dlg=False, skip_typing=True)
                mols = RDKitMolCreate.from_pdbqt_mol(pmol)
                top = Chem.RemoveHs(mols[0])
                writer = Chem.SDWriter(str(sdf))
                conf_id = top.GetConformers()[0].GetId()
                writer.write(top, confId=conf_id)
                writer.close()
            except Exception as exc:
                records.append({"complex": row["complex"],
                                "pb_error": f"{type(exc).__name__}: {exc}"})
                continue
        try:
            df = buster.bust([sdf],
                             mol_true=entry / f"{row['complex']}_ligand.sdf",
                             mol_cond=entry / f"{row['complex']}_protein.pdb",
                             full_report=False)
            rec = {"complex": row["complex"], "pb_error": None}
            checks: list[bool] = []
            for col in df.columns:
                val = df.iloc[0][col]
                as_bool = bool(val) if isinstance(val, (bool, np.bool_)) else None
                rec[f"pb_{col}"] = as_bool if as_bool is not None else val
                if as_bool is not None and not col.startswith(_RMSD_CHECK_PREFIX):
                    checks.append(as_bool)
            if not checks:
                raise RuntimeError("no boolean checks returned")
            rec["pb_valid"] = all(checks)
            rec["pb_n_checks"] = len(checks)
            rec["pb_n_failed_checks"] = sum(1 for c in checks if not c)
            records.append(rec)
        except Exception as exc:
            records.append({"complex": row["complex"],
                            "pb_error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(records)


def summarise(df: pd.DataFrame, check_cols: list[str]) -> dict:
    ok = df[df["error"].isna()]
    out: dict = {
        "n_attempted": int(len(df)),
        "n_docked": int(len(ok)),
        "n_failed": int(df["error"].notna().sum()),
        # Stated so a reader can tell a fresh run from one that reused poses
        # computed before the receptor-preparation fix landed.
        "n_resumed_from_existing_poses": (int(df["resumed"].fillna(False).sum())
                                          if "resumed" in df.columns else 0),
    }
    if not len(ok):
        return out
    out.update({
        "top_pose_success_rate": float(ok["top_success"].mean()),
        "best_of_9_success_rate": float(ok["best_of_success"].mean()),
        "median_top_rmsd": float(ok["top_rmsd"].median()),
        "median_best_of_rmsd": float(ok["best_of_poses_rmsd"].median()),
    })
    if "pb_valid" in ok.columns:
        valid = ok["pb_valid"].fillna(False)
        out["physically_valid_rate"] = float(valid.mean())
        out["valid_and_accurate_rate"] = float((valid & ok["top_success"]).mean())
        out["accurate_but_invalid_rate"] = float((~valid & ok["top_success"]).mean())
        failures = {}
        for col in check_cols:
            series = ok[col].dropna()
            if len(series):
                failures[col.replace("pb_", "")] = float(1.0 - series.mean())
        out["per_check_failure_rate"] = dict(
            sorted(failures.items(), key=lambda kv: -kv[1]))
    by_set = {}
    for name, grp in ok.groupby("set"):
        by_set[name] = {
            "n": int(len(grp)),
            "top_pose_success_rate": float(grp["top_success"].mean()),
            "best_of_9_success_rate": float(grp["best_of_success"].mean()),
        }
    out["by_set"] = by_set
    return out


def write_report(df: pd.DataFrame, summary: dict, phase4: dict) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Redocking across unrelated targets")
    add("")
    add("Generated by `src/41_docking_scaleup.py`. AutoDock Vina "
        f"{docking.VINA_VERSION}, exhaustiveness {docking.EXHAUSTIVENESS}, "
        f"{docking.NUM_MODES} modes, `--cpu 1`, project seed {DOCK_SEED}. "
        "Identical settings to the Phase 4 EGFR run.")
    add("")
    add("## 1. Success rate, one target versus many")
    add("")
    add("| set | complexes | top-pose within 2 A | best-of-9 within 2 A |")
    add("|---|---|---|---|")
    add(f"| Phase 4 EGFR (superseded) | {phase4['n']} | "
        f"{phase4['top']:.1%} | {phase4['best_of']:.1%} |")
    for name, block in summary.get("by_set", {}).items():
        add(f"| {name} | {block['n']} | {block['top_pose_success_rate']:.1%} | "
            f"{block['best_of_9_success_rate']:.1%} |")
    add("")
    add(f"Pooled over {summary['n_docked']} complexes that docked successfully "
        f"({summary['n_failed']} failed preparation or docking): "
        f"{summary['top_pose_success_rate']:.1%} top-pose and "
        f"{summary['best_of_9_success_rate']:.1%} best-of-9.")
    add("")

    if "physically_valid_rate" in summary:
        add("## 2. Physical validity")
        add("")
        add(f"PoseBusters passes {summary['physically_valid_rate']:.1%} of "
            "top-ranked poses on all checks. Combining both requirements, "
            f"{summary['valid_and_accurate_rate']:.1%} of top poses are within "
            "2 A *and* physically valid, while "
            f"{summary['accurate_but_invalid_rate']:.1%} are within 2 A but fail "
            "at least one check - a pose in roughly the right place that a "
            "chemist would reject.")
        add("")
        worst = list(summary.get("per_check_failure_rate", {}).items())[:8]
        if worst:
            add("| check | failure rate |")
            add("|---|---|")
            for check, rate in worst:
                add(f"| {check} | {rate:.1%} |")
            add("")

    add("## 3. Scope")
    add("")
    add("This is self-docking: the search box comes from the crystal ligand of "
        "the same structure being redocked, so these rates are an upper bound "
        "on what the tool achieves on a target whose binding site is not already "
        "known. The ligand geometry is rebuilt by ETKDG rather than taken from "
        "the crystal, so the pose itself is not handed to the search.")
    add("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")


def main() -> int:
    args = parse_args()
    names = list(SETS) if args.set == "all" else [args.set]

    entries: list[tuple[Path, str]] = []
    for name in names:
        root = SETS[name]
        if not root.exists():
            raise FileNotFoundError(
                f"{root} missing; run src/40_structure_sets.py first")
        found = sorted(p for p in root.iterdir() if p.is_dir())
        if args.smoke:
            found = found[:6]
        elif args.limit:
            found = found[: args.limit]
        entries += [(p, name) for p in found]

    print(f"Docking {len(entries)} complexes with Vina "
          f"{docking.vina_version().split()[-1]} on {args.workers} workers")
    WORK.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(dock_one, p, s): p.name for p, s in entries}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            flag = "FAIL" if row["error"] else (
                "ok  " if row.get("top_success") else "miss")
            detail = (row["error"][:60] if row["error"]
                      else f"rmsd {row['top_rmsd']:.2f}")
            print(f"[{i}/{len(entries)}] {flag} {row['complex']:<14} {detail}")

    df = pd.DataFrame(rows).sort_values(["set", "complex"]).reset_index(drop=True)
    print(f"\nDocking took {time.time() - t0:.0f}s; "
          f"{int(df['error'].notna().sum())} failures")

    print("Running PoseBusters validity checks...")
    pb = run_posebusters(rows)
    if len(pb):
        df = df.merge(pb, on="complex", how="left")
    # Only the boolean physical checks feed the per-check failure table: the
    # counters and the RMSD column would otherwise be reported as "checks".
    excluded = {"pb_error", "pb_valid", "pb_n_checks", "pb_n_failed_checks"}
    check_cols = [
        c for c in df.columns
        if c.startswith("pb_") and c not in excluded
        and not c.startswith(f"pb_{_RMSD_CHECK_PREFIX}")
        and df[c].dropna().isin([True, False]).all() and df[c].notna().any()
    ]

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(df)} rows, {len(df.columns)} columns)")

    phase4_raw = json.loads((paths.RESULTS / "self_docking.json").read_text())
    overall = phase4_raw.get("overall", phase4_raw)
    phase4 = {
        "n": phase4_raw.get("n_entries_attempted", 20),
        "top": float(overall.get("top_success_rate", 0.41)),
        "best_of": float(overall.get("best_of_success_rate", 0.76)),
    }

    summary = summarise(df, check_cols)
    summary.update({
        "vina_version": docking.vina_version(),
        "exhaustiveness": docking.EXHAUSTIVENESS,
        "num_modes": docking.NUM_MODES,
        "success_rmsd_angstrom": docking.SUCCESS_RMSD,
        "box": f"crystal-ligand bounding box + {BOX_PADDING} A per side",
        "seed": DOCK_SEED,
        "smoke": bool(args.smoke),
        "phase4_egfr_reference": phase4,
        "docking_seconds": round(time.time() - t0, 1),
    })
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")

    write_report(df, summary, phase4)
    if summary.get("n_docked"):
        print(f"\ntop-pose {summary['top_pose_success_rate']:.1%}  "
              f"best-of-9 {summary['best_of_9_success_rate']:.1%}  "
              f"valid {summary.get('physically_valid_rate', float('nan')):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
