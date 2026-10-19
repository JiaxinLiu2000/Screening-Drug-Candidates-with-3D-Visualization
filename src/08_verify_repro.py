"""Verify the pipeline re-runs to identical results, by hash, not by eyeball.

Snapshots the SHA-256 of every committed artefact, re-runs scripts 02-05 with the
current interpreter, and compares. Writes results/reproducibility.json, which
`src/07_report.py` folds into RESULTS.md.

    python src/08_verify_repro.py              # re-run 02,03,05 (fast, ~1 min)
    python src/08_verify_repro.py --with-rf    # also re-run 04 (adds ~4 min)

Run it with a freshly built virtualenv to check the environment too; the
interpreter path and package versions are recorded in the JSON.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import paths  # noqa: E402

SRC = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(files: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(paths.ROOT)): sha256(p) for p in files if p.exists()}


def artefacts(with_rf: bool) -> list[Path]:
    out = sorted(paths.SPLITS.glob("*.csv"))
    out += sorted(paths.RECEPTORS.glob("*"))
    out += [
        paths.RESULTS / "standardization_stats.json",
        paths.RESULTS / "scaffold_stats.json",
        paths.RESULTS / "split_sizes.csv",
        paths.RESULTS / "clean_endpoint_stats.csv",
        paths.CLEAN_PARQUET,
    ]
    if with_rf:
        out += [
            paths.RESULTS / "baseline_metrics.csv",
            paths.RESULTS / "baseline_summary.csv",
        ]
    return out


def run(script: str) -> None:
    print(f"  $ python src/{script}")
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", str(SRC / script)],
        cwd=paths.ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(f"FAILED: src/{script} exited {proc.returncode}")


def versions() -> dict[str, str]:
    import numpy
    import pandas
    import rdkit
    import sklearn

    out = {
        "python": sys.version.split()[0],
        "interpreter": sys.executable,
        "rdkit": rdkit.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
    }
    try:
        import gemmi

        out["gemmi"] = gemmi.__version__
    except Exception:  # noqa: BLE001
        pass
    return out


def main() -> int:
    with_rf = "--with-rf" in sys.argv
    tracked = artefacts(with_rf)
    missing = [p for p in tracked if not p.exists()]
    if missing:
        raise SystemExit(
            "FAILED: run the pipeline before verifying it; missing "
            + ", ".join(str(p.relative_to(paths.ROOT)) for p in missing)
        )

    print("Snapshotting artefact hashes...")
    before = snapshot(tracked)

    scripts = ["02_standardize.py", "03_splits.py"]
    if with_rf:
        scripts.append("04_baseline_rf.py")
    scripts.append("05_egfr_structure.py")

    print(f"Re-running {len(scripts)} scripts with {sys.executable}")
    for script in scripts:
        run(script)

    after = snapshot(tracked)

    identical = sorted(k for k in before if before[k] == after.get(k))
    differing = sorted(k for k in before if before[k] != after.get(k))

    print(f"\nidentical: {len(identical)}/{len(before)}")
    for k in differing:
        print(f"  DIFFERS: {k}")

    def verdict(match: str) -> str:
        keys = [k for k in before if match in k.replace("\\", "/")]
        if not keys:
            return "not checked"
        bad = [k for k in keys if before[k] != after.get(k)]
        noun = "file" if len(keys) == 1 else "files"
        return f"identical ({len(keys)} {noun})" if not bad else f"DIFFERS: {bad}"

    checks = [
        {
            "artefact": "splits/*.csv (10 split index files + scaffolds.csv)",
            "check": "SHA-256 before vs after re-running 02+03",
            "result": verdict("splits/"),
        },
        {
            "artefact": "data/processed/tox21_clean.parquet",
            "check": "SHA-256 before vs after re-running 02",
            "result": verdict("tox21_clean.parquet"),
        },
        {
            "artefact": "results/*.json, *.csv from 02+03",
            "check": "SHA-256 before vs after re-running 02+03",
            "result": verdict("results/"),
        },
        {
            "artefact": "receptors/* (cleaned receptors, ligand SDFs, pocket JSON)",
            "check": "SHA-256 before vs after re-running 05",
            "result": verdict("receptors/"),
        },
    ]
    if with_rf:
        checks.append(
            {
                "artefact": "results/baseline_metrics.csv + baseline_summary.csv",
                "check": "SHA-256 before vs after re-running 04 (120 forest fits)",
                "result": verdict("baseline_"),
            }
        )

    payload = {
        "date": date.today().isoformat(),
        "scripts_rerun": scripts,
        "environment": versions(),
        "n_artefacts": len(before),
        "n_identical": len(identical),
        "differing": differing,
        "checks": checks,
        "hashes": after,
    }
    out = paths.RESULTS / "reproducibility.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    return 1 if differing else 0


if __name__ == "__main__":
    raise SystemExit(main())
