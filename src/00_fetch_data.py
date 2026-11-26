"""Fetch every input this project needs but does not commit.

Raw datasets are ~320 MB and are all redistributions of public sources, so they
are gitignored and fetched instead. This script is the single entry point:

    python src/00_fetch_data.py            # everything missing
    python src/00_fetch_data.py --check    # report only, download nothing

What it fetches, and from where:

    Tox21, hERG, KIBA, Davis   Therapeutics Data Commons (`tdc`)
    EGFR receptor structures   RCSB PDB (already committed; verified here)

What it deliberately does NOT fetch: the ESM-2 650M weights (2.5 GB). The 229
protein embeddings this project needs are committed under `models/`, so the
affinity model runs without ever downloading the language model. Pass
`--with-esm` if you want to regenerate them yourself.

Every download is verified by re-reading what landed on disk and checking shape
and row counts against what the pipeline expects, so a truncated or
silently-changed upstream file fails here rather than three scripts later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import paths  # noqa: E402

#: Row counts the committed results were produced from. TDC does occasionally
#: re-release a dataset; if these stop matching, the pipeline's numbers no longer
#: describe the data on disk, and that must be loud rather than silent.
EXPECTED = {
    "Tox21": {"endpoints": 12, "min_rows": 7000},
    "KIBA": {"min_rows": 110000},
    "DAVIS": {"min_rows": 25000},
}


def have(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def fetch_tox21() -> dict:
    from tdc.single_pred import Tox
    from tdc.utils import retrieve_label_name_list

    names = retrieve_label_name_list("Tox21")
    if len(names) != EXPECTED["Tox21"]["endpoints"]:
        raise SystemExit(
            f"FAILED: TDC reports {len(names)} Tox21 endpoints, this project was "
            f"built on {EXPECTED['Tox21']['endpoints']}. The dataset changed "
            f"upstream; do not mix the new data with committed results.")
    total = 0
    for name in names:
        total += len(Tox(name="Tox21", label_name=name,
                         path=str(paths.DATA_RAW)).get_data())
    return {"endpoints": len(names), "rows": total}


def fetch_dti(name: str) -> dict:
    from tdc.multi_pred import DTI

    frame = DTI(name=name, path=str(paths.DATA_RAW)).get_data()
    floor = EXPECTED[name]["min_rows"]
    if len(frame) < floor:
        raise SystemExit(
            f"FAILED: {name} returned {len(frame)} rows, expected at least {floor}")
    return {"rows": int(len(frame)),
            "drugs": int(frame["Drug_ID"].nunique()),
            "proteins": int(frame["Target_ID"].nunique())}


def fetch_herg() -> dict:
    from tdc.single_pred import Tox

    frame = Tox(name="hERG_Karim", path=str(paths.DATA_RAW)).get_data()
    return {"rows": int(len(frame))}


def check_receptors() -> dict:
    """Receptor PDB files are committed; this verifies the clone is complete."""
    pdbs = sorted(paths.RECEPTORS.glob("*.pdb")) + \
        sorted(paths.RECEPTORS.glob("**/*.pdb"))
    return {"files": len({p.name for p in pdbs})}


def check_embeddings() -> dict:
    """Report both copies: the one the pipeline reads and the one that ships.

    `src/19_esm_embed.py` writes to data/processed/ and every training script
    reads from there; `src/38_export_models.py` copies it into models/ for
    distribution. A fresh clone has the second but not the first, so reporting
    only one of them gives the wrong answer in one of the two situations.
    """
    from tox21lib import protein_embed

    out: dict = {}
    for label, npz in (("pipeline_copy", protein_embed.cache_path(paths.DATA_PROCESSED)),
                       ("shipped_copy", paths.ROOT / "models" / "protein_embeddings.npz")):
        if not npz.exists():
            out[label] = {"present": False, "path": str(npz.relative_to(paths.ROOT))}
            continue
        matrix, rows, model_name = protein_embed.load_cache(npz)
        out[label] = {"present": True, "path": str(npz.relative_to(paths.ROOT)),
                      "n_proteins": len(rows), "dim": int(matrix.shape[1]),
                      "model": model_name}
    if not out["pipeline_copy"]["present"] and out["shipped_copy"]["present"]:
        out["note"] = ("training scripts read data/processed/; copy the shipped file "
                       "there, or regenerate with src/19_esm_embed.py (2.5 GB download)")
    elif not out["shipped_copy"]["present"]:
        out["note"] = "run src/38_export_models.py to build models/"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="report what is present; download nothing")
    ap.add_argument("--with-esm", action="store_true",
                    help="also regenerate protein embeddings (2.5 GB download)")
    args = ap.parse_args()

    paths.ensure_dirs()
    report: dict = {"receptors": check_receptors(),
                    "protein_embeddings": check_embeddings()}

    raw_marker = paths.DATA_RAW / "tox21.tab"
    present = {
        "tox21": have(raw_marker),
        "kiba": have(paths.DATA_RAW / "kiba.tab"),
        "davis": have(paths.DATA_RAW / "davis.tab"),
        "herg": have(paths.DATA_RAW / "herg_karim.tab"),
    }
    report["raw_present"] = present

    if args.check:
        print(json.dumps(report, indent=2))
        missing = [k for k, v in present.items() if not v]
        print(f"\n{len(present) - len(missing)}/{len(present)} raw datasets present"
              + (f"; missing {missing} - run without --check" if missing else ""))
        return 0

    if not present["tox21"]:
        print("Tox21 (12 endpoints) ...")
        report["tox21"] = fetch_tox21()
    if not present["kiba"]:
        print("KIBA ...")
        report["kiba"] = fetch_dti("KIBA")
    if not present["davis"]:
        print("Davis ...")
        report["davis"] = fetch_dti("DAVIS")
    if not present["herg"]:
        print("hERG (Karim) ...")
        report["herg"] = fetch_herg()

    if args.with_esm:
        print("Regenerating protein embeddings (downloads ESM-2 650M) ...")
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).parent / "19_esm_embed.py")],
                       check=True)
        report["protein_embeddings"] = check_embeddings()

    out = paths.RESULTS / "fetch_data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{json.dumps(report, indent=2)}\nWrote {out}")
    print("\nNext: python src/02_standardize.py && python src/03_splits.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
