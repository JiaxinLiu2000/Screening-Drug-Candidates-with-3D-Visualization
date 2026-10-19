"""Task 3 - build the random and scaffold splits ONCE on the deduplicated set.

Index files are written to splits/ (committed) so every later stage replays the
exact same partition without re-running RDKit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import paths, splits as splitlib  # noqa: E402


def main() -> int:
    paths.ensure_dirs()
    if not paths.CLEAN_PARQUET.exists():
        print(f"FAILED: {paths.CLEAN_PARQUET} missing - run src/02_standardize.py first")
        return 1

    clean = pd.read_parquet(paths.CLEAN_PARQUET)
    print(f"Deduplicated molecule set: {len(clean)} molecules")

    scaffolds = pd.Series(
        [splitlib.scaffold_smiles(s) for s in clean["canonical_smiles"]],
        name="scaffold_smiles",
    )
    stats = splitlib.scaffold_stats(scaffolds)
    print("\n--- Bemis-Murcko scaffolds ---")
    print(f"unique scaffolds        : {stats['n_unique_scaffolds']}")
    print(f"largest group size      : {stats['largest_group_size']}"
          f"  (scaffold: {stats['largest_group_scaffold'] or '<acyclic / empty scaffold>'})")
    print(f"singleton scaffolds     : {stats['n_singleton_scaffolds']}")
    print(f"acyclic molecules       : {stats['n_acyclic_molecules']}")

    scaffolds.to_frame().assign(inchikey=clean["inchikey"]).to_csv(
        paths.SPLITS / "scaffolds.csv", index=False
    )

    print("\n--- split sizes ---")
    rows = []
    for kind in ("random", "scaffold"):
        for seed in paths.SEEDS:
            frame = splitlib.split_frame(clean, kind, seed, scaffolds)
            out = paths.SPLITS / f"{kind}_split_seed{seed}.csv"
            frame.to_csv(out, index=False)
            counts = frame["subset"].value_counts()
            leak = 0
            if kind == "scaffold":
                per_scaffold = frame.groupby("scaffold_smiles")["subset"].nunique()
                leak = int((per_scaffold > 1).sum())
            rows.append(
                {
                    "split": kind,
                    "seed": seed,
                    "train": int(counts.get("train", 0)),
                    "valid": int(counts.get("valid", 0)),
                    "test": int(counts.get("test", 0)),
                    "scaffold_groups_spanning_subsets": leak,
                }
            )
    sizes = pd.DataFrame(rows)
    print(sizes.to_string(index=False))

    if (sizes["scaffold_groups_spanning_subsets"] != 0).any():
        print("FAILED: a scaffold group was split across subsets")
        return 1

    sizes.to_csv(paths.RESULTS / "split_sizes.csv", index=False)
    (paths.RESULTS / "scaffold_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {len(rows)} index files to {paths.SPLITS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
