"""Task 2 - standardize, deduplicate and filter the merged Tox21 table.

All chemistry lives in ``tox21lib.standardize``; this script is only I/O +
reporting. Output: data/processed/tox21_clean.parquet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import paths, standardize  # noqa: E402


def main() -> int:
    paths.ensure_dirs()
    if not paths.WIDE_PARQUET.exists():
        print(f"FAILED: {paths.WIDE_PARQUET} missing - run src/01_download_tox21.py first")
        return 1

    wide = pd.read_parquet(paths.WIDE_PARQUET)
    label_cols = [c for c in wide.columns if c not in ("drug_id", "smiles")]
    print(f"Input: {wide.shape} with {len(label_cols)} label columns")

    clean, stats = standardize.standardize_table(wide, label_cols)

    print("\n--- standardization report ---")
    print(f"input rows                     : {stats['input_rows']}")
    print(f"dropped, failed to parse       : {stats['parse_failures']}"
          f"  {stats['parse_failure_breakdown'] or ''}")
    print(f"rows entering dedup            : {stats['rows_before_dedup']}")
    print(f"unique InChIKeys               : {stats['unique_inchikeys']}")
    print(f"duplicate rows merged          : {stats['duplicate_rows_merged']}")
    print(f"label cells NaN'd by tied vote : {stats['label_cells_nan_from_tie']}")
    print(f"removed by heavy-atom/element  : {stats['filtered_out']}"
          f"  {stats['filter_breakdown'] or ''}")
    print(f"FINAL molecules                : {stats['final_molecules']}")

    print("\nPer-endpoint labels after cleaning:")
    rows = []
    for col in label_cols:
        tested = int(clean[col].notna().sum())
        pos = int((clean[col] == 1).sum())
        rows.append(
            {
                "endpoint": col,
                "n_tested": tested,
                "n_positive": pos,
                "positive_rate": round(pos / tested, 4) if tested else float("nan"),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    clean.to_parquet(paths.CLEAN_PARQUET, index=False)
    (paths.RESULTS / "standardization_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(paths.RESULTS / "clean_endpoint_stats.csv", index=False)
    print(f"\nWrote {paths.CLEAN_PARQUET}  shape={clean.shape}")
    print(f"Wrote {paths.RESULTS / 'standardization_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
