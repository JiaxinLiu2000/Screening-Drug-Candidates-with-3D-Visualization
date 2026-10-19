"""Canonical project paths and global constants (single source of truth)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "splits"
RECEPTORS = ROOT / "receptors"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

WIDE_PARQUET = DATA_PROCESSED / "tox21_wide.parquet"
CLEAN_PARQUET = DATA_PROCESSED / "tox21_clean.parquet"

# Seed used for anything that is not itself swept over seeds.
GLOBAL_SEED = 0
# Seeds swept in the baseline experiment (controls split *and* forest).
SEEDS = (0, 1, 2, 3, 4)


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, SPLITS, RECEPTORS, RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
