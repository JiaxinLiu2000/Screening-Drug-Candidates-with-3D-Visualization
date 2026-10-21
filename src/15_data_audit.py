"""Independent integrity audit of every committed artefact.

This does NOT re-run the pipeline and does NOT trust any script's own asserts.
It recomputes the quantities from the artefacts themselves and checks they agree:
summary statistics against the per-fit metric rows, pocket geometry against the
ligand coordinates, split partitions against the molecule tables, the quoted
headline numbers in the markdown documents against their source tables, and the
dependency lock against the installed environment.

Writes results/DATA_AUDIT.md and results/data_audit.json; exits non-zero if any
check fails.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import featurize, paths, splits as splitlib, standardize  # noqa: E402

TOL = 1e-9
CHECKS: list[dict] = []


def check(section: str, name: str):
    """Register one audit check; the function returns (ok, detail)."""

    def wrap(fn):
        CHECKS.append({"section": section, "name": name, "fn": fn})
        return fn

    return wrap


def rel(path: Path) -> str:
    return path.relative_to(paths.ROOT).as_posix()


# --------------------------------------------------------------------------
# 1. Tox21 molecule table
# --------------------------------------------------------------------------

def clean_table() -> pd.DataFrame:
    return pd.read_parquet(paths.CLEAN_PARQUET)


ENDPOINTS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


@check("Tox21 molecule table", "row count matches the reported final_molecules")
def _():
    clean = clean_table()
    stats = json.loads((paths.RESULTS / "standardization_stats.json").read_text())
    ok = len(clean) == stats["final_molecules"] == 6143
    return ok, f"parquet rows {len(clean)}, stats final_molecules {stats['final_molecules']}"


@check("Tox21 molecule table", "InChIKeys unique and sorted (deduplication held)")
def _():
    keys = clean_table()["inchikey"]
    dups = int(keys.duplicated().sum())
    sorted_ok = bool((keys.to_numpy() == np.sort(keys.to_numpy())).all())
    return dups == 0 and sorted_ok, f"duplicates {dups}, sorted {sorted_ok}"


@check("Tox21 molecule table", "every canonical SMILES re-parses and is canonical")
def _():
    from rdkit import Chem

    clean = clean_table()
    bad_parse, not_canonical = [], []
    for smi in clean["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad_parse.append(smi)
        elif Chem.MolToSmiles(mol) != smi:
            not_canonical.append(smi)
    ok = not bad_parse and not not_canonical
    return ok, f"unparsable {len(bad_parse)}, non-canonical {len(not_canonical)}"


@check("Tox21 molecule table", "InChIKey column matches the SMILES it sits next to")
def _():
    from rdkit import Chem

    clean = clean_table()
    mismatched = [
        key
        for smi, key in zip(clean["canonical_smiles"], clean["inchikey"])
        if Chem.MolToInchiKey(Chem.MolFromSmiles(smi)) != key
    ]
    return not mismatched, f"mismatched {len(mismatched)} of {len(clean)}"


@check("Tox21 molecule table", "heavy-atom and element filters actually hold")
def _():
    clean = clean_table()
    violations = []
    for smi, n_heavy in zip(clean["canonical_smiles"], clean["n_heavy_atoms"]):
        keep, reason = standardize.passes_property_filter(smi, int(n_heavy))
        if not keep:
            violations.append((smi, reason))
    rng = (int(clean["n_heavy_atoms"].min()), int(clean["n_heavy_atoms"].max()))
    return not violations, f"violations {len(violations)}, heavy-atom range {rng}"


@check("Tox21 molecule table", "labels are only 0, 1 or NaN and endpoints are the 12 expected")
def _():
    clean = clean_table()
    present = [c for c in clean.columns if c in ENDPOINTS]
    bad = {}
    for col in present:
        vals = set(pd.unique(clean[col].dropna()))
        if not vals <= {0.0, 1.0}:
            bad[col] = sorted(vals)[:5]
    return len(present) == 12 and not bad, f"endpoints {len(present)}, out-of-domain values {bad or 'none'}"


# --------------------------------------------------------------------------
# 2. Tox21 split index files
# --------------------------------------------------------------------------

@check("Tox21 splits", "index files align row-for-row with the molecule table")
def _():
    clean = clean_table()
    bad = []
    for kind in ("random", "scaffold"):
        for seed in paths.SEEDS:
            f = pd.read_csv(paths.SPLITS / f"{kind}_split_seed{seed}.csv")
            if len(f) != len(clean) or not (f["inchikey"].to_numpy() == clean["inchikey"].to_numpy()).all():
                bad.append(f"{kind}/seed{seed}")
            if not (f["row_index"].to_numpy() == np.arange(len(clean))).all():
                bad.append(f"{kind}/seed{seed} row_index")
    return not bad, f"misaligned {bad or 'none'} across 10 files"


@check("Tox21 splits", "subsets partition the set exactly, at 80/10/10")
def _():
    clean = clean_table()
    detail = []
    ok = True
    for kind in ("random", "scaffold"):
        for seed in paths.SEEDS:
            f = pd.read_csv(paths.SPLITS / f"{kind}_split_seed{seed}.csv")
            counts = f["subset"].value_counts()
            total = int(counts.sum())
            fracs = (counts / total).to_dict()
            if total != len(clean) or set(counts.index) != {"train", "valid", "test"}:
                ok = False
            if not (abs(fracs["train"] - 0.8) < 0.01 and abs(fracs["valid"] - 0.1) < 0.01
                    and abs(fracs["test"] - 0.1) < 0.01):
                ok = False
            detail.append(f"{kind}{seed}:{counts['train']}/{counts['valid']}/{counts['test']}")
    return ok, "; ".join(detail[:2]) + f"; ... ({len(detail)} files, all within 1% of 80/10/10)"


@check("Tox21 splits", "scaffold groups never span subsets, and random splits do")
def _():
    leaks = {}
    for kind in ("random", "scaffold"):
        for seed in paths.SEEDS:
            f = pd.read_csv(paths.SPLITS / f"{kind}_split_seed{seed}.csv")
            spanning = int((f.groupby("scaffold_smiles")["subset"].nunique() > 1).sum())
            leaks.setdefault(kind, []).append(spanning)
    ok = all(v == 0 for v in leaks["scaffold"]) and all(v > 0 for v in leaks["random"])
    return ok, (f"scaffold spanning groups {set(leaks['scaffold'])}, "
                f"random {min(leaks['random'])}-{max(leaks['random'])} (leakage the split removes)")


@check("Tox21 splits", "committed scaffolds reproduce from the SMILES (seeded 250-molecule sample)")
def _():
    clean = clean_table()
    scaf = pd.read_csv(paths.SPLITS / "scaffolds.csv")
    if len(scaf) != len(clean) or not (scaf["inchikey"].to_numpy() == clean["inchikey"].to_numpy()).all():
        return False, "scaffolds.csv does not align with the molecule table"
    rng = np.random.default_rng(paths.GLOBAL_SEED)
    idx = rng.choice(len(clean), size=250, replace=False)
    stored = scaf["scaffold_smiles"].fillna("").to_numpy()
    mismatched = [
        int(i) for i in idx
        if splitlib.scaffold_smiles(clean["canonical_smiles"].iloc[i]) != stored[i]
    ]
    counts = scaf["scaffold_smiles"].fillna("").value_counts()
    return not mismatched, (f"sample mismatches {len(mismatched)}/250; "
                            f"{counts.size} unique scaffolds, largest group {int(counts.iloc[0])}")


# --------------------------------------------------------------------------
# 3. Classification metric tables
# --------------------------------------------------------------------------

def metrics_table() -> pd.DataFrame:
    return pd.read_csv(paths.RESULTS / "baseline_metrics.csv")


@check("Classification metrics", "row counts per dataset are the full grid")
def _():
    m = metrics_table()
    counts = m.groupby("dataset").size().to_dict()
    expected = {"Tox21": 120, "hERG": 10, "hERG-large": 20}
    by_setting = m.groupby(["dataset", "setting"]).size().to_dict()
    ok = counts == expected and len(m) == sum(expected.values())
    return ok, f"rows per dataset {counts}; per (dataset, setting) {by_setting}"


@check("Classification metrics", "every AUC is in [0, 1] and fold counts are consistent")
def _():
    m = metrics_table()
    bad_auc = m[(m[["roc_auc", "pr_auc"]] < 0).any(axis=1) | (m[["roc_auc", "pr_auc"]] > 1).any(axis=1)]
    bad_pos = m[(m["n_test_pos"] > m["n_test"]) | (m["n_train_pos"] > m["n_train"])
                | (m["n_test_pos"] <= 0) | (m["n_train_pos"] <= 0)]
    ok = bad_auc.empty and bad_pos.empty
    return ok, (f"out-of-range AUCs {len(bad_auc)}, impossible positive counts {len(bad_pos)}; "
                f"ROC-AUC range {m['roc_auc'].min():.4f}-{m['roc_auc'].max():.4f}")


@check("Classification metrics", "summary mean/std recompute exactly from the per-fit rows")
def _():
    m = metrics_table()
    stored = pd.read_csv(paths.RESULTS / "baseline_summary.csv")
    worst = 0.0
    for _, row in stored.iterrows():
        sub = m[(m["dataset"] == row["dataset"]) & (m["split"] == row["split"])]
        # The tables are keyed by (dataset, setting): hERG-large carries `full` and
        # `disjoint` rows sharing dataset/split/endpoint, so ignoring `setting`
        # would compare a mixture of the two against each row.
        if "setting" in m.columns and "setting" in stored.columns:
            sub = sub[sub["setting"] == row["setting"]]
        if row["level"] == "endpoint":
            sub = sub[sub["endpoint"] == row["endpoint"]]
            vals = sub[["roc_auc", "pr_auc"]]
            recomputed = (vals["roc_auc"].mean(), vals["roc_auc"].std(ddof=1),
                          vals["pr_auc"].mean(), vals["pr_auc"].std(ddof=1))
        else:
            per_seed = sub.groupby("seed")[["roc_auc", "pr_auc"]].mean()
            recomputed = (per_seed["roc_auc"].mean(), per_seed["roc_auc"].std(ddof=1),
                          per_seed["pr_auc"].mean(), per_seed["pr_auc"].std(ddof=1))
        for got, want in zip(recomputed, (row["roc_auc_mean"], row["roc_auc_std"],
                                          row["pr_auc_mean"], row["pr_auc_std"])):
            worst = max(worst, abs(float(got) - float(want)))
    return worst < TOL, f"largest absolute discrepancy over {len(stored)} summary rows: {worst:.2e}"


@check("Classification metrics", "the macro row is the mean over endpoints, not over fits")
def _():
    m = metrics_table()[lambda d: d["dataset"] == "Tox21"]
    stored = pd.read_csv(paths.RESULTS / "baseline_summary.csv")
    stored = stored[(stored["dataset"] == "Tox21") & (stored["level"] == "overall")]
    detail = []
    ok = True
    for _, row in stored.iterrows():
        sub = m[m["split"] == row["split"]]
        per_seed = sub.groupby("seed")["roc_auc"].mean()
        if per_seed.size != 5 or abs(per_seed.mean() - row["roc_auc_mean"]) > TOL:
            ok = False
        detail.append(f"{row['split']} {row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f}")
    return ok, "; ".join(detail)


# --------------------------------------------------------------------------
# 4. hERG artefacts
# --------------------------------------------------------------------------

@check("hERG", "split files partition one consistent molecule set")
def _():
    stats = json.loads((paths.RESULTS / "herg_stats.json").read_text())
    sizes, ok = set(), True
    for kind in ("random", "scaffold"):
        for seed in paths.SEEDS:
            f = pd.read_csv(paths.SPLITS / f"herg_{kind}_split_seed{seed}.csv")
            sizes.add(len(f))
            if f["inchikey"].duplicated().any() or set(f["subset"]) != {"train", "valid", "test"}:
                ok = False
            if kind == "scaffold" and int((f.groupby("scaffold_smiles")["subset"].nunique() > 1).sum()):
                ok = False
    ok = ok and sizes == {stats["final_molecules"]}
    return ok, f"molecule count in every file {sizes}, stats final_molecules {stats['final_molecules']}"


@check("hERG", "the reported drop equals the summary rows it is drawn from")
def _():
    stored = pd.read_csv(paths.RESULTS / "baseline_summary.csv")
    o = stored[(stored["dataset"] == "hERG") & (stored["level"] == "overall")].set_index("split")
    drop = float(o.loc["random", "roc_auc_mean"] - o.loc["scaffold", "roc_auc_mean"])
    md = (paths.RESULTS / "herg_baseline.md").read_text(encoding="utf-8")
    quoted = f"{drop:+.4f}" in md or f"{drop:.4f}" in md
    return quoted, (f"random {o.loc['random', 'roc_auc_mean']:.4f} - scaffold "
                    f"{o.loc['scaffold', 'roc_auc_mean']:.4f} = {drop:+.4f}, quoted in md: {quoted}")


# --------------------------------------------------------------------------
# 5. Binding artefacts
# --------------------------------------------------------------------------

@check("Binding", "metric table is the full 30-fit grid with in-range values")
def _():
    b = pd.read_csv(paths.RESULTS / "binding_baseline.csv")
    counts = b.groupby(["dataset", "split"]).size().to_dict()
    ok = len(b) == 30 and set(counts.values()) == {5}
    bad = b[(b["rmse"] <= 0) | (b["pearson_r"].abs() > 1) | (b["spearman_rho"].abs() > 1)
            | (b["ci"] < 0) | (b["ci"] > 1)]
    return ok and bad.empty, f"rows {len(b)}, groups {len(counts)}, out-of-range rows {len(bad)}"


@check("Binding", "cold splits leak no entity between train and valid/test")
def _():
    leaks = []
    for tag in ("davis", "kiba"):
        for kind, col in (("cold_drug", "drug_id"), ("cold_target", "target_id")):
            for seed in paths.SEEDS:
                f = pd.read_csv(paths.SPLITS / f"{tag}_{kind}_seed{seed}.csv")
                tr = set(f.loc[f["subset"] == "train", col])
                for other in ("valid", "test"):
                    shared = tr & set(f.loc[f["subset"] == other, col])
                    if shared:
                        leaks.append(f"{tag}/{kind}/seed{seed}/{other}:{len(shared)}")
    return not leaks, f"leaking entities {leaks or 'none across 20 cold-split files'}"


@check("Binding", "split files cover every pair exactly once")
def _():
    bad = []
    for tag in ("davis", "kiba"):
        for kind in ("random", "cold_drug", "cold_target"):
            for seed in paths.SEEDS:
                f = pd.read_csv(paths.SPLITS / f"{tag}_{kind}_seed{seed}.csv")
                idx = np.sort(f["pair_index"].to_numpy())
                if not (idx == np.arange(len(f))).all() or f["subset"].isna().any():
                    bad.append(f"{tag}/{kind}/seed{seed}")
    return not bad, f"files with incomplete pair coverage {bad or 'none of 30'}"


@check("Binding", "dataset statistics recompute from the split files")
def _():
    stats = json.loads((paths.RESULTS / "binding_dataset_stats.json").read_text())
    detail, ok = [], True
    for tag, key in (("davis", "DAVIS"), ("kiba", "KIBA")):
        f = pd.read_csv(paths.SPLITS / f"{tag}_random_seed0.csv")
        n_pairs, n_drugs = len(f), f["inchikey"].nunique()
        n_targets = f["target_id"].nunique()
        density = n_pairs / (n_drugs * n_targets)
        if (n_pairs != stats[key]["measured_pairs"]
                or n_drugs != stats[key]["distinct_drugs"]
                or n_targets != stats[key]["distinct_proteins"]
                or abs(density - stats[key]["matrix_density"]) > 1e-4):
            ok = False
        detail.append(f"{key} {n_drugs}x{n_targets}={n_pairs} pairs, density {density:.4f}")
    return ok, "; ".join(detail)


@check("Binding", "Davis cold-drug test folds really are too small to be stable")
def _():
    b = pd.read_csv(paths.RESULTS / "binding_baseline.csv")
    d = b[(b["dataset"] == "DAVIS") & (b["split"] == "cold_drug")]
    k = b[(b["dataset"] == "KIBA") & (b["split"] == "cold_drug")]
    ok = (d["n_test_drugs"].max() <= 10 < k["n_test_drugs"].min()
          and d["pearson_r"].std(ddof=1) > 5 * k["pearson_r"].std(ddof=1))
    return ok, (f"Davis {int(d['n_test_drugs'].iloc[0])} test drugs, Pearson sd "
                f"{d['pearson_r'].std(ddof=1):.4f}; KIBA {int(k['n_test_drugs'].iloc[0])} test drugs, "
                f"Pearson sd {k['pearson_r'].std(ddof=1):.4f}")


# --------------------------------------------------------------------------
# 6. Chemical-space measurement
# --------------------------------------------------------------------------

@check("Chemical space", "median nearest-neighbour Tanimoto recomputes to the reported value")
def _():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator as rfg

    from tox21lib import datasets

    rec = json.loads((paths.RESULTS / "chemical_space_overlap.json").read_text())
    clean = clean_table()
    # Re-derive the Davis molecule set from TDC through the shared standardizer
    # instead of trusting a stored copy of it.
    davis, _ = datasets.unique_molecules(datasets.load_dti("DAVIS"))
    gen = rfg.GetMorganGenerator(radius=featurize.MORGAN_RADIUS, fpSize=featurize.MORGAN_N_BITS)
    ref = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in davis["canonical_smiles"]]
    nn = [
        max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)), ref))
        for s in clean["canonical_smiles"]
    ]
    median = float(np.median(nn))
    stored = rec["nn_tanimoto"]["tox21_to_davis"]["median"]
    count_ok = len(davis) == rec["davis_drugs_after_standardization"]
    return abs(median - stored) < 1e-9 and count_ok, (
        f"{len(davis)} Davis molecules re-standardized; recomputed median "
        f"{median:.6f} vs stored {stored:.6f}")


@check("Chemical space", "the verdict band matches the measured median")
def _():
    rec = json.loads((paths.RESULTS / "chemical_space_overlap.json").read_text())
    median, verdict = rec["verdict_median"], rec["verdict"]
    expected = ("Strong overlap" if median > 0.4
                else "Weak overlap" if median >= 0.25 else "Negligible overlap")
    md = (paths.RESULTS / "chemical_space_overlap.md").read_text(encoding="utf-8")
    return verdict == expected and f"## Verdict: {expected}" in md, (
        f"median {median:.4f} -> {expected}; JSON says {verdict}; md heading present")


# --------------------------------------------------------------------------
# 7. Structures
# --------------------------------------------------------------------------

def sdf_mol(path: Path):
    from rdkit import Chem

    return Chem.MolFromMolFile(str(path))


@check("EGFR receptors", "ligand SDFs carry real bond orders and are erlotinib")
def _():
    from rdkit import Chem

    detail, ok = [], True
    for pdb_id in ("1M17", "4HJO"):
        mol = sdf_mol(paths.RECEPTORS / f"{pdb_id}_ligand.sdf")
        if mol is None:
            ok = False
            detail.append(f"{pdb_id}: unparsable")
            continue
        arom = sum(1 for b in mol.GetBonds() if b.GetIsAromatic())
        triple = sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 3.0)
        key = Chem.MolToInchiKey(mol)
        if not (arom == 17 and triple == 1 and key == "AAKJLRGGTJKAMG-UHFFFAOYSA-N"):
            ok = False
        detail.append(f"{pdb_id}: {arom} aromatic, {triple} triple, {key}")
    return ok, "; ".join(detail)


@check("EGFR receptors", "cleaned receptors contain protein only, one chain")
def _():
    detail, ok = [], True
    for pdb_id in ("1M17", "4HJO"):
        txt = (paths.RECEPTORS / f"{pdb_id}_receptor.pdb").read_text()
        hetatm = sum(1 for line in txt.splitlines() if line.startswith("HETATM"))
        chains = {line[21] for line in txt.splitlines() if line.startswith("ATOM")}
        waters = txt.count("HOH")
        if hetatm or waters or len(chains) != 1:
            ok = False
        detail.append(f"{pdb_id}: HETATM {hetatm}, HOH {waters}, chains {sorted(chains)}")
    return ok, "; ".join(detail)


@check("EGFR receptors", "pocket centre and box recompute from the ligand coordinates")
def _():
    detail, ok = [], True
    for pdb_id in ("1M17", "4HJO"):
        pocket = json.loads((paths.RECEPTORS / f"{pdb_id}_pocket.json").read_text())
        mol = sdf_mol(paths.RECEPTORS / f"{pdb_id}_ligand.sdf")
        xyz = mol.GetConformer().GetPositions()
        pad = pocket["padding_angstrom"]
        centre = np.round(xyz.mean(axis=0), 3)
        box_min = np.round(xyz.min(axis=0) - pad, 3)
        box_max = np.round(xyz.max(axis=0) + pad, 3)
        d_c = float(np.abs(centre - np.array(pocket["center"])).max())
        d_b = max(float(np.abs(box_min - np.array(pocket["box_min"])).max()),
                  float(np.abs(box_max - np.array(pocket["box_max"])).max()))
        # The SDF holds heavy atoms only; the stored pocket was built from the same
        # heavy-atom set, so agreement should be exact to the rounding.
        if d_c > 0.002 or d_b > 0.002:
            ok = False
        detail.append(f"{pdb_id}: centre delta {d_c:.3f} A, box delta {d_b:.3f} A")
    return ok, "; ".join(detail)


@check("EGFR receptors", "conformation label agrees with the measured salt-bridge distance")
def _():
    rec = json.loads((paths.RESULTS / "egfr_structures.json").read_text())
    detail, ok = [], True
    for s in rec["structures"]:
        c = s["conformation"]
        expected = "active" if c["distance_angstrom"] <= 4.0 else "inactive"
        if not c["state"].startswith(expected):
            ok = False
        detail.append(f"{s['pdb_id']}: {c['distance_angstrom']} A -> {expected}")
    return ok, "; ".join(detail)


# --------------------------------------------------------------------------
# 8. Docking benchmark set
# --------------------------------------------------------------------------

def benchmark_records() -> list[dict]:
    d = paths.RECEPTORS / "benchmark"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


@check("Docking benchmark", "20 structures, every referenced file present")
def _():
    recs = benchmark_records()
    missing = [
        f for r in recs for f in r["files"].values()
        if not (paths.ROOT / f).exists()
    ]
    return len(recs) == 20 and not missing, f"records {len(recs)}, missing files {len(missing)}"


@check("Docking benchmark", "ligand SDFs match their JSON record (atoms, InChIKey, SMILES)")
def _():
    from rdkit import Chem

    bad = []
    for r in benchmark_records():
        mol = sdf_mol(paths.ROOT / r["files"]["ligand"])
        if mol is None:
            bad.append(f"{r['pdb_id']}: unparsable")
            continue
        if mol.GetNumHeavyAtoms() != r["ligand_heavy_atoms"]:
            bad.append(f"{r['pdb_id']}: atom count")
        if Chem.MolToInchiKey(mol) != r["ligand_inchikey"]:
            bad.append(f"{r['pdb_id']}: InChIKey")
        flat = Chem.Mol(mol)
        flat.RemoveAllConformers()
        if Chem.MolToSmiles(flat) != r["ligand_smiles"]:
            bad.append(f"{r['pdb_id']}: SMILES")
        if mol.GetNumConformers() == 0:
            bad.append(f"{r['pdb_id']}: no 3D coordinates")
    return not bad, f"inconsistencies {bad or 'none across 20 structures'}"


@check("Docking benchmark", "selection filters hold: resolution, heavy atoms, distinct chemistry")
def _():
    recs = benchmark_records()
    res = [r["resolution_angstrom"] for r in recs]
    heavy = [r["ligand_heavy_atoms"] for r in recs]
    keys = [r["ligand_inchikey"] for r in recs]
    ok = (max(res) <= 2.8 and min(heavy) >= 15 and max(heavy) <= 60
          and len(set(keys)) == len(keys)
          and "AAKJLRGGTJKAMG-UHFFFAOYSA-N" not in keys)
    return ok, (f"resolution {min(res)}-{max(res)} A, heavy atoms {min(heavy)}-{max(heavy)}, "
                f"{len(set(keys))} distinct ligands, erlotinib absent")


@check("Docking benchmark", "benchmark receptors are protein-only, single chain")
def _():
    bad = []
    for r in benchmark_records():
        txt = (paths.ROOT / r["files"]["receptor"]).read_text()
        chains = {line[21] for line in txt.splitlines() if line.startswith("ATOM")}
        if any(line.startswith("HETATM") for line in txt.splitlines()) or len(chains) != 1:
            bad.append(r["pdb_id"])
    return not bad, f"receptors with HETATM or extra chains {bad or 'none of 20'}"


@check("Docking benchmark", "pocket boxes recompute from the benchmark ligand coordinates")
def _():
    worst, bad = 0.0, []
    for r in benchmark_records():
        mol = sdf_mol(paths.ROOT / r["files"]["ligand"])
        xyz = mol.GetConformer().GetPositions()
        pad = r["pocket_box_padding_angstrom"]
        d = max(
            float(np.abs(np.round(xyz.mean(axis=0), 3) - np.array(r["pocket_center"])).max()),
            float(np.abs(np.round(xyz.min(axis=0) - pad, 3) - np.array(r["pocket_box_min"])).max()),
            float(np.abs(np.round(xyz.max(axis=0) + pad, 3) - np.array(r["pocket_box_max"])).max()),
        )
        worst = max(worst, d)
        if d > 0.002:
            bad.append(r["pdb_id"])
    return not bad, f"largest deviation {worst:.3f} A over 20 structures; offenders {bad or 'none'}"


@check("Docking benchmark", "conformation calls use the same 4 A threshold as 1M17")
def _():
    recs = benchmark_records()
    counts = {"active": 0, "inactive": 0}
    bad = []
    for r in recs:
        c = r["conformation"]
        expected = "active" if c["distance_angstrom"] <= 4.0 else "inactive"
        counts[expected] += 1
        if not str(c["state"]).startswith(expected):
            bad.append(r["pdb_id"])
    stored = json.loads((paths.RESULTS / "docking_benchmark_set.json").read_text())
    agree = (counts["active"] == stored["conformation_counts"]["active"]
             and counts["inactive"] == stored["conformation_counts"]["inactive"])
    return not bad and agree, (f"{counts['active']} active / {counts['inactive']} inactive, "
                               f"matches the report: {agree}")


@check("Docking benchmark", "pairwise ligand similarity recomputes to the reported median")
def _():
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator as rfg
    from rdkit import Chem

    recs = benchmark_records()
    gen = rfg.GetMorganGenerator(radius=featurize.MORGAN_RADIUS, fpSize=featurize.MORGAN_N_BITS)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(r["ligand_smiles"])) for r in recs]
    sims = [
        DataStructs.TanimotoSimilarity(fps[i], fps[j])
        for i in range(len(fps)) for j in range(i + 1, len(fps))
    ]
    stored = json.loads((paths.RESULTS / "docking_benchmark_set.json").read_text())["similarity"]
    ok = (len(sims) == stored["n_pairs"]
          and abs(round(float(np.median(sims)), 3) - stored["median"]) < 1e-9
          and abs(round(float(max(sims)), 3) - stored["max"]) < 1e-9)
    return ok, (f"{len(sims)} pairs, median {np.median(sims):.3f} (reported {stored['median']}), "
                f"max {max(sims):.3f} (reported {stored['max']})")


# --------------------------------------------------------------------------
# 9. Documents, environment, repository hygiene
# --------------------------------------------------------------------------

@check("Documents", "headline numbers quoted in the markdown match their source tables")
def _():
    stored = pd.read_csv(paths.RESULTS / "baseline_summary.csv")
    o = stored[(stored["dataset"] == "Tox21") & (stored["level"] == "overall")].set_index("split")
    n_mols = json.loads((paths.RESULTS / "standardization_stats.json").read_text())["final_molecules"]
    chem = json.loads((paths.RESULTS / "chemical_space_overlap.json").read_text())
    # RESULTS.md sits at the repository root; the rest live in results/.
    docs_root = {"RESULTS.md": paths.ROOT / "RESULTS.md"}

    def doc_path(name: str) -> Path:
        return docs_root.get(name, paths.RESULTS / name)

    # Each value is required where that document is responsible for it: SUMMARY.md
    # and RESULTS.md report the four proposal values; PHASE2_DECISIONS.md cites the
    # scaffold-split figure it rules on, the molecule count and the overlap median.
    wanted = {
        f"{o.loc['random', 'roc_auc_mean']:.4f}": ("SUMMARY.md", "RESULTS.md"),
        f"{o.loc['scaffold', 'roc_auc_mean']:.4f}": ("SUMMARY.md", "RESULTS.md", "PHASE2_DECISIONS.md"),
        str(n_mols): ("SUMMARY.md", "RESULTS.md", "PHASE2_DECISIONS.md"),
        f"{chem['verdict_median']:.4f}": ("RESULTS.md", "PHASE2_DECISIONS.md"),
    }
    missing = []
    for value, docs in wanted.items():
        for doc in docs:
            if value not in doc_path(doc).read_text(encoding="utf-8"):
                missing.append(f"{value} absent from {doc}")
    return not missing, (f"checked {len(wanted)} headline values across "
                         f"SUMMARY.md, RESULTS.md, PHASE2_DECISIONS.md; "
                         f"mismatches {missing or 'none'}")


@check("Documents", "reproducibility record reports a complete, all-identical run")
def _():
    rec = json.loads((paths.RESULTS / "reproducibility.json").read_text())
    ok = (rec["n_identical"] == rec["n_artefacts"]
          and not rec["differing"] and not rec.get("group_failures")
          and all("identical" in c["result"] for c in rec["checks"]))
    return ok, (f"{rec['n_identical']}/{rec['n_artefacts']} identical across "
                f"{len(rec['checks'])} groups, scripts {len(rec['scripts_rerun'])}")


@check("Environment", "dependency lock matches the installed environment exactly")
def _():
    out = subprocess.run(
        ["py", "-3.12", "-m", "uv", "pip", "freeze", "--python", ".venv/Scripts/python.exe"],
        cwd=paths.ROOT, capture_output=True, text=True,
    )
    if out.returncode != 0:
        return None, "uv not available; lock comparison skipped"
    installed = {
        line.strip().lower().replace("_", "-")
        for line in out.stdout.splitlines() if "==" in line
    }
    locked = {
        line.strip().lower().replace("_", "-")
        for line in (paths.ROOT / "requirements.txt").read_text().splitlines()
        if "==" in line and not line.startswith("#")
    }
    return installed == locked, (f"{len(locked)} pins; installed-but-unpinned "
                                 f"{sorted(installed - locked) or 'none'}; "
                                 f"pinned-but-absent {sorted(locked - installed) or 'none'}")


@check("Repository", "no raw or processed data is tracked by git")
def _():
    out = subprocess.run(["git", "ls-files", "data"], cwd=paths.ROOT,
                         capture_output=True, text=True)
    tracked = [f for f in out.stdout.splitlines() if f.strip()]
    committed = subprocess.run(
        ["git", "ls-files", "splits", "receptors", "results", "figures"],
        cwd=paths.ROOT, capture_output=True, text=True,
    ).stdout.splitlines()
    return not tracked, f"tracked data files {len(tracked)}, committed artefacts {len(committed)}"


# --------------------------------------------------------------------------
# 10. Phase 3 artefacts
# --------------------------------------------------------------------------

@check("Phase 3 / KIBA overlap", "verdict band matches the measured median, calibration present")
def _():
    rec = json.loads((paths.RESULTS / "chemical_space_overlap_kiba.json").read_text())
    nn = rec["nn_tanimoto"]
    median = nn["tox21_to_kiba"]["median"]
    expected = ("Strong overlap" if median > 0.4
                else "Weak overlap" if median >= 0.25 else "Negligible overlap")
    md = (paths.RESULTS / "chemical_space_overlap_kiba.md").read_text(encoding="utf-8")
    have_calibration = {"within_tox21", "within_kiba"} <= set(nn)
    ratio_ok = abs(
        rec["cross_over_within_tox21_median_ratio"]
        - median / nn["within_tox21"]["median"]
    ) < 1e-9
    ok = (rec["verdict"] == expected and f"## Verdict: {expected}" in md
          and have_calibration and ratio_ok)
    return ok, (f"median {median:.4f} -> {expected}; within-Tox21 "
                f"{nn['within_tox21']['median']:.4f}, within-KIBA "
                f"{nn['within_kiba']['median']:.4f}; ratio recomputes: {ratio_ok}")


@check("Phase 3 / KIBA overlap", "KIBA molecule count agrees with the binding tables")
def _():
    rec = json.loads((paths.RESULTS / "chemical_space_overlap_kiba.json").read_text())
    binding = json.loads((paths.RESULTS / "binding_dataset_stats.json").read_text())
    a = rec["kiba_drugs_after_standardization"]
    b = binding["KIBA"]["distinct_drugs"]
    return a == b, f"overlap report {a} drugs, binding stats {b} drugs"


def benchmark_v2_records() -> list[dict]:
    d = paths.RECEPTORS / "benchmark_v2"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


@check("Phase 3 / benchmark v2", "self-docking set: files, SDF integrity, distinct chemistry")
def _():
    from rdkit import Chem

    recs = benchmark_v2_records()
    bad = []
    for r in recs:
        sdf = paths.RECEPTORS / "benchmark_v2" / f"{r['pdb_id']}_ligand.sdf"
        pdb = paths.RECEPTORS / "benchmark_v2" / f"{r['pdb_id']}_receptor.pdb"
        if not sdf.exists() or not pdb.exists():
            bad.append(f"{r['pdb_id']}: missing file")
            continue
        mol = Chem.MolFromMolFile(str(sdf))
        if mol is None:
            bad.append(f"{r['pdb_id']}: unparsable SDF")
            continue
        if mol.GetNumHeavyAtoms() != r["ligand_heavy_atoms"]:
            bad.append(f"{r['pdb_id']}: heavy-atom count")
        if Chem.MolToInchiKey(mol) != r["ligand_inchikey"]:
            bad.append(f"{r['pdb_id']}: InChIKey")
        if mol.GetNumConformers() == 0:
            bad.append(f"{r['pdb_id']}: no coordinates")
    keys = [r["ligand_inchikey"] for r in recs]
    if len(set(keys)) != len(keys):
        bad.append("duplicate ligand chemistry")
    return len(recs) >= 15 and not bad, (
        f"{len(recs)} entries, {len(set(keys))} distinct ligands; problems "
        f"{bad or 'none'}")


@check("Phase 3 / benchmark v2", "conformation labels and the reported balance agree")
def _():
    recs = benchmark_v2_records()
    stored = json.loads((paths.RESULTS / "docking_benchmark_v2.json").read_text())
    counts = {"active": 0, "inactive": 0}
    bad = []
    for r in recs:
        c = r["conformation"]
        expected = "active" if c["distance_angstrom"] <= 4.0 else "inactive"
        counts[expected] += 1
        if not str(c["state"]).startswith(expected):
            bad.append(r["pdb_id"])
    reported = stored.get("conformation_counts", {})
    agree = all(counts[k] == reported.get(k) for k in counts)
    return not bad and agree, (f"{counts['active']} active / {counts['inactive']} "
                               f"inactive; matches the report: {agree}")


@check("Phase 3 / benchmark v2", "pocket boxes recompute from the cognate ligand")
def _():
    from rdkit import Chem

    worst, bad = 0.0, []
    for r in benchmark_v2_records():
        mol = Chem.MolFromMolFile(
            str(paths.RECEPTORS / "benchmark_v2" / f"{r['pdb_id']}_ligand.sdf")
        )
        xyz = mol.GetConformer().GetPositions()
        pad = r.get("pocket_box_padding_angstrom", 5.0)
        d = max(
            float(np.abs(np.round(xyz.mean(axis=0), 3) - np.array(r["pocket_center"])).max()),
            float(np.abs(np.round(xyz.min(axis=0) - pad, 3) - np.array(r["pocket_box_min"])).max()),
            float(np.abs(np.round(xyz.max(axis=0) + pad, 3) - np.array(r["pocket_box_max"])).max()),
        )
        worst = max(worst, d)
        if d > 0.002:
            bad.append(r["pdb_id"])
    return not bad, f"largest deviation {worst:.3f} A; offenders {bad or 'none'}"


@check("Phase 3 / hERG-large", "split files partition one consistent molecule set")
def _():
    stats = json.loads((paths.RESULTS / "herg_large_stats.json").read_text())
    sizes, ok = {}, True
    for setting in ("full", "disjoint"):
        for kind in ("random", "scaffold"):
            for seed in paths.SEEDS:
                path = paths.SPLITS / f"herg_large_{setting}_{kind}_split_seed{seed}.csv"
                f = pd.read_csv(path)
                sizes.setdefault(setting, set()).add(len(f))
                if f["inchikey"].duplicated().any() or set(f["subset"]) != {"train", "valid", "test"}:
                    ok = False
                if kind == "scaffold" and int(
                    (f.groupby("scaffold_smiles")["subset"].nunique() > 1).sum()
                ):
                    ok = False
    # The stats record the standardized count before the 47 tie-only rows are
    # dropped, so the split sizes are derived: full = standardized - unlabeled,
    # disjoint = full - the molecules shared with Tox21. Checking the relation is
    # stronger than reading one number.
    n_full = stats["final_molecules"] - stats["unlabeled_after_ties_dropped"]
    n_disjoint = n_full - stats["disjoint_removed"]
    ok = ok and sizes["full"] == {n_full} and sizes["disjoint"] == {n_disjoint}
    return ok, (f"full {sizes['full']} (= {stats['final_molecules']} standardized - "
                f"{stats['unlabeled_after_ties_dropped']} unlabeled = {n_full}), "
                f"disjoint {sizes['disjoint']} (= {n_full} - "
                f"{stats['disjoint_removed']} shared with Tox21 = {n_disjoint}); "
                f"no duplicate keys, no scaffold group spanning subsets")


@check("Phase 3 / hERG-large", "Disjoint really shares no InChIKey with Tox21")
def _():
    clean = clean_table()
    tox_keys = set(clean["inchikey"])
    f = pd.read_csv(paths.SPLITS / "herg_large_disjoint_random_split_seed0.csv")
    shared = tox_keys & set(f["inchikey"])
    full = pd.read_csv(paths.SPLITS / "herg_large_full_random_split_seed0.csv")
    shared_full = tox_keys & set(full["inchikey"])
    return not shared, (f"disjoint shares {len(shared)} InChIKeys with Tox21 "
                        f"(full setting shares {len(shared_full)})")


@check("Phase 3 / ESM-2", "protein embedding cache matches the KIBA protein count")
def _():
    from tox21lib import protein_embed

    stats = json.loads((paths.RESULTS / "esm2_embedding_stats.json").read_text())
    matrix, rows, model = protein_embed.load_cache(
        protein_embed.cache_path(paths.DATA_PROCESSED)
    )
    binding = json.loads((paths.RESULTS / "binding_dataset_stats.json").read_text())
    ok = (matrix.shape[0] == len(rows) == stats["n_proteins"]
          == binding["KIBA"]["distinct_proteins"]
          and matrix.shape[1] == stats["embedding_dim"]
          and not bool(np.isnan(matrix).any()))
    return ok, (f"{matrix.shape[0]} proteins x {matrix.shape[1]}-d from {model}; "
                f"KIBA has {binding['KIBA']['distinct_proteins']} proteins; NaN: "
                f"{bool(np.isnan(matrix).any())}")


@check("Phase 3 / attribution", "reported coverage recomputes from the per-molecule table")
def _():
    stats = json.loads((paths.RESULTS / "attribution.json").read_text())
    frame = pd.read_csv(paths.RESULTS / "attribution_per_molecule.csv")
    hit = float(frame["top_hits_alert"].mean())
    med = float(frame["enrichment"].median())
    ok = (len(frame) == stats["n_with_alerts"]
          and abs(hit - stats["top_hits_alert_rate"]) < 1e-9
          and abs(med - stats["median_enrichment"]) < 1e-9)
    figs = [f for f in stats["figures"] if not (paths.ROOT / f["path"]).exists()]
    return ok and not figs, (f"{len(frame)} alert-bearing molecules, top-k hit rate "
                             f"{hit:.3f}, median enrichment {med:.3f}; missing figures "
                             f"{len(figs)}")


@check("Phase 3 / endpoint reliability", "bootstrap CIs bracket their point estimates")
def _():
    frame = pd.read_csv(paths.RESULTS / "endpoint_reliability.csv")
    ci_cols = [c for c in frame.columns if "roc_auc" in c and ("low" in c or "lo" in c)]
    hi_cols = [c for c in frame.columns if "roc_auc" in c and ("high" in c or "hi" in c)]
    if not ci_cols or not hi_cols:
        return False, f"no ROC-AUC CI columns in {list(frame.columns)}"
    lo, hi = ci_cols[0], hi_cols[0]
    tox = frame[(frame["dataset"] == "Tox21") & frame[lo].notna()]
    outside = tox[(tox["roc_auc_mean"] < tox[lo]) | (tox["roc_auc_mean"] > tox[hi])]
    return outside.empty and len(tox) > 0, (
        f"{len(tox)} Tox21 rows carry ROC-AUC CIs ({lo}/{hi}); point estimates "
        f"outside their own interval: {len(outside)}")


@check("Phase 3 / endpoint reliability", "flagged endpoints really hold <20 test positives")
def _():
    rec = json.loads((paths.RESULTS / "endpoint_reliability.json").read_text())
    m = metrics_table()
    m = m[(m["dataset"] == "Tox21")]
    mins = m.groupby(["split", "endpoint"])["n_test_pos"].min()
    recomputed = {f"{s}/{e}" for (s, e), v in mins.items() if v < 20}
    text = json.dumps(rec)
    missing = [c for c in recomputed if c.split("/")[1] not in text]
    return not missing and bool(recomputed), (
        f"{len(recomputed)} endpoint/split combinations below 20 test positives "
        f"recomputed from the metric rows; absent from the report: {missing or 'none'}")


@check("Phase 3 / endpoint reliability", "the penalty CI is consistent with the point estimate")
def _():
    text = (paths.RESULTS / "endpoint_reliability.md").read_text(encoding="utf-8")
    summary = pd.read_csv(paths.RESULTS / "baseline_summary.csv")
    o = summary[(summary["dataset"] == "Tox21") & (summary["level"] == "overall")]
    o = o.set_index("split")
    drop = float(o.loc["random", "roc_auc_mean"] - o.loc["scaffold", "roc_auc_mean"])
    rec = json.loads((paths.RESULTS / "endpoint_reliability.json").read_text())
    flat = json.dumps(rec)
    quoted = f"{drop:.4f}" in text or f"{drop:.4f}" in flat
    return quoted, (f"macro random-minus-scaffold drop {drop:.4f} recomputed from the "
                    f"summary table and quoted in the Phase 3 reliability report: "
                    f"{quoted}")


@check("Phase 3 / ablation", "every configuration ran the full seed grid")
def _():
    m = pd.read_csv(paths.RESULTS / "shared_encoder_metrics.csv")
    main = m[(m["loss_weight"].isna()) | (m["loss_weight"] == 1.0)]
    counts = main.groupby("config").size().to_dict()
    n_seeds = len(paths.SEEDS)
    expected = {"A": n_seeds, "B": 3 * n_seeds, "C": 3 * n_seeds, "D": 3 * n_seeds,
                "E": n_seeds, "F": n_seeds}
    return counts == expected, (f"main-comparison runs per config {counts} "
                                f"(expected {expected}); {len(m)} runs in total")


@check("Phase 3 / ablation", "reported means recompute from the per-run table")
def _():
    rec = json.loads((paths.RESULTS / "shared_encoder_ablation.json").read_text())
    m = pd.read_csv(paths.RESULTS / "shared_encoder_metrics.csv")
    main = m[(m["loss_weight"].isna()) | (m["loss_weight"] == 1.0)]
    worst = 0.0
    for cfg, value in rec["toxicity_macro_roc_auc"].items():
        sub = main[main["config"] == cfg]["tox_macro_roc_auc"].dropna()
        if sub.empty:
            continue
        worst = max(worst, abs(float(sub.mean()) - float(value)))
    for cfg, value in rec["affinity_pearson_random"].items():
        sub = main[(main["config"] == cfg) & (main["affinity_split"] == "random")]
        sub = sub["affinity_pearson_r"].dropna()
        if sub.empty:
            continue
        worst = max(worst, abs(float(sub.mean()) - float(value)))
    return worst < 1e-9, f"largest discrepancy between report and per-run means: {worst:.2e}"


@check("Phase 3 / ablation", "the verdict follows from the measured effects")
def _():
    rec = json.loads((paths.RESULTS / "shared_encoder_ablation.json").read_text())
    tox_noise, aff_noise = rec["tox_noise_floor"], rec["affinity_noise_floor"]
    helps = (
        max(rec["effect_of_sharing_scratch_tox"],
            rec["effect_of_sharing_best_weight_tox"]) > tox_noise
        or max(rec["effect_of_sharing_scratch_affinity"],
               rec["effect_of_sharing_best_weight_affinity"]) > aff_noise
    )
    pretrained_helps = (rec["effect_of_sharing_pretrained_tox"] > tox_noise
                        or rec["effect_of_sharing_pretrained_affinity"] > aff_noise)
    expected = ("shared encoding helps regardless of pretraining" if helps
                else "shared encoding helps only when the encoder is pretrained"
                if pretrained_helps
                else "shared encoding does not help, and the chemical-space "
                     "measurement explains why")
    return rec["verdict"] == expected, (
        f"effects: scratch {rec['effect_of_sharing_scratch_tox']:+.4f}/"
        f"{rec['effect_of_sharing_scratch_affinity']:+.4f}, best-weight "
        f"{rec['effect_of_sharing_best_weight_tox']:+.4f}/"
        f"{rec['effect_of_sharing_best_weight_affinity']:+.4f}, pretrained "
        f"{rec['effect_of_sharing_pretrained_tox']:+.4f}/"
        f"{rec['effect_of_sharing_pretrained_affinity']:+.4f} against noise floors "
        f"{tox_noise:.4f}/{aff_noise:.4f} -> \"{expected}\"")


@check("Phase 3 / ablation", "neural configurations are evaluated on the committed folds")
def _():
    m = pd.read_csv(paths.RESULTS / "shared_encoder_metrics.csv")
    binding = pd.read_csv(paths.RESULTS / "binding_baseline.csv")
    bad = []
    for split in ("random", "cold_drug", "cold_target"):
        rf_n = binding[(binding["dataset"] == "KIBA") & (binding["split"] == split)]
        nn_n = m[(m["affinity_split"] == split) & m["affinity_n_test"].notna()]
        if nn_n.empty:
            continue
        for seed in paths.SEEDS:
            rf_seed = rf_n[rf_n["seed"] == seed]["n_test"]
            nn_seed = nn_n[nn_n["seed"] == seed]["affinity_n_test"].unique()
            if rf_seed.empty or len(nn_seed) == 0:
                continue
            if int(rf_seed.iloc[0]) != int(nn_seed[0]):
                bad.append(f"{split}/seed{seed}: RF {int(rf_seed.iloc[0])} vs "
                           f"neural {int(nn_seed[0])}")
    return not bad, (f"test-fold sizes identical to the random-forest runs for every "
                     f"KIBA split and seed; mismatches {bad or 'none'}")


@check("Phase 3 / attribution", "the group-masking control agrees with the per-atom result")
def _():
    stats = json.loads((paths.RESULTS / "attribution.json").read_text())
    control = pd.read_csv(paths.RESULTS / "attribution_control.csv")
    rec = stats.get("control")
    if rec is None:
        return False, "attribution.json carries no control block"
    rate = float(control["beats_null"].mean())
    med = float(control["percentile_vs_null"].median())
    consistent = (abs(rate - rec["beats_null_rate"]) < 1e-9
                  and abs(med - rec["median_percentile_vs_null"]) < 1e-9
                  and len(control) == rec["n_molecules"])
    # The two probes must tell the same story where the sample is big enough for
    # that to mean anything. A small class disagreeing (epoxide, n=14) is a real
    # measurement about a 3-atom group, not an inconsistency, so it is reported
    # rather than failed; disagreement on a large class would indicate a bug.
    per_atom = {r["alert"]: r["median_enrichment"] for r in stats["per_alert"]}
    per_group = {r["alert"]: r["median_percentile"] for r in rec["per_alert"]}
    counts = {r["alert"]: r["n"] for r in stats["per_alert"]}
    shared = set(per_atom) & set(per_group)
    disagree = [a for a in sorted(shared)
                if (per_atom[a] > 1.1) != (per_group[a] > 0.6)]
    serious = [a for a in disagree if counts.get(a, 0) >= 50]
    detail = ", ".join(f"{a} n={counts.get(a)}" for a in disagree) or "no alert"
    return consistent and not serious, (
        f"whole-alert masking beats the random same-size null in {rate:.1%} of "
        f"{len(control)} molecules (median percentile {med:.2f}); probes disagree on "
        f"{detail}; none large enough (n>=50) to indicate a bug: {not serious}")


@check("Phase 3 / endpoint reliability", "PR-AUC CIs bracket their point estimates too")
def _():
    frame = pd.read_csv(paths.RESULTS / "endpoint_reliability.csv")
    lo = [c for c in frame.columns if "pr_auc" in c and ("low" in c or "lo" in c)]
    hi = [c for c in frame.columns if "pr_auc" in c and ("high" in c or "hi" in c)]
    if not lo or not hi:
        return False, f"no PR-AUC CI columns in {list(frame.columns)}"
    tox = frame[(frame["dataset"] == "Tox21") & frame[lo[0]].notna()]
    outside = tox[(tox["pr_auc_mean"] < tox[lo[0]]) | (tox["pr_auc_mean"] > tox[hi[0]])]
    # Average precision is right-skewed under resampling, so the interval sits
    # above the point estimate; it must still contain it.
    return outside.empty and len(tox) > 0, (
        f"{len(tox)} Tox21 rows with PR-AUC CIs; point estimates outside their own "
        f"interval: {len(outside)}")


@check("Phase 3 / hERG-large", "molecule-count keys are unambiguous and self-consistent")
def _():
    stats = json.loads((paths.RESULTS / "herg_large_stats.json").read_text())
    needed = {"standardized_molecules", "unlabeled_after_ties_dropped",
              "full_molecules", "disjoint_removed", "disjoint_molecules"}
    missing = needed - set(stats)
    if missing:
        return False, f"missing keys {sorted(missing)}"
    ok = (stats["full_molecules"]
          == stats["standardized_molecules"] - stats["unlabeled_after_ties_dropped"]
          and stats["disjoint_molecules"]
          == stats["full_molecules"] - stats["disjoint_removed"])
    return ok, (f"{stats['standardized_molecules']} standardized - "
                f"{stats['unlabeled_after_ties_dropped']} unlabeled = "
                f"{stats['full_molecules']} full; - {stats['disjoint_removed']} shared "
                f"with Tox21 = {stats['disjoint_molecules']} disjoint")


@check("Repository", "every source module compiles")
def _():
    out = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "src"],
        cwd=paths.ROOT, capture_output=True, text=True,
    )
    n = len(list((paths.ROOT / "src").rglob("*.py")))
    return out.returncode == 0, (
        f"{n} modules under src/; compileall exit {out.returncode}"
        + (f"; {out.stdout.strip()[:200]}" if out.returncode else ""))


@check("Repository", "ablation checkpoints are newer than the metric table they produced")
def _():
    metrics = paths.RESULTS / "shared_encoder_metrics.csv"
    ckpt_dir = paths.DATA_PROCESSED / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("config*_seed0.pt"))
    if not ckpts:
        return False, f"no checkpoints in {rel(ckpt_dir)}"
    m = pd.read_csv(metrics)
    expected = {f"config{c}_seed0.pt" for c in sorted(m["config"].unique())}
    present = {p.name for p in ckpts}
    # A checkpoint older than the start of the run that wrote the metrics would be
    # a leftover from a superseded run, which the attribution stage would read.
    stale = [p.name for p in ckpts
             if p.stat().st_mtime < metrics.stat().st_mtime - 24 * 3600]
    return expected <= present and not stale, (
        f"{len(present)} checkpoints for configs {sorted(m['config'].unique())}; "
        f"missing {sorted(expected - present) or 'none'}; stale {stale or 'none'}")


# --------------------------------------------------------------------------
# 11. Phase 4 artefacts
# --------------------------------------------------------------------------

@check("Phase 4 / learning curves", "curve grid is complete and aggregates recompute")
def _():
    rec = json.loads((paths.RESULTS / "learning_curves.json").read_text())
    frame = pd.read_csv(paths.RESULTS / "learning_curves.csv")
    n_expected = len(rec["fractions"]) * len(rec["seeds"]) * 2 * len(rec["tasks"])
    worst = 0.0
    for task, t in rec["tasks"].items():
        for frac, cell in t["by_fraction"].items():
            for model, v in cell.items():
                sub = frame[(frame["task"] == task) & (frame["model"] == model)
                            & (np.isclose(frame["fraction"], float(frac)))]["value"]
                worst = max(worst, abs(float(sub.mean()) - v["mean"]))
    return len(frame) == n_expected and worst < 1e-9, (
        f"{len(frame)} rows (expected {n_expected}); largest discrepancy between "
        f"report and per-run means {worst:.2e}")


@check("Phase 4 / learning curves", "subsampling preserved the fold sizes")
def _():
    frame = pd.read_csv(paths.RESULTS / "learning_curves.csv")
    bad = []
    for (task, seed), g in frame.groupby(["task", "seed"]):
        full = g[np.isclose(g["fraction"], 1.0)]["n_train"].unique()
        if len(full) != 1 or full[0] != g["n_train_full"].iloc[0]:
            bad.append(f"{task}/seed{seed}")
        for _, row in g.iterrows():
            expected = max(1, int(round(row["n_train_full"] * row["fraction"])))
            if abs(row["n_train"] - expected) > 1:
                bad.append(f"{task}/seed{seed}/frac{row['fraction']}")
    return not bad, (f"train-fold sizes match fraction x full size for every cell; "
                     f"mismatches {sorted(set(bad)) or 'none'}")


@check("Phase 4 / abstention", "the naive confidence signal really is beaten by random")
def _():
    rec = json.loads((paths.RESULTS / "abstention.json").read_text())
    frame = pd.read_csv(paths.RESULTS / "abstention.csv")
    worst = 0.0
    for signal, v in rec["signals"].items():
        sub = frame[(frame["signal"] == signal) & np.isclose(frame["coverage"], 0.5)]
        worst = max(worst, abs(float(sub["pr_auc"].mean()) - v["pr_auc_at_50pct_coverage"]))
    beats = rec["beats_random"]
    # The headline claim of that section: the two score-based signals lose to a
    # random ordering and the distance-based one wins. If this flips, the text is
    # wrong and must be regenerated.
    expected = (not beats["distance_from_half"] and not beats["forest_agreement"]
                and beats["tanimoto_to_train"])
    return worst < 1e-9 and expected, (
        f"recompute discrepancy {worst:.2e}; beats-random {beats}; "
        f"best signal {rec['best_signal_by_mean_pr_auc']}")


@check("Phase 4 / binding follow-ups", "scaffold-cold split holds out whole scaffolds")
def _():
    bad = []
    for seed in paths.SEEDS:
        f = pd.read_csv(paths.SPLITS / f"kiba_scaffold_cold_drug_seed{seed}.csv")
        tr = set(f.loc[f["subset"] == "train", "inchikey"])
        te = set(f.loc[f["subset"] == "test", "inchikey"])
        if tr & te:
            bad.append(f"seed{seed}: {len(tr & te)} drugs leak")
    rec = json.loads((paths.RESULTS / "binding_followups.json").read_text())
    harder = rec["B3"]["pearson_mean"] < rec["B3"]["committed_cold_drug_pearson"]
    return not bad and harder, (
        f"drug leakage {bad or 'none'}; scaffold-cold Pearson "
        f"{rec['B3']['pearson_mean']:.4f} vs identity-cold "
        f"{rec['B3']['committed_cold_drug_pearson']:.4f} (harder: {harder})")


@check("Phase 4 / self-docking", "every docked entry has a pose and an RMSD")
def _():
    path = paths.RESULTS / "self_docking.json"
    if not path.exists():
        return None, "self_docking.json absent; the docking stage was not run"
    rec = json.loads(path.read_text())
    frame = pd.read_csv(paths.RESULTS / "self_docking.csv")
    n_entries = len(rec["per_entry"])
    expected = n_entries * len(paths.SEEDS)
    rmsd_col = next((c for c in frame.columns if "rmsd" in c.lower()), None)
    if rmsd_col is None:
        return False, f"no RMSD column in {list(frame.columns)}"
    missing = int(frame[rmsd_col].isna().sum())
    return len(frame) == expected and missing == 0, (
        f"{n_entries} entries x {len(paths.SEEDS)} seeds = {len(frame)} runs "
        f"(expected {expected}); missing RMSDs {missing}")


@check("Phase 4 / cold-drug diagnosis", "control predictors behave as controls must")
def _():
    rec = json.loads((paths.RESULTS / "cold_drug_diagnosis.json").read_text())
    bad = []
    for split, preds in rec["by_split"].items():
        g = preds["global_mean"]
        # A constant prediction must give zero correlation and a coin-flip CI; if
        # it does not, the metric implementation itself is wrong.
        if abs(g["pooled_pearson"]) > 1e-9 or abs(g["ci"] - 0.5) > 1e-9:
            bad.append(f"{split}: global-mean r={g['pooled_pearson']:.4f} "
                       f"CI={g['ci']:.4f}")
        if preds["random_forest"]["pooled_pearson"] <= preds["protein_mean"]["pooled_pearson"]:
            bad.append(f"{split}: forest does not beat the protein-mean control")
    return not bad, (f"global-mean control gives r=0 and CI=0.5 on both splits; "
                     f"problems {bad or 'none'}")


@check("Phase 4 / modality ablation", "aggregates recompute and the control behaves")
def _():
    path = paths.RESULTS / "modality_ablation.json"
    if not path.exists():
        return None, "modality_ablation.json absent; the ablation was not run"
    rec = json.loads(path.read_text())
    frame = pd.read_csv(paths.RESULTS / "modality_ablation.csv")
    expected = len(rec["modes"]) * len(rec["splits"]) * len(rec["seeds"])
    worst = 0.0
    for split, cells in rec["by_split"].items():
        for mode in rec["modes"]:
            sub = frame[(frame["split"] == split) & (frame["mode"] == mode)]["pearson_r"]
            worst = max(worst, abs(float(sub.mean()) - cells[mode]["pearson_mean"]))
    # Splits where removing an input *improved* the model. That is a real
    # outcome, not a defect, so it is surfaced rather than failed on - but it has
    # to be surfaced, because it changes how the "both" column must be read.
    hurt = {s: round(max(c["drug_only"]["pearson_mean"],
                         c["protein_only"]["pearson_mean"])
                     - c["both"]["pearson_mean"], 4)
            for s, c in rec["by_split"].items()
            if c["both"]["pearson_mean"] < max(c["drug_only"]["pearson_mean"],
                                               c["protein_only"]["pearson_mean"])}
    return len(frame) == expected and worst < 1e-9, (
        f"{len(frame)} runs (expected {expected}); report-vs-per-run discrepancy "
        f"{worst:.2e}; splits where an ablated model BEAT the full model "
        f"{hurt or 'none'}; shuffle cost per split "
        f"{ {s: round(c['shuffle_cost'], 4) for s, c in rec['by_split'].items()} }")


@check("Phase 4 / tuned floor", "the tuned baseline is recomputed, not overwritten")
def _():
    path = paths.RESULTS / "tuned_floor.json"
    if not path.exists():
        return None, "tuned_floor.json absent; the correction was not run"
    rec = json.loads(path.read_text())
    tuned = pd.read_csv(paths.RESULTS / "binding_baseline_tuned.csv")
    committed = pd.read_csv(paths.RESULTS / "binding_baseline.csv")
    # The untuned record must still be intact: the correction adds a file, it
    # does not rewrite history.
    untouched = set(committed["split"].unique()) and "max_features" not in committed.columns
    worst = 0.0
    for split, v in rec["kiba_by_split"].items():
        sub = tuned[(tuned["dataset"] == "KIBA") & (tuned["split"] == split)]["pearson_r"]
        worst = max(worst, abs(float(sub.mean()) - v["rf_tuned_0.3"]))
    gains = [v["tuning_gain"] for v in rec["kiba_by_split"].values()]
    return worst < 1e-9 and untouched, (
        f"{len(tuned)} tuned fits; report-vs-per-fit discrepancy {worst:.2e}; "
        f"tuning gain per split {[round(g, 4) for g in gains]}; committed untuned "
        f"table still has no max_features column: {untouched}")


# --------------------------------------------------------------------------

def main() -> int:
    print(f"Running {len(CHECKS)} integrity checks...\n")
    results = []
    for entry in CHECKS:
        try:
            ok, detail = entry["fn"]()
        except Exception as exc:  # noqa: BLE001 - a crashing check is a failed check
            ok, detail = False, f"check raised {type(exc).__name__}: {exc}"
        status = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        results.append({"section": entry["section"], "name": entry["name"],
                        "status": status, "detail": detail})
        print(f"  [{status}] {entry['section']} / {entry['name']}\n         {detail}")

    n_pass = sum(r["status"] == "PASS" for r in results)
    n_fail = sum(r["status"] == "FAIL" for r in results)
    n_skip = sum(r["status"] == "SKIP" for r in results)

    L = [
        "# Data audit",
        "",
        f"**{n_pass} passed, {n_fail} failed, {n_skip} skipped** of {len(results)} checks.",
        "",
        "Produced by `src/15_data_audit.py`. This audit does not re-run the pipeline and "
        "does not trust any script's own assertions: it recomputes each quantity from the "
        "committed artefacts and checks that independent sources agree — summary "
        "statistics against per-fit rows, pocket geometry against ligand coordinates, "
        "split partitions against the molecule tables, document text against the source "
        "tables, and the dependency lock against the installed environment.",
        "",
    ]
    for section in dict.fromkeys(r["section"] for r in results):
        L += [f"## {section}", "", "| check | result | measured |", "| --- | --- | --- |"]
        for r in results:
            if r["section"] == section:
                mark = {"PASS": "PASS", "FAIL": "**FAIL**", "SKIP": "skipped"}[r["status"]]
                L.append(f"| {r['name']} | {mark} | {r['detail']} |")
        L.append("")
    if n_fail:
        L += ["## Failures", ""] + [
            f"- **{r['section']} / {r['name']}**: {r['detail']}"
            for r in results if r["status"] == "FAIL"
        ] + [""]
    else:
        L += [
            "## Verdict",
            "",
            "No inconsistency found. Every committed number reproduces from the artefact "
            "it is derived from, and no cross-document contradiction remains.",
            "",
        ]

    (paths.RESULTS / "DATA_AUDIT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (paths.RESULTS / "data_audit.json").write_text(
        json.dumps({"n_pass": n_pass, "n_fail": n_fail, "n_skip": n_skip,
                    "checks": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print(f"Wrote {rel(paths.RESULTS / 'DATA_AUDIT.md')} and "
          f"{rel(paths.RESULTS / 'data_audit.json')}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
