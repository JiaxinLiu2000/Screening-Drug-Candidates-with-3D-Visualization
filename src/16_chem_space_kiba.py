"""Phase 3 Task 1: chemical-space overlap between Tox21 and KIBA (with within-set calibration).

Follows the exact procedure of ``src/10_chem_space.py`` (whose helpers are reused
verbatim) so the Davis and KIBA measurements are directly comparable.
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tox21lib import datasets, featurize, paths, properties  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "chem_space_davis", Path(__file__).resolve().parent / "10_chem_space.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

OUT_MD = paths.RESULTS / "chemical_space_overlap_kiba.md"
OUT_JSON = paths.RESULTS / "chemical_space_overlap_kiba.json"
DAVIS_JSON = paths.RESULTS / "chemical_space_overlap.json"
FIG_PCA = paths.FIGURES / "chemspace_pca_three_way.png"
VERDICT_DIRECTION = "tox21_to_kiba"
STAT_KEYS = ["min", "p25", "median", "p75", "max", "frac_gt_0.4", "frac_gt_0.6"]


def nn_tanimoto_within(a: np.ndarray, block: int = 2048) -> np.ndarray:
    """Exact max Tanimoto of each row of a to any OTHER row of a (self excluded by index)."""
    a32 = a.astype(np.int32)
    pop = a32.sum(1)
    out = np.empty(len(a32), dtype=np.float64)
    for s in range(0, len(a32), block):
        e = min(s + block, len(a32))
        inter = a32[s:e] @ a32.T
        union = pop[s:e, None] + pop[None, :] - inter
        sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        sim[np.arange(e - s), np.arange(s, e)] = -1.0
        out[s:e] = sim.max(axis=1)
    return out


def load_set(name: str) -> tuple[pd.DataFrame, int, dict]:
    dti = datasets.load_dti(name)
    n_raw = int(len(dti[["drug_id", "smiles"]].drop_duplicates()))
    clean, stats = datasets.unique_molecules(dti)
    return clean.sort_values("inchikey").reset_index(drop=True), n_raw, stats


def main() -> None:
    paths.ensure_dirs()
    tox = pd.read_parquet(paths.CLEAN_PARQUET).sort_values("inchikey").reset_index(drop=True)
    kiba, n_kiba_raw, kiba_stats = load_set("KIBA")
    davis, _, _ = load_set("DAVIS")

    X_tox = featurize.featurize_smiles(tox["canonical_smiles"].tolist())
    X_kib = featurize.featurize_smiles(kiba["canonical_smiles"].tolist())
    X_dav = featurize.featurize_smiles(davis["canonical_smiles"].tolist())

    nn = {
        "tox21_to_kiba": cs.summary(cs.nn_tanimoto(X_tox, X_kib)),
        "kiba_to_tox21": cs.summary(cs.nn_tanimoto(X_kib, X_tox)),
        "within_tox21": cs.summary(nn_tanimoto_within(X_tox)),
        "within_kiba": cs.summary(nn_tanimoto_within(X_kib)),
    }

    shared = sorted(set(tox["inchikey"]) & set(kiba["inchikey"]))
    shared_rows = kiba[kiba["inchikey"].isin(shared)][["inchikey", "canonical_smiles"]]

    P_tox = properties.descriptor_table(tox["canonical_smiles"].tolist())
    P_kib = properties.descriptor_table(kiba["canonical_smiles"].tolist())
    prop_rows = []
    for p in properties.PROPERTIES:
        ks = ks_2samp(P_tox[p].to_numpy(), P_kib[p].to_numpy())
        prop_rows.append({
            "property": p,
            "tox21_mean": float(P_tox[p].mean()), "tox21_std": float(P_tox[p].std()),
            "kiba_mean": float(P_kib[p].mean()), "kiba_std": float(P_kib[p].std()),
            "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
        })
    prop_df = pd.DataFrame(prop_rows)

    X_all = np.vstack([X_tox, X_dav, X_kib]).astype(np.float64)
    pca = PCA(n_components=2, random_state=0)
    pc = pca.fit_transform(X_all)
    ev = pca.explained_variance_ratio_
    n_t, n_d = len(tox), len(davis)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pc[:n_t, 0], pc[:n_t, 1], s=4, alpha=0.3, c="tab:blue", label=f"Tox21 (n={n_t})")
    ax.scatter(pc[n_t + n_d:, 0], pc[n_t + n_d:, 1], s=6, alpha=0.4, c="tab:green",
               label=f"KIBA (n={len(kiba)})")
    ax.scatter(pc[n_t:n_t + n_d, 0], pc[n_t:n_t + n_d, 1], s=18, alpha=0.9, c="tab:red",
               edgecolors="k", linewidths=0.3, label=f"Davis (n={n_d})")
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.2f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.2f}% var)")
    ax.set_title("Tox21 vs Davis vs KIBA, Morgan r=2 2048-bit PCA")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG_PCA, dpi=200, metadata={"Software": None})
    plt.close(fig)

    med = nn[VERDICT_DIRECTION]["median"]
    head, sentence = cs.verdict(med)
    davis_med = json.loads(DAVIS_JSON.read_text(encoding="utf-8"))["nn_tanimoto"]["tox21_to_davis"]["median"]
    davis_head, _ = cs.verdict(davis_med)
    w_tox, w_kib = nn["within_tox21"]["median"], nn["within_kiba"]["median"]
    ratio_tox = med / w_tox
    survives = head == davis_head

    result = {
        "kiba_drugs_before_standardization": n_kiba_raw,
        "kiba_drugs_after_standardization": int(len(kiba)),
        "kiba_standardization_stats": {k: (v if isinstance(v, (int, float, str)) else str(v))
                                       for k, v in kiba_stats.items()},
        "tox21_molecules": int(len(tox)),
        "davis_molecules": int(len(davis)),
        "nn_tanimoto": nn,
        "exact_inchikey_overlap": {"count": len(shared), "inchikeys": shared},
        "properties": prop_rows,
        "pca_explained_variance_ratio": [float(ev[0]), float(ev[1])],
        "pca_sources_order": ["tox21", "davis", "kiba"],
        "verdict_direction": VERDICT_DIRECTION,
        "verdict_median": med,
        "verdict": head,
        "davis_verdict_median": davis_med,
        "davis_verdict": davis_head,
        "davis_conclusion_holds": survives,
        "cross_over_within_tox21_median_ratio": ratio_tox,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    nn_df = pd.DataFrame([{"direction": d, "n": s["n"], **{k: s[k] for k in STAT_KEYS}}
                          for d, s in nn.items()])
    stats_df = pd.DataFrame([{"stat": k, "value": str(v)} for k, v in kiba_stats.items()])
    para = (
        f"The Davis-based conclusion ({davis_head.lower()}, Tox21->Davis median "
        f"{davis_med:.4f}) {'still holds' if survives else 'does NOT hold'} for KIBA: with "
        f"{len(kiba)} drugs instead of {n_d}, the Tox21->KIBA median is {med:.4f}. "
        f"Calibration: the within-set nearest-neighbour medians are {w_tox:.4f} (Tox21) and "
        f"{w_kib:.4f} (KIBA), so the cross-set median is {ratio_tox:.2f}x the within-Tox21 "
        "median. "
    )
    if med < 0.5 * min(w_tox, w_kib):
        para += ("Because molecules in each set find much closer neighbours inside their own "
                 "set than in the other, the low cross-set figure is not just the floor of the "
                 "Morgan metric: the two sets occupy genuinely different regions of chemical "
                 "space, not merely sparse fingerprints.")
    elif med >= 0.8 * min(w_tox, w_kib):
        para += ("The cross-set figure is close to the within-set floor, so the low absolute "
                 "value mostly reflects Morgan fingerprint sparsity rather than disjoint spaces.")
    else:
        para += ("The cross-set figure is clearly below the within-set values, so part of the "
                 "low absolute value is fingerprint sparsity, but a real distributional gap "
                 "remains between the two sets.")
    lines = [
        "# Chemical-space overlap: Tox21 vs KIBA",
        "",
        "Generated by `src/16_chem_space_kiba.py`, following the procedure of "
        "`src/10_chem_space.py` (same helpers). All sets standardized with the same "
        "`tox21lib.standardize` pipeline; fingerprints are Morgan r=2, 2048 bits; Tanimoto "
        "computed exactly (popcount algebra over all pairs). Within-set rows give each "
        "molecule's max similarity to any OTHER molecule of its own set (self excluded by index).",
        "",
        "## Dataset sizes",
        "",
        f"- Tox21 standardized molecules: {len(tox)}",
        f"- KIBA distinct (drug_id, smiles) before standardization: {n_kiba_raw}",
        f"- KIBA molecules after standardization: {len(kiba)}",
        f"- Davis molecules after standardization (PCA only): {n_d}",
        "",
        "KIBA standardization stats:",
        "",
        cs.md_table(stats_df),
        "",
        "## Nearest-neighbour Tanimoto (cross-set and within-set calibration)",
        "",
        cs.md_table(nn_df),
        "",
        "## Exact InChIKey overlap",
        "",
        f"Shared InChIKeys: {len(shared)}",
        "",
    ]
    if len(shared) and len(shared) <= 50:
        lines += [cs.md_table(shared_rows.reset_index(drop=True)), ""]
    lines += [
        "## Property distributions (two-sample KS)",
        "",
        cs.md_table(prop_df, "{:.4g}"),
        "",
        "## Projection",
        "",
        f"![PCA](../figures/chemspace_pca_three_way.png) — PCA (`random_state=0`) over "
        f"Tox21+Davis+KIBA, PC1 {ev[0] * 100:.2f}% / PC2 {ev[1] * 100:.2f}% explained variance.",
        "",
        f"Verdict uses the median nearest-neighbour Tanimoto in the Tox21->KIBA direction "
        f"(each Tox21 molecule's max similarity to any KIBA molecule): {med:.4f}. "
        "Thresholds: >0.4 strong, 0.25-0.4 weak, <0.25 negligible.",
        "",
        f"## Verdict: {head}",
        "",
        sentence,
        "",
        para,
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"nn": nn, "shared": len(shared), "verdict": head}, indent=2))


if __name__ == "__main__":
    main()
