"""Phase 2 Task 2: chemical-space overlap between Tox21 and Davis."""

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

OUT_MD = paths.RESULTS / "chemical_space_overlap.md"
OUT_JSON = paths.RESULTS / "chemical_space_overlap.json"
FIG_PCA = paths.FIGURES / "chemspace_pca.png"
FIG_UMAP = paths.FIGURES / "chemspace_umap.png"
VERDICT_DIRECTION = "tox21_to_davis"


def nn_tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact max Tanimoto of each row of a to any row of b (popcount algebra)."""
    a32, b32 = a.astype(np.int32), b.astype(np.int32)
    inter = a32 @ b32.T
    union = a32.sum(1)[:, None] + b32.sum(1)[None, :] - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return sim.max(axis=1)


def summary(x: np.ndarray) -> dict:
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "max": float(np.max(x)),
        "frac_gt_0.4": float(np.mean(x > 0.4)),
        "frac_gt_0.6": float(np.mean(x > 0.6)),
    }


def scatter(coords, n_tox, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:n_tox, 0], coords[:n_tox, 1], s=4, alpha=0.35, c="tab:blue",
               label=f"Tox21 (n={n_tox})")
    ax.scatter(coords[n_tox:, 0], coords[n_tox:, 1], s=18, alpha=0.9, c="tab:red",
               edgecolors="k", linewidths=0.3, label=f"Davis (n={len(coords) - n_tox})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, metadata={"Software": None})
    plt.close(fig)


def verdict(median: float) -> tuple[str, str]:
    if median > 0.4:
        return (
            "Strong overlap",
            "Shared encoder is well supported, proceed as planned.",
        )
    if median >= 0.25:
        return (
            "Weak overlap",
            "Shared encoder is defensible but we should expect modest gains and say so "
            "in the report.",
        )
    return (
        "Negligible overlap",
        "The shared-encoder experiment will likely show no gain for distributional "
        "reasons, and we need to decide whether to add bridging data or reframe the "
        "research question.",
    )


def md_table(df: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    paths.ensure_dirs()
    tox = pd.read_parquet(paths.CLEAN_PARQUET).sort_values("inchikey").reset_index(drop=True)

    dti = datasets.load_dti("DAVIS")
    raw_drugs = dti[["drug_id", "smiles"]].drop_duplicates()
    n_davis_raw = int(len(raw_drugs))
    davis, davis_stats = datasets.unique_molecules(dti)
    davis = davis.sort_values("inchikey").reset_index(drop=True)

    X_tox = featurize.featurize_smiles(tox["canonical_smiles"].tolist())
    X_dav = featurize.featurize_smiles(davis["canonical_smiles"].tolist())
    nn = {
        "tox21_to_davis": summary(nn_tanimoto(X_tox, X_dav)),
        "davis_to_tox21": summary(nn_tanimoto(X_dav, X_tox)),
    }

    shared = sorted(set(tox["inchikey"]) & set(davis["inchikey"]))
    shared_rows = davis[davis["inchikey"].isin(shared)][["inchikey", "canonical_smiles"]]

    P_tox = properties.descriptor_table(tox["canonical_smiles"].tolist())
    P_dav = properties.descriptor_table(davis["canonical_smiles"].tolist())
    prop_rows = []
    for p in properties.PROPERTIES:
        ks = ks_2samp(P_tox[p].to_numpy(), P_dav[p].to_numpy())
        prop_rows.append({
            "property": p,
            "tox21_mean": float(P_tox[p].mean()), "tox21_std": float(P_tox[p].std()),
            "davis_mean": float(P_dav[p].mean()), "davis_std": float(P_dav[p].std()),
            "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
        })
    prop_df = pd.DataFrame(prop_rows)

    X_all = np.vstack([X_tox, X_dav]).astype(np.float64)
    pca = PCA(n_components=2, random_state=0)
    pc = pca.fit_transform(X_all)
    ev = pca.explained_variance_ratio_
    scatter(pc, len(tox), f"PC1 ({ev[0] * 100:.2f}% var)", f"PC2 ({ev[1] * 100:.2f}% var)",
            "Tox21 vs Davis, Morgan r=2 2048-bit PCA", FIG_PCA)

    umap_note = "not produced"
    try:
        import umap

        emb = umap.UMAP(n_components=2, random_state=0, metric="jaccard").fit_transform(
            X_all.astype(bool))
        scatter(emb, len(tox), "UMAP-1", "UMAP-2",
                "Tox21 vs Davis, UMAP (Jaccard, random_state=0)", FIG_UMAP)
        umap_note = f"umap-learn {umap.__version__}, metric=jaccard, random_state=0"
    except ImportError:
        pass

    med = nn[VERDICT_DIRECTION]["median"]
    head, sentence = verdict(med)

    result = {
        "davis_drugs_before_standardization": n_davis_raw,
        "davis_drugs_after_standardization": int(len(davis)),
        "davis_standardization_stats": {k: (v if isinstance(v, (int, float, str)) else str(v))
                                        for k, v in davis_stats.items()},
        "tox21_molecules": int(len(tox)),
        "nn_tanimoto": nn,
        "exact_inchikey_overlap": {"count": len(shared), "inchikeys": shared},
        "properties": prop_rows,
        "pca_explained_variance_ratio": [float(ev[0]), float(ev[1])],
        "umap": umap_note,
        "verdict_direction": VERDICT_DIRECTION,
        "verdict_median": med,
        "verdict": head,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    stat_keys = ["min", "p25", "median", "p75", "max", "frac_gt_0.4", "frac_gt_0.6"]
    nn_df = pd.DataFrame([{"direction": d, "n": s["n"], **{k: s[k] for k in stat_keys}}
                          for d, s in nn.items()])
    stats_df = pd.DataFrame([{"stat": k, "value": str(v)} for k, v in davis_stats.items()])
    lines = [
        "# Chemical-space overlap: Tox21 vs Davis",
        "",
        "Generated by `src/10_chem_space.py`. Both sets standardized with the same "
        "`tox21lib.standardize` pipeline; fingerprints are Morgan r=2, 2048 bits; "
        "Tanimoto computed exactly (popcount algebra over all pairs).",
        "",
        "## Dataset sizes",
        "",
        f"- Tox21 standardized molecules: {len(tox)}",
        f"- Davis distinct (drug_id, smiles) before standardization: {n_davis_raw}",
        f"- Davis molecules after standardization: {len(davis)}",
        "",
        "Davis standardization stats:",
        "",
        md_table(stats_df),
        "",
        "## Nearest-neighbour Tanimoto",
        "",
        md_table(nn_df),
        "",
        "## Exact InChIKey overlap",
        "",
        f"Shared InChIKeys: {len(shared)}",
        "",
    ]
    if len(shared):
        lines += [md_table(shared_rows.reset_index(drop=True)), ""]
    lines += [
        "## Property distributions (two-sample KS)",
        "",
        md_table(prop_df, "{:.4g}"),
        "",
        "## Projection",
        "",
        f"![PCA](../figures/chemspace_pca.png) — PCA (`random_state=0`), "
        f"PC1 {ev[0] * 100:.2f}% / PC2 {ev[1] * 100:.2f}% explained variance.",
        "",
    ]
    if umap_note != "not produced":
        lines += [f"![UMAP](../figures/chemspace_umap.png) — {umap_note}.", ""]
    lines += [
        f"Verdict uses the median nearest-neighbour Tanimoto in the Tox21->Davis direction "
        f"(each Tox21 molecule's max similarity to any Davis molecule): {med:.4f}. "
        "Thresholds: >0.4 strong, 0.25-0.4 weak, <0.25 negligible.",
        "",
        f"## Verdict: {head}",
        "",
        sentence,
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"nn": nn, "shared": len(shared), "verdict": head}, indent=2))


if __name__ == "__main__":
    main()
