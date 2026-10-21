"""Phase 4 task E2 - a confidence signal that actually works.

Task T2 measured the obvious abstention rule, "reject predictions near 0.5", and
it failed in an instructive way: on the scaffold split at 50% rejection the kept
pool is 2.7% positive while the rejected pool is 15.8% positive. In a 3-18%
positive problem, distance from 0.5 abstains on precisely the actives a toxicity
screen exists to find - accuracy goes up because what remains is easy negatives,
while macro ROC-AUC goes down.

This script compares four confidence signals on the same predictions, scored the
way selective prediction should be scored - as a risk-coverage curve, where risk
is measured by a metric that is not fooled by class imbalance:

    distance_from_half   the naive rule T2 showed to be broken
    forest_variance      disagreement between trees (an uncertainty estimate)
    tanimoto_to_train    distance to the training chemistry (an applicability
                         domain, and the thing the cyclohexanol failure argues for)
    random               a control, so "any ranking looks good" can be ruled out

Coverage is the fraction of test molecules kept. The headline number is the area
under the risk-coverage curve for PR-AUC: lower is better, and a signal that
cannot beat `random` is not a confidence signal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfg
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import ablation_data as adata  # noqa: E402
from tox21lib import featurize, paths  # noqa: E402

COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
SPLIT = "scaffold"


def tanimoto_to_train(smiles: pd.Series, train_mask: np.ndarray,
                      test_mask: np.ndarray) -> np.ndarray:
    """Max Tanimoto from each test molecule to any training molecule.

    This is an applicability-domain signal: it asks "have I seen chemistry like
    this?", which is a question the model can answer honestly even when its
    probability is miscalibrated.
    """
    gen = rfg.GetMorganGenerator(radius=featurize.MORGAN_RADIUS,
                                 fpSize=featurize.MORGAN_N_BITS)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    train_fps = [fps[i] for i in np.flatnonzero(train_mask)]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps))
                     for i in np.flatnonzero(test_mask)])


def risk_coverage(y: np.ndarray, prob: np.ndarray, confidence: np.ndarray) -> list[dict]:
    """Metrics as a function of how much of the test set is kept.

    Molecules are sorted by `confidence` descending; at coverage c the top c
    fraction is kept. Reporting PR-AUC and the kept/rejected positive rates
    alongside accuracy is what makes the imbalance failure visible.
    """
    order = np.argsort(-confidence)
    rows = []
    for cov in COVERAGES:
        keep = order[: max(2, int(round(len(order) * cov)))]
        yk, pk = y[keep], prob[keep]
        rejected = order[len(keep):]
        rows.append({
            "coverage": cov,
            "n_kept": int(len(keep)),
            "roc_auc": float(roc_auc_score(yk, pk)) if len(np.unique(yk)) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(yk, pk)) if len(np.unique(yk)) > 1 else float("nan"),
            "accuracy": float(((pk >= 0.5).astype(int) == yk).mean()),
            "positive_rate_kept": float(yk.mean()),
            "positive_rate_rejected": float(y[rejected].mean()) if len(rejected) else float("nan"),
            "recall_retained": float(yk.sum() / max(y.sum(), 1)),
        })
    return rows


def main() -> int:
    paths.ensure_dirs()
    clean = pd.read_parquet(paths.CLEAN_PARQUET)
    X = featurize.featurize_smiles(clean["canonical_smiles"])
    endpoints = adata.TOX_ENDPOINTS

    rows = []
    for seed in paths.SEEDS:
        subsets = adata.tox21_subsets(SPLIT, seed)
        tr, te = subsets == "train", subsets == "test"
        sim = tanimoto_to_train(clean["canonical_smiles"], tr, te)
        rng = np.random.default_rng(20_000 + seed)
        random_conf = rng.random(int(te.sum()))

        for ep in endpoints:
            y = clean[ep].to_numpy(dtype=float)
            labeled = ~np.isnan(y)
            a = tr & labeled
            model = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                           n_jobs=-1, random_state=seed).fit(
                X[a], y[a].astype(int))

            b = te & labeled
            prob = model.predict_proba(X[b])[:, 1]
            truth = y[b].astype(int)
            if len(np.unique(truth)) < 2:
                continue
            # Per-tree votes give a disagreement estimate without refitting.
            per_tree = np.stack([t.predict_proba(X[b])[:, 1] for t in model.estimators_])
            variance = per_tree.var(axis=0)
            # Align the per-test-molecule signals with this endpoint's labelled subset.
            keep_in_test = labeled[te]
            signals = {
                "distance_from_half": np.abs(prob - 0.5),
                "forest_agreement": -variance,
                "tanimoto_to_train": sim[keep_in_test],
                "random": random_conf[keep_in_test],
            }
            for name, conf in signals.items():
                for row in risk_coverage(truth, prob, conf):
                    rows.append({"seed": seed, "endpoint": ep, "signal": name, **row})

        print(f"  seed {seed} done")

    frame = pd.DataFrame(rows)
    frame.to_csv(paths.RESULTS / "abstention.csv", index=False)

    agg = (frame.groupby(["signal", "coverage"])[
        ["roc_auc", "pr_auc", "accuracy", "positive_rate_kept", "recall_retained"]]
        .mean().reset_index())
    print("\n" + agg.round(4).to_string(index=False))

    # Area under the risk-coverage curve for PR-AUC: higher is better here, since
    # PR-AUC itself is higher-is-better. A good signal keeps PR-AUC up as coverage
    # falls; the naive rule does the opposite.
    aurc = {}
    for signal, g in agg.groupby("signal"):
        g = g.sort_values("coverage")
        aurc[signal] = {
            "mean_pr_auc_over_coverages": float(g["pr_auc"].mean()),
            "pr_auc_at_full_coverage": float(g[g["coverage"] == 1.0]["pr_auc"].iloc[0]),
            "pr_auc_at_50pct_coverage": float(g[g["coverage"] == 0.5]["pr_auc"].iloc[0]),
            "recall_retained_at_50pct": float(
                g[g["coverage"] == 0.5]["recall_retained"].iloc[0]),
            "positive_rate_kept_at_50pct": float(
                g[g["coverage"] == 0.5]["positive_rate_kept"].iloc[0]),
        }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    for signal, g in agg.groupby("signal"):
        g = g.sort_values("coverage")
        axes[0].plot(g["coverage"], g["pr_auc"], marker="o", label=signal)
        axes[1].plot(g["coverage"], g["recall_retained"], marker="o", label=signal)
    axes[0].set_xlabel("coverage (fraction kept)")
    axes[0].set_ylabel("macro PR-AUC on the kept pool")
    axes[0].set_title("Risk-coverage: does abstention help?")
    axes[1].set_xlabel("coverage (fraction kept)")
    axes[1].set_ylabel("fraction of actives retained")
    axes[1].set_title("What abstention throws away")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths.FIGURES / "abstention_risk_coverage.png")
    plt.close(fig)

    best = max(aurc, key=lambda s: aurc[s]["mean_pr_auc_over_coverages"])
    stats = {
        "split": SPLIT,
        "coverages": list(COVERAGES),
        "signals": aurc,
        "best_signal_by_mean_pr_auc": best,
        "beats_random": {
            s: bool(v["mean_pr_auc_over_coverages"]
                    > aurc["random"]["mean_pr_auc_over_coverages"])
            for s, v in aurc.items() if s != "random"
        },
    }
    (paths.RESULTS / "abstention.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"\nbest signal by mean PR-AUC across coverages: {best}")
    print(f"beats random: {stats['beats_random']}")
    print(f"Wrote {paths.RESULTS / 'abstention.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
