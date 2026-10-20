"""The shared random-forest classification baseline.

Both the Tox21 grid (``04_baseline_rf.py``) and the hERG external check
(``11_herg_baseline.py``) call ``run_grid`` so the two datasets are measured by
identical code: same features, same forest, same metrics, same seed semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

N_ESTIMATORS = 500

#: Column order of the per-fit metric table (results/baseline_metrics.csv).
METRIC_COLUMNS = [
    "dataset",
    "split",
    "seed",
    "endpoint",
    "n_train",
    "n_train_pos",
    "n_test",
    "n_test_pos",
    "roc_auc",
    "pr_auc",
]


def fit_one(
    X: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
    *,
    n_estimators: int = N_ESTIMATORS,
) -> dict:
    """Train one forest on ``train`` rows and score it on ``test`` rows."""
    y_tr, y_te = y[train].astype(int), y[test].astype(int)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        raise RuntimeError(
            f"single-class fold (train={np.bincount(y_tr).tolist()}, "
            f"test={np.bincount(y_te).tolist()}) - metrics would be undefined"
        )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X[train], y_tr)
    scores = clf.predict_proba(X[test])[:, 1]
    return {
        "n_train": int(train.sum()),
        "n_train_pos": int(y_tr.sum()),
        "n_test": int(test.sum()),
        "n_test_pos": int(y_te.sum()),
        "roc_auc": roc_auc_score(y_te, scores),
        "pr_auc": average_precision_score(y_te, scores),
    }


def run_grid(
    dataset: str,
    X: np.ndarray,
    labels: pd.DataFrame,
    endpoints: Sequence[str],
    splits: Sequence[str],
    seeds: Sequence[int],
    subsets_for: Callable[[str, int], np.ndarray],
    *,
    n_estimators: int = N_ESTIMATORS,
    progress: bool = True,
) -> pd.DataFrame:
    """Fit every (split, seed, endpoint) combination and return the metric table.

    ``subsets_for(split, seed)`` returns the per-row subset labels
    ("train"/"valid"/"test") for that combination, normally read back from the
    committed index files in ``splits/``.
    """
    records: list[dict] = []
    for split in splits:
        for seed in seeds:
            subsets = subsets_for(split, seed)
            is_train, is_test = subsets == "train", subsets == "test"
            for endpoint in endpoints:
                y = labels[endpoint].to_numpy(dtype=float)
                labeled = ~np.isnan(y)
                try:
                    scores = fit_one(
                        X,
                        y,
                        is_train & labeled,
                        is_test & labeled,
                        seed,
                        n_estimators=n_estimators,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"{dataset}/{split}/seed{seed}/{endpoint}: {exc}"
                    ) from exc
                records.append(
                    {"dataset": dataset, "split": split, "seed": seed,
                     "endpoint": endpoint, **scores}
                )
            if progress:
                done = records[-len(endpoints):]
                print(
                    f"  {dataset:<6} {split:<8} seed={seed}  "
                    f"mean ROC-AUC={np.mean([r['roc_auc'] for r in done]):.4f}  "
                    f"mean PR-AUC={np.mean([r['pr_auc'] for r in done]):.4f}"
                )
    return round_metrics(pd.DataFrame.from_records(records)[METRIC_COLUMNS])


#: Decimal places the metric tables are written with. sklearn's forests accumulate
#: per-tree probabilities across threads, so the last bit of a score can differ
#: between otherwise identical runs; rounding well below any reported precision
#: (results are quoted to 4 dp) makes the committed tables byte-reproducible
#: without pretending the 16th digit was ever meaningful.
METRIC_DECIMALS = 10


def round_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    float_cols = [c for c in frame.columns if frame[c].dtype.kind == "f"]
    if float_cols:
        frame = frame.copy()
        frame[float_cols] = frame[float_cols].round(METRIC_DECIMALS)
    return frame


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean/std over seeds, per endpoint and macro-averaged over endpoints.

    The overall rows average the endpoints *within* each seed first, so the
    reported std is seed-to-seed variability (sample std, ddof=1).
    """
    per_endpoint = (
        metrics.groupby(["dataset", "split", "endpoint"])[["roc_auc", "pr_auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_endpoint.columns = [
        "dataset", "split", "endpoint",
        "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std",
    ]
    per_endpoint.insert(3, "level", "endpoint")

    per_seed = (
        metrics.groupby(["dataset", "split", "seed"])[["roc_auc", "pr_auc"]]
        .mean()
        .reset_index()
    )
    overall = (
        per_seed.groupby(["dataset", "split"])[["roc_auc", "pr_auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    overall.columns = [
        "dataset", "split", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std"
    ]
    n_endpoints = metrics["endpoint"].nunique()
    overall.insert(2, "endpoint", f"ALL_{n_endpoints}_MACRO_AVG")
    overall.insert(3, "level", "overall")

    return round_metrics(pd.concat([per_endpoint, overall], ignore_index=True))
