"""Phase 3 Task 4 - test-set positive counts, bootstrap CIs, unreliable-endpoint flags.

1. Verifies n_test_pos is populated in results/baseline_metrics.csv and builds a
   per-(dataset, setting, split, endpoint) table of mean/min test positives next to
   the existing mean +/- std ROC-AUC / PR-AUC.
2. Refits the Tox21 random-forest baseline exactly as src/04_baseline_rf.py does,
   checks the reproduced point estimates against baseline_metrics.csv (|dev| < 1e-6
   or the script aborts), then bootstraps each test set 1000 times.
3. Flags endpoints with fewer than 20 test positives (min over seeds).
4. Macro-average CIs per split and a paired CI on random - scaffold.

Bootstrap construction (per split, per seed): rng = np.random.default_rng(10_000 + seed);
B = 1000 index vectors are drawn with replacement over the *test molecules* of that
split/seed (all test rows, labelled or not). Resample b is shared by all 12 endpoints:
each endpoint is scored on the labelled molecules among draw b. A resample in which an
endpoint's labels are single-class is skipped for that endpoint (counted).
Seed-pooled statistic for resample b = mean over the 5 seeds of the per-seed statistic
for resample b; macro statistic = mean over 12 endpoints first. Paired difference for
resample b = random macro_b - scaffold macro_b (same b, same seed RNG). 95% CI =
2.5/97.5 percentiles over b (numpy 'linear' method).

Outputs: results/endpoint_reliability.{csv,json,md}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import baseline, featurize, paths  # noqa: E402

N_BOOT = 1000
BOOT_SEED_OFFSET = 10_000
THRESHOLD = 20
TOL = 1e-6
SPLITS = ("random", "scaffold")
NON_ENDPOINT = ("inchikey", "canonical_smiles", "n_heavy_atoms", "drug_ids", "n_source_rows")


def load_split(kind: str, seed: int, inchikeys: pd.Series) -> np.ndarray:
    path = paths.SPLITS / f"{kind}_split_seed{seed}.csv"
    frame = pd.read_csv(path)
    if len(frame) != len(inchikeys) or not (frame["inchikey"].to_numpy() == inchikeys.to_numpy()).all():
        raise RuntimeError(f"{path} does not match {paths.CLEAN_PARQUET}")
    return frame["subset"].to_numpy()


def ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def r(x: float) -> float:
    return round(float(x), 6)


def main() -> int:
    metrics = pd.read_csv(paths.RESULTS / "baseline_metrics.csv")
    # ---- 1. positive-count check -------------------------------------------------
    n_rows = len(metrics)
    missing = int(metrics["n_test_pos"].isna().sum())
    nonint = int((metrics["n_test_pos"].dropna() % 1 != 0).sum())
    nonpos = int((metrics["n_test_pos"].dropna() <= 0).sum())
    if n_rows != 150 or missing or nonint or nonpos:
        print(f"FAILED: n_test_pos check rows={n_rows} missing={missing} nonint={nonint} <=0={nonpos}")
        return 1
    print(f"n_test_pos populated for all {n_rows} rows")

    keys = ["dataset", "setting", "split", "endpoint"]
    table = (
        metrics.groupby(keys, sort=False)
        .agg(n_seeds=("seed", "nunique"), n_test_mean=("n_test", "mean"),
             n_test_pos_mean=("n_test_pos", "mean"), n_test_pos_min=("n_test_pos", "min"),
             n_test_pos_max=("n_test_pos", "max"),
             roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
             pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"))
        .reset_index()
    )
    table["flag_lt20_pos"] = table["n_test_pos_min"] < THRESHOLD

    # ---- 2. reproduce Tox21 fits and bootstrap ------------------------------------
    clean = pd.read_parquet(paths.CLEAN_PARQUET)
    endpoints = [c for c in clean.columns if c not in NON_ENDPOINT]
    X = featurize.featurize_smiles(clean["canonical_smiles"])
    tox = metrics[metrics["dataset"] == "Tox21"].set_index(["split", "seed", "endpoint"])

    max_dev = 0.0
    fits = []  # per (split, seed, endpoint)
    # boot[split][stat] -> array (n_seeds, n_endpoints, N_BOOT), NaN where skipped
    boot = {s: {m: np.full((len(paths.SEEDS), len(endpoints), N_BOOT), np.nan)
                for m in ("roc_auc", "pr_auc")} for s in SPLITS}
    for split in SPLITS:
        for si, seed in enumerate(paths.SEEDS):
            subsets = load_split(split, seed, clean["inchikey"])
            is_train, is_test = subsets == "train", subsets == "test"
            test_rows = np.flatnonzero(is_test)
            rng = np.random.default_rng(BOOT_SEED_OFFSET + seed)
            draws = rng.integers(0, len(test_rows), size=(N_BOOT, len(test_rows)))
            for ei, ep in enumerate(endpoints):
                y = clean[ep].to_numpy(dtype=float)
                labeled = ~np.isnan(y)
                tr = is_train & labeled
                clf = RandomForestClassifier(n_estimators=baseline.N_ESTIMATORS,
                                             class_weight="balanced", n_jobs=-1,
                                             random_state=seed)
                clf.fit(X[tr], y[tr].astype(int))
                # scores for every test molecule (unlabelled ones dropped per resample)
                s_all = clf.predict_proba(X[test_rows])[:, 1]
                y_all = y[test_rows]
                lab = ~np.isnan(y_all)
                roc = roc_auc_score(y_all[lab].astype(int), s_all[lab])
                pr = average_precision_score(y_all[lab].astype(int), s_all[lab])
                ref = tox.loc[(split, seed, ep)]
                dev = max(abs(roc - ref["roc_auc"]), abs(pr - ref["pr_auc"]))
                max_dev = max(max_dev, dev)
                if dev >= TOL or int(lab.sum()) != ref["n_test"]:
                    print(f"FAILED: reproduction mismatch {split}/{seed}/{ep} dev={dev:.3e}")
                    return 1
                skipped = 0
                for b in range(N_BOOT):
                    idx = draws[b]
                    keep = lab[idx]
                    yb = y_all[idx][keep].astype(int)
                    if yb.min() == yb.max():
                        skipped += 1
                        continue
                    sb = s_all[idx][keep]
                    boot[split]["roc_auc"][si, ei, b] = roc_auc_score(yb, sb)
                    boot[split]["pr_auc"][si, ei, b] = average_precision_score(yb, sb)
                rlo, rhi = ci(boot[split]["roc_auc"][si, ei][~np.isnan(boot[split]["roc_auc"][si, ei])])
                plo, phi = ci(boot[split]["pr_auc"][si, ei][~np.isnan(boot[split]["pr_auc"][si, ei])])
                fits.append({"split": split, "seed": seed, "endpoint": ep,
                             "n_test": int(lab.sum()), "n_test_pos": int(ref["n_test_pos"]),
                             "roc_auc": r(roc), "roc_auc_ci": [r(rlo), r(rhi)],
                             "pr_auc": r(pr), "pr_auc_ci": [r(plo), r(phi)],
                             "n_skipped": skipped, "abs_dev_vs_baseline": float(f"{dev:.3e}")})
            print(f"  {split:<8} seed={seed} done (max dev so far {max_dev:.2e})")
    total_skipped = int(sum(f["n_skipped"] for f in fits))

    # ---- seed-pooled per-endpoint CIs --------------------------------------------
    pooled_rows = []
    for split in SPLITS:
        for ei, ep in enumerate(endpoints):
            row = {"split": split, "endpoint": ep}
            for m in ("roc_auc", "pr_auc"):
                pooled = np.nanmean(boot[split][m][:, ei, :], axis=0)  # mean over seeds per b
                row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = (r(v) for v in ci(pooled))
            row["n_skipped"] = int(np.isnan(boot[split]["roc_auc"][:, ei, :]).sum())
            pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows)
    pooled.insert(0, "dataset", "Tox21")
    table = table.merge(pooled, on=["dataset", "split", "endpoint"], how="left")

    # ---- 4. macro CIs and paired difference ----------------------------------------
    macro = {}
    macro_b = {}
    for split in SPLITS:
        macro[split] = {}
        macro_b[split] = {}
        for m in ("roc_auc", "pr_auc"):
            per_seed_b = np.nanmean(boot[split][m], axis=1)  # (seeds, B), mean over endpoints
            mb = per_seed_b.mean(axis=0)
            macro_b[split][m] = mb
            point = float(tox.loc[split][m].groupby(level="seed").mean().mean())
            lo, hi = ci(mb)
            macro[split][m] = {"point": r(point), "ci": [r(lo), r(hi)]}
    diff = {}
    for m in ("roc_auc", "pr_auc"):
        d = macro_b["random"][m] - macro_b["scaffold"][m]
        lo, hi = ci(d)
        point = macro["random"][m]["point"] - macro["scaffold"][m]["point"]
        disjoint = macro["random"][m]["ci"][0] > macro["scaffold"][m]["ci"][1]
        diff[m] = {"point": r(point), "ci": [r(lo), r(hi)], "ci_excludes_zero": bool(lo > 0),
                   "fraction_resamples_le_0": r(float((d <= 0).mean())),
                   "split_cis_disjoint": bool(disjoint)}
    survives = diff["roc_auc"]["ci_excludes_zero"]

    # ---- outputs -------------------------------------------------------------------
    out_cols = keys + ["n_seeds", "n_test_mean", "n_test_pos_mean", "n_test_pos_min",
                       "n_test_pos_max", "roc_auc_mean", "roc_auc_std", "roc_auc_ci_lo",
                       "roc_auc_ci_hi", "pr_auc_mean", "pr_auc_std", "pr_auc_ci_lo",
                       "pr_auc_ci_hi", "n_skipped", "flag_lt20_pos"]
    table = table[out_cols]
    table["n_skipped"] = table["n_skipped"].astype("Int64")
    table.round(6).to_csv(paths.RESULTS / "endpoint_reliability.csv", index=False)

    flagged = table[table["flag_lt20_pos"]]
    flagged_list = [{k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
                     for k, v in row.items()}
                    for row in flagged[keys + ["n_test_pos_mean", "n_test_pos_min"]]
                    .to_dict("records")]
    payload = {
        "n_test_pos_check": {"rows": n_rows, "missing": missing, "non_integer": nonint,
                             "non_positive": nonpos, "ok": True},
        "reproduction": {"n_fits": len(fits), "max_abs_deviation": float(f"{max_dev:.3e}"),
                         "tolerance": TOL},
        "bootstrap": {"n_resamples": N_BOOT, "rng": "np.random.default_rng(10000 + seed)",
                      "unit": "test molecules of each split/seed, shared across endpoints",
                      "ci": "95% percentile", "total_skipped": total_skipped},
        "threshold_test_pos": THRESHOLD,
        "flagged": flagged_list,
        "macro": macro,
        "difference_random_minus_scaffold": diff,
        "penalty_survives": survives,
        "fits": fits,
    }
    (paths.RESULTS / "endpoint_reliability.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # ---- markdown ------------------------------------------------------------------
    f4 = lambda x: f"{x:.4f}"  # noqa: E731
    L = ["# Endpoint reliability (Phase 3 Task 4)", "",
         "Generated by `src/21_endpoint_reliability.py`; machine-readable siblings "
         "`endpoint_reliability.csv` and `endpoint_reliability.json`.", "",
         "## Checks", "",
         f"- `n_test_pos` populated for all {n_rows}/{n_rows} rows of `baseline_metrics.csv` "
         "(0 missing, 0 non-integer, 0 non-positive).",
         f"- Reproduced all {len(fits)} Tox21 fits; largest |deviation| from "
         f"`baseline_metrics.csv` ROC-AUC/PR-AUC = {max_dev:.3e} (tolerance 1e-6).",
         f"- Bootstrap: {N_BOOT} resamples per split/seed/endpoint; resamples skipped "
         f"because one class was absent: {total_skipped} of {len(fits) * N_BOOT}.", "",
         "## Bootstrap construction", "",
         "For each split and seed, `rng = np.random.default_rng(10000 + seed)` draws 1000 "
         "index vectors with replacement over that split's test molecules. Resample *b* is "
         "shared by all 12 endpoints; each endpoint is scored on the labelled molecules in "
         "draw *b* with the fit's fixed predictions (no refitting). The seed-pooled statistic "
         "for *b* is the mean over the 5 seeds of the per-seed value for *b*; the macro "
         "statistic averages the 12 endpoints within each seed first. The difference for *b* "
         "is random macro_b minus scaffold macro_b, paired by seed and resample index. CIs are "
         "the 2.5th/97.5th percentiles over the 1000 *b*. These CIs capture test-set "
         "sampling variability only, conditional on the fitted models and the 5 splits.", "",
         "## Tox21 per-endpoint table", "",
         "| split | endpoint | test pos mean | test pos min | ROC-AUC mean ± std | ROC-AUC 95% CI | "
         "PR-AUC mean ± std | PR-AUC 95% CI | flag (<20 pos) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, t in table[table["dataset"] == "Tox21"].iterrows():
        L.append(f"| {t.split} | {t.endpoint} | {t.n_test_pos_mean:.1f} | {int(t.n_test_pos_min)} | "
                 f"{f4(t.roc_auc_mean)} ± {f4(t.roc_auc_std)} | [{f4(t.roc_auc_ci_lo)}, {f4(t.roc_auc_ci_hi)}] | "
                 f"{f4(t.pr_auc_mean)} ± {f4(t.pr_auc_std)} | [{f4(t.pr_auc_ci_lo)}, {f4(t.pr_auc_ci_hi)}] | "
                 f"{'FLAG' if t.flag_lt20_pos else ''} |")
    L += ["", "## Other datasets (positive counts only; no bootstrap)", "",
          "| dataset | setting | split | test pos mean | test pos min | ROC-AUC mean ± std | "
          "PR-AUC mean ± std | flag (<20 pos) |", "|---|---|---|---|---|---|---|---|"]
    for _, t in table[table["dataset"] != "Tox21"].iterrows():
        L.append(f"| {t.dataset} | {t.setting} | {t.split} | {t.n_test_pos_mean:.1f} | "
                 f"{int(t.n_test_pos_min)} | {f4(t.roc_auc_mean)} ± {f4(t.roc_auc_std)} | "
                 f"{f4(t.pr_auc_mean)} ± {f4(t.pr_auc_std)} | {'FLAG' if t.flag_lt20_pos else ''} |")
    L += ["", f"## Flagged endpoints (min test positives over seeds < {THRESHOLD})", ""]
    if flagged_list:
        for f in flagged_list:
            L.append(f"- {f['dataset']} / {f['split']} / {f['endpoint']}: min "
                     f"{int(f['n_test_pos_min'])}, mean {f['n_test_pos_mean']:.1f}")
    else:
        L.append("- none")
    L += ["", "**Rule:** endpoints below the threshold of 20 test positives are included in "
          "the macro average but not interpreted individually.", "",
          "## Macro-average CIs (Tox21, 12 endpoints, 5 seeds)", "",
          "| split | ROC-AUC | ROC-AUC 95% CI | PR-AUC | PR-AUC 95% CI |", "|---|---|---|---|---|"]
    for s in SPLITS:
        a, p = macro[s]["roc_auc"], macro[s]["pr_auc"]
        L.append(f"| {s} | {f4(a['point'])} | [{f4(a['ci'][0])}, {f4(a['ci'][1])}] | "
                 f"{f4(p['point'])} | [{f4(p['ci'][0])}, {f4(p['ci'][1])}] |")
    L += ["", "| random − scaffold | point | paired 95% CI | resamples ≤ 0 | split CIs disjoint |",
          "|---|---|---|---|---|"]
    for m in ("roc_auc", "pr_auc"):
        d = diff[m]
        L.append(f"| {m} | {f4(d['point'])} | [{f4(d['ci'][0])}, {f4(d['ci'][1])}] | "
                 f"{d['fraction_resamples_le_0']:.3f} | {'yes' if d['split_cis_disjoint'] else 'no'} |")
    d = diff["roc_auc"]
    verdict = ("**Verdict: the penalty SURVIVES.**" if survives else
               "**Verdict: the penalty does NOT survive.**")
    L += ["", f"{verdict} The random-minus-scaffold macro ROC-AUC difference is "
          f"{f4(d['point'])} with paired 95% CI [{f4(d['ci'][0])}, {f4(d['ci'][1])}], which "
          f"{'excludes' if survives else 'includes'} zero; the two split-level macro CIs are "
          f"{'disjoint' if d['split_cis_disjoint'] else 'overlapping'}.", ""]
    (paths.RESULTS / "endpoint_reliability.md").write_text("\n".join(L), encoding="utf-8")
    print(f"max dev {max_dev:.3e}; skipped {total_skipped}; survives={survives}; diff={diff['roc_auc']}")
    print(f"flagged: {flagged_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
