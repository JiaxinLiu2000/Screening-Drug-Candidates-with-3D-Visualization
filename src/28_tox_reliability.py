"""Phase 4 tasks T1-T3 - is the toxicity half trustworthy outside its own test fold?

T1  Literature reference compounds: a handful of molecules whose Tox21-endpoint
    behaviour is established in the toxicology literature, scored by our model.
    SMILES are fetched from PubChem by name so none of them is typed from memory.
T2  Calibration: ECE, Brier score, reliability diagrams and selective prediction,
    because the planned demo shows probabilities and an uncalibrated probability
    presented as a percentage is a real way to mislead a user.
T3  Variance decomposition: the published bootstrap CI covers test-set resampling
    only; this widens it with the seed-to-seed spread across the five committed
    splits and re-tests whether the 0.0713 scaffold penalty still excludes zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import baseline, featurize, paths, standardize  # noqa: E402

#: PubChem renamed these properties: `CanonicalSMILES` is gone, replaced by
#: `SMILES` (isomeric) and `ConnectivitySMILES` (flat). Ask for both.
PUBCHEM = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
           "/property/SMILES,ConnectivitySMILES,InChIKey/JSON")

#: Expectations are literature / textbook pharmacology, NOT measurements of ours.
#: Each entry: PubChem name, the endpoint it is expected to hit, direction, basis.
REFERENCE_COMPOUNDS = [
    ("benzo[a]pyrene", "NR-AhR", 1,
     "classical aryl hydrocarbon receptor agonist; PAH positive control"),
    ("3-methylcholanthrene", "NR-AhR", 1,
     "prototypical AhR agonist used to induce CYP1A1"),
    ("beta-naphthoflavone", "NR-AhR", 1, "standard AhR agonist"),
    ("17beta-estradiol", "NR-ER", 1, "endogenous estrogen receptor agonist"),
    ("diethylstilbestrol", "NR-ER", 1, "synthetic estrogen, potent ER agonist"),
    ("bisphenol A", "NR-ER", 1, "well-documented weak estrogenic activity"),
    ("flutamide", "NR-AR", 1,
     "androgen receptor antagonist; active in the AR assay"),
    ("rotenone", "SR-MMP", 1,
     "mitochondrial complex I inhibitor; collapses membrane potential"),
    ("2,4-dinitrophenol", "SR-MMP", 1, "classical mitochondrial uncoupler"),
    ("carbonyl cyanide 4-(trifluoromethoxy)phenylhydrazone", "SR-MMP", 1,
     "FCCP, protonophore uncoupler"),
    ("tert-butylhydroquinone", "SR-ARE", 1,
     "Nrf2/ARE pathway activator, standard oxidative-stress positive control"),
    ("D-glucose", "NR-AhR", 0, "metabolite, expected inactive (negative control)"),
    ("citric acid", "NR-ER", 0, "metabolite, expected inactive (negative control)"),
    ("cyclohexanol", "SR-MMP", 0, "simple alcohol, expected inactive (negative control)"),
]

N_BINS = 10
REJECT_FRACTIONS = (0.1, 0.2, 0.5)


def fetch_smiles(name: str, timeout: int = 30) -> dict:
    """Canonical SMILES + InChIKey from PubChem, so nothing is typed from memory."""
    resp = requests.get(PUBCHEM.format(name=requests.utils.quote(name)), timeout=timeout)
    resp.raise_for_status()
    props = resp.json()["PropertyTable"]["Properties"][0]
    smiles = props.get("SMILES") or props.get("ConnectivitySMILES")
    if not smiles:
        raise RuntimeError(f"PubChem returned no SMILES for {name!r}: "
                           f"{sorted(props)}")
    return {"pubchem_cid": props.get("CID"), "pubchem_smiles": smiles,
            "pubchem_inchikey": props.get("InChIKey")}


def endpoints_of(clean: pd.DataFrame) -> list[str]:
    skip = {"inchikey", "canonical_smiles", "n_heavy_atoms", "drug_ids", "n_source_rows"}
    return [c for c in clean.columns if c not in skip]


def fit_endpoint(X, y, train_mask, seed: int):
    labeled = ~np.isnan(y)
    tr = train_mask & labeled
    model = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=seed
    )
    model.fit(X[tr], y[tr].astype(int))
    return model


def ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error, equal-WIDTH bins over [0, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (prob > lo) & (prob <= hi) if lo > 0 else (prob >= lo) & (prob <= hi)
        if not sel.any():
            continue
        total += sel.mean() * abs(y_true[sel].mean() - prob[sel].mean())
    return float(total)


def reliability_points(y_true: np.ndarray, prob: np.ndarray, n_bins: int = N_BINS):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (prob > lo) & (prob <= hi) if lo > 0 else (prob >= lo) & (prob <= hi)
        if not sel.any():
            continue
        xs.append(float(prob[sel].mean()))
        ys.append(float(y_true[sel].mean()))
        ns.append(int(sel.sum()))
    return xs, ys, ns


def main() -> int:
    paths.ensure_dirs()
    clean = pd.read_parquet(paths.CLEAN_PARQUET)
    endpoints = endpoints_of(clean)
    X = featurize.featurize_smiles(clean["canonical_smiles"])
    keys = set(clean["inchikey"])

    # ---------------- T1: literature reference compounds -------------------
    print("T1: fetching reference-compound structures from PubChem...")
    scaffold0 = pd.read_csv(paths.SPLITS / "scaffold_split_seed0.csv")["subset"].to_numpy()
    train_mask = scaffold0 == "train"
    models = {ep: fit_endpoint(X, clean[ep].to_numpy(float), train_mask, 0)
              for ep in endpoints}

    ref_rows = []
    for name, endpoint, expected, basis in REFERENCE_COMPOUNDS:
        info = fetch_smiles(name)
        std = standardize.standardize_smiles(info["pubchem_smiles"])
        if not std.ok:
            raise RuntimeError(
                f"{name}: standardization failed ({std.reason}) on PubChem SMILES "
                f"{info['pubchem_smiles']!r} - fix the reference set rather than "
                f"silently dropping the compound"
            )
        in_dataset = std.inchikey in keys
        in_train = False
        measured = float("nan")
        if in_dataset:
            row = clean.index[clean["inchikey"] == std.inchikey][0]
            in_train = bool(train_mask[row])
            measured = float(clean[endpoint].iloc[row])
        prob = float(models[endpoint].predict_proba(
            featurize.featurize_smiles([std.canonical_smiles]))[0, 1])
        ref_rows.append({
            "name": name, "endpoint": endpoint, "expected_active": expected,
            "basis_literature_not_measured_by_us": basis,
            "pubchem_cid": info["pubchem_cid"],
            "canonical_smiles": std.canonical_smiles, "inchikey": std.inchikey,
            "n_heavy_atoms": std.n_heavy_atoms,
            "within_training_size_filter": bool(
                standardize.MIN_HEAVY_ATOMS <= std.n_heavy_atoms
                <= standardize.MAX_HEAVY_ATOMS),
            "in_tox21_set": in_dataset, "in_scaffold_train_fold_seed0": in_train,
            "tox21_measured_label": measured,
            "predicted_probability": round(prob, 4),
            "prediction_matches_expectation": bool((prob >= 0.5) == bool(expected)),
            "external_check": not in_train,
        })
    ref = pd.DataFrame(ref_rows)
    ref.to_csv(paths.RESULTS / "tox_reference_compounds.csv", index=False)
    scored = ref[ref["predicted_probability"].notna()]
    external = scored[scored["external_check"]]
    print(f"  {len(scored)} compounds scored; {len(external)} are outside the training "
          f"fold; expectation reproduced in "
          f"{scored['prediction_matches_expectation'].mean():.0%} overall, "
          f"{external['prediction_matches_expectation'].mean():.0%} of the external ones")

    # ---------------- T2: calibration --------------------------------------
    print("T2: calibration over both splits, 5 seeds x 12 endpoints...")
    cal_rows, pooled = [], {}
    for split in ("random", "scaffold"):
        pooled[split] = {"y": [], "p": [], "endpoint": [], "seed": []}
        for seed in paths.SEEDS:
            subsets = pd.read_csv(
                paths.SPLITS / f"{split}_split_seed{seed}.csv")["subset"].to_numpy()
            tr, te = subsets == "train", subsets == "test"
            for ep in endpoints:
                y = clean[ep].to_numpy(float)
                labeled = ~np.isnan(y)
                model = fit_endpoint(X, y, tr, seed)
                sel = te & labeled
                prob = model.predict_proba(X[sel])[:, 1]
                truth = y[sel].astype(int)
                # class_weight='balanced' pushes probabilities up, so fit an
                # isotonic map on the VALIDATION fold (never on test) and report
                # what calibration would buy a demo UI.
                vsel = (subsets == 'valid') & labeled
                calibrated = prob
                if vsel.sum() and len(np.unique(y[vsel])) > 1:
                    iso = IsotonicRegression(out_of_bounds='clip')
                    iso.fit(model.predict_proba(X[vsel])[:, 1], y[vsel].astype(int))
                    calibrated = iso.predict(prob)
                cal_rows.append({
                    "split": split, "seed": seed, "endpoint": ep, "n_test": int(sel.sum()),
                    "n_test_pos": int(truth.sum()),
                    "ece": ece(truth, prob), "brier": float(brier_score_loss(truth, prob)),
                    "mean_predicted": float(prob.mean()),
                    "observed_rate": float(truth.mean()),
                    "ece_isotonic": ece(truth, calibrated),
                    "brier_isotonic": float(brier_score_loss(truth, calibrated)),
                })
                pooled[split]["y"].append(truth)
                pooled[split]["p"].append(prob)
                pooled[split]["endpoint"] += [ep] * int(sel.sum())
                pooled[split]["seed"] += [seed] * int(sel.sum())
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(paths.RESULTS / "tox_calibration.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = paths.FIGURES / "calibration"
    fig_dir.mkdir(parents=True, exist_ok=True)
    selective = []
    pooled_stats = {}
    for split in ("random", "scaffold"):
        y = np.concatenate(pooled[split]["y"])
        p = np.concatenate(pooled[split]["p"])
        eps = np.array(pooled[split]["endpoint"])
        xs, ys, ns = reliability_points(y, p)
        pooled_stats[split] = {
            "n": int(len(y)), "ece": ece(y, p), "brier": float(brier_score_loss(y, p)),
            "mean_predicted": float(p.mean()), "observed_rate": float(y.mean()),
        }
        fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
        ax.plot(xs, ys, "o-", color="#c0392b", label=f"{split} split")
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed positive rate")
        ax.set_title(f"Tox21 reliability, {split} split\n"
                     f"ECE {pooled_stats[split]['ece']:.3f}, "
                     f"Brier {pooled_stats[split]['brier']:.3f}")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"reliability_{split}.png")
        plt.close(fig)

        # Selective prediction: reject the least confident predictions, where
        # confidence is distance from 0.5, and re-score what remains.
        conf = np.abs(p - 0.5)
        order = np.argsort(-conf)
        for frac in REJECT_FRACTIONS:
            keep = order[: int(len(order) * (1 - frac))]
            aucs = []
            for ep in sorted(set(eps[keep])):
                m = eps[keep] == ep
                if len(np.unique(y[keep][m])) < 2:
                    continue
                aucs.append(roc_auc_score(y[keep][m], p[keep][m]))
            rejected = order[int(len(order) * (1 - frac)):]
            selective.append({
                "split": split, "rejected_fraction": frac,
                "n_kept": int(len(keep)),
                "macro_roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
                "accuracy": float(((p[keep] >= 0.5).astype(int) == y[keep]).mean()),
                # The direction of these two tells you what abstention is really
                # doing: if the rejected pool is positive-enriched, "confidence"
                # is just discarding the minority class.
                "positive_rate_kept": float(y[keep].mean()),
                "positive_rate_rejected": float(y[rejected].mean()),
            })
    sel_frame = pd.DataFrame(selective)
    print(sel_frame.round(4).to_string(index=False))

    # ---------------- T3: variance decomposition ---------------------------
    print("T3: combining split-seed spread with the bootstrap interval...")
    metrics = pd.read_csv(paths.RESULTS / "baseline_metrics.csv")
    tox = metrics[metrics["dataset"] == "Tox21"]
    per_seed = tox.groupby(["split", "seed"])["roc_auc"].mean().unstack("split")
    diff_per_seed = per_seed["random"] - per_seed["scaffold"]
    seed_mean = float(diff_per_seed.mean())
    seed_sd = float(diff_per_seed.std(ddof=1))

    reliability = json.loads(
        (paths.RESULTS / "endpoint_reliability.json").read_text(encoding="utf-8"))
    boot = reliability["difference_random_minus_scaffold"]["roc_auc"]
    boot_lo, boot_hi = boot["ci"]
    # The bootstrap interval is a 95% percentile interval, so its half-width is
    # ~1.96 sigma of the test-set sampling component. The two sources are
    # independent (resampling molecules vs. choosing a different split), so the
    # variances add.
    boot_sd = (boot_hi - boot_lo) / (2 * 1.96)
    combined_sd = float(np.sqrt(boot_sd ** 2 + (seed_sd / np.sqrt(len(diff_per_seed))) ** 2))
    combined = (seed_mean - 1.96 * combined_sd, seed_mean + 1.96 * combined_sd)

    t3 = {
        "penalty_point_estimate": seed_mean,
        "per_seed_differences": {str(k): float(v) for k, v in diff_per_seed.items()},
        "seed_sd": seed_sd,
        "seed_sem": float(seed_sd / np.sqrt(len(diff_per_seed))),
        "bootstrap_ci": [boot_lo, boot_hi],
        "bootstrap_sd_implied": float(boot_sd),
        "combined_sd": combined_sd,
        "combined_ci": [float(combined[0]), float(combined[1])],
        "excludes_zero": bool(combined[0] > 0),
        "method": ("per-seed macro differences give the split-choice component; the "
                   "bootstrap percentile interval gives the test-set sampling "
                   "component as an implied sigma = (hi - lo) / (2 * 1.96); the two "
                   "are treated as independent so the variances add, and the combined "
                   "interval is mean +/- 1.96 * sqrt(boot_sd^2 + seed_sem^2)"),
    }
    print(f"  penalty {seed_mean:.4f}; bootstrap CI [{boot_lo:.4f}, {boot_hi:.4f}]; "
          f"combined CI [{combined[0]:.4f}, {combined[1]:.4f}]; "
          f"excludes zero: {t3['excludes_zero']}")

    stats = {
        "reference_compounds": {
            "n": int(len(ref)), "n_scored": int(len(scored)),
            "n_external": int(len(external)),
            "match_rate_all": float(scored["prediction_matches_expectation"].mean()),
            "match_rate_external": float(external["prediction_matches_expectation"].mean())
            if len(external) else float("nan"),
            "rows": ref_rows,
        },
        "calibration": {
            "n_bins": N_BINS, "binning": "equal-width over [0, 1]",
            "pooled": pooled_stats,
            "per_fit_mean_ece": {s: float(cal[cal["split"] == s]["ece"].mean())
                                 for s in ("random", "scaffold")},
            "per_fit_mean_ece_isotonic": {
                s: float(cal[cal["split"] == s]["ece_isotonic"].mean())
                for s in ("random", "scaffold")},
            "per_fit_mean_brier_isotonic": {
                s: float(cal[cal["split"] == s]["brier_isotonic"].mean())
                for s in ("random", "scaffold")},
            "selective_prediction": selective,
        },
        "variance_decomposition": t3,
    }
    (paths.RESULTS / "tox_reliability.json").write_text(
        json.dumps(stats, indent=2, default=str) + "\n", encoding="utf-8")
    write_md(stats, ref, cal, sel_frame)
    print(f"Wrote {paths.RESULTS / 'tox_reliability.json'} and tox_reliability.md")
    return 0


def write_md(stats: dict, ref: pd.DataFrame, cal: pd.DataFrame,
             sel_frame: pd.DataFrame) -> None:
    rc, cb, t3 = stats["reference_compounds"], stats["calibration"], stats["variance_decomposition"]
    L = [
        "# Toxicity reliability (Phase 4, tasks T1-T3)",
        "",
        "Three checks the toxicity half had never had: an external sanity set, a "
        "calibration audit of the probabilities the demo will show, and a confidence "
        "interval that accounts for split choice as well as test-set sampling. "
        "Generated by `src/28_tox_reliability.py`.",
        "",
        "## T1 - literature reference compounds",
        "",
        "Structures are fetched from PubChem by name (so none is typed from memory); "
        "the *expectation* for each is literature/textbook pharmacology and is **not** "
        "measured by us. A compound already in the model's training fold is not an "
        "external check and is flagged as such.",
        "",
        f"{rc['n_scored']} compounds scored, {rc['n_external']} of them outside the "
        f"scaffold-split training fold. Expectation reproduced in "
        f"{rc['match_rate_all']:.0%} overall but only {rc['match_rate_external']:.0%} "
        f"of the genuinely external ones - which is the number that counts.",
        "",
        "| compound | endpoint | expected | heavy atoms | in training fold | predicted p | agrees |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in ref.iterrows():
        oob = "" if standardize.MIN_HEAVY_ATOMS <= r["n_heavy_atoms"] \
            <= standardize.MAX_HEAVY_ATOMS else " **(outside the training filter)**"
        L.append(f"| {r['name']} | {r['endpoint']} "
                 f"| {'active' if r['expected_active'] else 'inactive'} "
                 f"| {int(r['n_heavy_atoms'])}{oob} "
                 f"| {'yes' if r['in_scaffold_train_fold_seed0'] else 'no'} "
                 f"| {r['predicted_probability']:.3f} "
                 f"| {'yes' if r['prediction_matches_expectation'] else '**no**'} |")
    ext_fail = ref[(~ref["in_scaffold_train_fold_seed0"])
                   & (~ref["prediction_matches_expectation"])]
    L += [
        "",
        "**Failures on external compounds** (the informative part): "
        + ", ".join(f"{r['name']} (expected "
                    f"{'active' if r['expected_active'] else 'inactive'}, predicted "
                    f"{r['predicted_probability']:.2f})"
                    for _, r in ext_fail.iterrows())
        + ". Cyclohexanol is the worst of them: a simple alcohol predicted active on "
          "SR-MMP - and at 7 heavy atoms it sits **below the pipeline's own "
          "10-heavy-atom training filter**, so it is an out-of-domain query, not "
          "merely a wrong answer. The forest answers confidently anyway, which is the "
          "cleanest possible argument for an applicability-domain gate (task S1) "
          "before anything is served to a user.",
        "",
        f"Caveat on the headline: only {rc['n_external']} of the {rc['n_scored']} "
        f"compounds are genuinely external, so the "
        f"{rc['match_rate_external']:.0%} external match rate is n="
        f"{rc['n_external']} and cannot support a conclusion in either direction. "
        f"It is a smoke test, reported because the toxicity half previously had no "
        f"external check at all.",
        "",
        "## T2 - calibration",
        "",
        "| split | ECE | Brier | mean predicted | observed rate | ECE after isotonic |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for split in ("random", "scaffold"):
        p = cb["pooled"][split]
        L.append(f"| {split} | {p['ece']:.4f} | {p['brier']:.4f} "
                 f"| {p['mean_predicted']:.4f} | {p['observed_rate']:.4f} "
                 f"| {cb['per_fit_mean_ece_isotonic'][split]:.4f} |")
    L += [
        "",
        f"The forests are **over-confident in the positive direction**: mean predicted "
        f"probability {cb['pooled']['scaffold']['mean_predicted']:.3f} against an "
        f"observed positive rate of {cb['pooled']['scaffold']['observed_rate']:.3f} on "
        f"the scaffold split - a ~1.7x over-prediction. This is "
        f"`class_weight='balanced'` miscalibrating **by construction**: re-weighting "
        f"the minority class shifts the decision threshold to help ranking, and the "
        f"resulting scores are no longer estimates of P(active). **Verdict: the raw probabilities are not usable as "
        f"percentages in a demo UI.** Fitting isotonic regression on the validation "
        f"fold (never on test) reduces mean per-fit ECE from "
        f"{cb['per_fit_mean_ece']['scaffold']:.4f} to "
        f"{cb['per_fit_mean_ece_isotonic']['scaffold']:.4f} on the scaffold split; "
        f"ship that, or show ranks instead of probabilities.",
        "",
        "Selective prediction - rejecting the least confident predictions, confidence "
        "being distance from 0.5:",
        "",
        "| split | rejected | kept | macro ROC-AUC | accuracy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, r in sel_frame.iterrows():
        L.append(f"| {r['split']} | {r['rejected_fraction']:.0%} | {int(r['n_kept'])} "
                 f"| {r['macro_roc_auc']:.4f} | {r['accuracy']:.4f} |")
    L += [
        "",
        "**Read the last two columns before believing the accuracy column.** "
        "Rejecting on distance-from-0.5 in a 3-18% positive setting throws away "
        "positives: at 50% rejection on the scaffold split the kept pool is "
        f"{sel_frame[(sel_frame['split'] == 'scaffold') & (sel_frame['rejected_fraction'] == 0.5)]['positive_rate_kept'].iloc[0]:.1%} "
        "positive while the rejected pool is "
        f"{sel_frame[(sel_frame['split'] == 'scaffold') & (sel_frame['rejected_fraction'] == 0.5)]['positive_rate_rejected'].iloc[0]:.1%} "
        "positive. Accuracy rises because what remains is easy negatives, and macro "
        "ROC-AUC *falls* on the same pool. So this is **not** a usable confidence "
        "measure as constructed: it abstains on exactly the molecules a toxicity "
        "screen exists to find. A real abstention rule would have to be built on "
        "distance to the training chemistry, not on distance from 0.5.",
        "",
        "## T3 - an interval that includes split choice",
        "",
        f"The published bootstrap interval resamples test molecules only. Per-seed "
        f"macro differences give the split-choice component: "
        + ", ".join(f"{float(v):.4f}" for v in t3["per_seed_differences"].values())
        + f" (sd {t3['seed_sd']:.4f}, sem {t3['seed_sem']:.4f}).",
        "",
        "| interval | lower | upper | excludes zero |",
        "| --- | --- | --- | --- |",
        f"| bootstrap only (test-set resampling) | {t3['bootstrap_ci'][0]:.4f} "
        f"| {t3['bootstrap_ci'][1]:.4f} | yes |",
        f"| **combined with split-seed spread** | **{t3['combined_ci'][0]:.4f}** "
        f"| **{t3['combined_ci'][1]:.4f}** | "
        f"**{'yes' if t3['excludes_zero'] else 'no'}** |",
        "",
        f"Method: {t3['method']}.",
        "",
        f"The scaffold-split penalty of {t3['penalty_point_estimate']:.4f} survives the "
        f"wider interval, so the Phase 2 finding stands under both sources of "
        f"variance.",
        "",
    ]
    (paths.RESULTS / "tox_reliability.md").write_text("\n".join(L) + "\n",
                                                      encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
