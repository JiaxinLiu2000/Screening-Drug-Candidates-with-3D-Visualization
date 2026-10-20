"""Phase 3 Task 6 - atom-level attribution for the toxicity model.

Masks one heavy atom at a time in the trained configuration-A encoder, measures
the drop in predicted probability, and asks whether the signal lands on known
structural alerts. Writes results/attribution.md plus annotated figures; the
failure cases are reported as prominently as the successes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tox21lib import ablation_data as adata  # noqa: E402
from tox21lib import attribution, models, paths  # noqa: E402

CHECKPOINT = paths.DATA_PROCESSED / "checkpoints" / "configA_seed0.pt"
FIG_DIR = paths.FIGURES / "attribution"
N_EXAMPLES = 6
TOP_K = 3
DIFFUSE_THRESHOLD = 0.95  # normalised entropy above this = no localised signal
N_NULL_DRAWS = 20         # random same-size atom sets per molecule (control)


def load_model(device: str):
    if not CHECKPOINT.exists():
        raise SystemExit(
            f"FAILED: {CHECKPOINT} missing - run src/20_shared_encoder_ablation.py first"
        )
    blob = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = models.MultiTaskModel(
        models.GINEncoder(), n_endpoints=blob["n_endpoints"], protein_dim=None
    )
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval(), blob


def draw(smiles: str, values: np.ndarray, alert_idx: list[int], path: Path,
         title: str) -> None:
    """Molecule drawing with attribution-coloured atoms and the alert ringed."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    scale = float(np.abs(values).max()) or 1.0
    colors, radii = {}, {}
    for i, v in enumerate(values):
        strength = float(v) / scale
        if strength > 0:                       # pushed the prediction up
            colors[i] = (1.0, 1.0 - 0.75 * strength, 1.0 - 0.75 * strength)
        else:                                  # pushed it down
            colors[i] = (1.0 + 0.75 * strength, 1.0 + 0.75 * strength, 1.0)
        radii[i] = 0.35
    for i in alert_idx:                        # ring the alerting substructure
        radii[i] = 0.55

    drawer = rdMolDraw2D.MolDraw2DCairo(520, 400)
    opts = drawer.drawOptions()
    opts.legendFontSize = 18
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, legend=title,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors=colors,
        highlightAtomRadii=radii,
    )
    drawer.FinishDrawing()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(drawer.GetDrawingText())


def main() -> int:
    paths.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, blob = load_model(device)
    print(f"loaded {CHECKPOINT.name} (config {blob['config']}, "
          f"{blob['encoder']}, tox split {blob['tox_split']}) on {device}")

    clean = pd.read_parquet(paths.CLEAN_PARQUET)
    subsets = adata.tox21_subsets(blob["tox_split"], blob["seed"])
    test_idx = np.flatnonzero(subsets == "test")
    print(f"{len(test_idx)} test molecules under the "
          f"{blob['tox_split']} split, seed {blob['seed']}")

    endpoints = adata.TOX_ENDPOINTS
    records: list[dict] = []
    for i in test_idx:
        smiles = clean["canonical_smiles"].iloc[i]
        alerts = attribution.alert_atoms(smiles)
        if not alerts:
            continue
        result = attribution.attribute(model, smiles, device=device)
        # Attribute against the endpoint the model is most confident about, so the
        # question is "where does THIS prediction come from", not an average over
        # endpoints the molecule may not even be tested on.
        ep = int(np.argmax(result["baseline_probs"]))
        values = result["attribution"][:, ep]
        alert_idx = sorted({a for idx in alerts.values() for a in idx})
        conc = attribution.concentration(values, alert_idx)
        top = attribution.top_atoms(values, TOP_K)
        records.append(
            {
                "row": int(i),
                "inchikey": clean["inchikey"].iloc[i],
                "smiles": smiles,
                "alerts": ",".join(sorted(alerts)),
                "n_alert_atoms": len(alert_idx),
                "n_atoms": result["n_atoms"],
                "endpoint": endpoints[ep],
                "baseline_prob": float(result["baseline_probs"][ep]),
                "label": float(clean[endpoints[ep]].iloc[i]),
                "fraction_of_mass_on_alert": conc["fraction_of_mass"],
                "enrichment": conc["enrichment"],
                "mean_on_alert": conc["mean_on_subset"],
                "mean_off_alert": conc["mean_off_subset"],
                "top_atoms": json.dumps(top),
                "top_hits_alert": bool(set(top) & set(alert_idx)),
                "diffuseness": attribution.diffuseness(values),
                "max_attribution": float(np.max(values)),
                "alert_atom_indices": json.dumps(alert_idx),
            }
        )
    frame = pd.DataFrame(records).sort_values(
        ["enrichment", "inchikey"], ascending=[False, True], ignore_index=True
    )
    frame.to_csv(paths.RESULTS / "attribution_per_molecule.csv", index=False)
    print(f"{len(frame)} test molecules carry at least one structural alert")

    hit_rate = float(frame["top_hits_alert"].mean())
    enriched = float((frame["enrichment"] > 1.0).mean())
    diffuse = frame[frame["diffuseness"] > DIFFUSE_THRESHOLD]
    off_target = frame[(~frame["top_hits_alert"]) & (frame["enrichment"] < 0.5)]

    # ---- control: is a low per-atom enrichment the model's fault or the probe's?
    # Masking one atom at a time understates a substructure whose signal spans
    # several atoms. Masking the whole alert at once, against seeded random atom
    # sets of the same size drawn from outside it, separates "the model ignores
    # the alert" from "single-atom masking cannot see it".
    control_rows: list[dict] = []
    for _, row in frame.iterrows():
        alert_idx = json.loads(row["alert_atom_indices"])
        n_atoms = int(row["n_atoms"])
        nulls = attribution.random_groups(
            n_atoms, len(alert_idx), N_NULL_DRAWS, seed=paths.GLOBAL_SEED + int(row["row"]),
            exclude=alert_idx,
        )
        if not nulls:
            continue
        drops = attribution.mask_group(model, row["smiles"], [alert_idx] + nulls,
                                       device=device)
        ep = endpoints.index(row["endpoint"])
        alert_drop = float(drops[0, ep])
        null_drops = drops[1:, ep]
        control_rows.append(
            {
                "inchikey": row["inchikey"],
                "alerts": row["alerts"],
                "endpoint": row["endpoint"],
                "n_alert_atoms": len(alert_idx),
                "alert_group_drop": alert_drop,
                "null_group_drop_mean": float(null_drops.mean()),
                "null_group_drop_std": float(null_drops.std(ddof=1)),
                "percentile_vs_null": float((null_drops < alert_drop).mean()),
                "beats_null": bool(alert_drop > null_drops.mean()),
            }
        )
    control = pd.DataFrame(control_rows)
    control.to_csv(paths.RESULTS / "attribution_control.csv", index=False)
    control_by_alert = (
        control.assign(alert=control["alerts"].str.split(","))
        .explode("alert")
        .groupby("alert")
        .agg(n=("inchikey", "size"),
             median_percentile=("percentile_vs_null", "median"),
             beats_null_rate=("beats_null", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    print(f"group-masking control: alert beats the random-group null in "
          f"{control['beats_null'].mean():.1%} of {len(control)} molecules "
          f"(median percentile {control['percentile_vs_null'].median():.2f})")

    # Successes: strongest enrichment with a confident prediction.
    wins = frame[(frame["baseline_prob"] > 0.5) & frame["top_hits_alert"]].head(N_EXAMPLES)
    if len(wins) < N_EXAMPLES:
        wins = frame[frame["top_hits_alert"]].head(N_EXAMPLES)
    figures = []
    for rank, (_, row) in enumerate(wins.iterrows(), 1):
        result = attribution.attribute(model, row["smiles"], device=device)
        ep = endpoints.index(row["endpoint"])
        path = FIG_DIR / f"example_{rank:02d}_{row['inchikey'][:14]}.png"
        draw(row["smiles"], result["attribution"][:, ep],
             json.loads(row["alert_atom_indices"]), path,
             f"{row['endpoint']}  p={row['baseline_prob']:.2f}  "
             f"alert: {row['alerts']}  enrichment {row['enrichment']:.2f}x")
        figures.append({"kind": "success", "path": str(path.relative_to(paths.ROOT)),
                        "inchikey": row["inchikey"], "alerts": row["alerts"],
                        "enrichment": row["enrichment"]})

    # Failures: diffuse attribution, or mass concentrated away from the alert.
    fails = pd.concat([diffuse.head(1), off_target.head(2)]).drop_duplicates("inchikey")
    for rank, (_, row) in enumerate(fails.iterrows(), 1):
        result = attribution.attribute(model, row["smiles"], device=device)
        ep = endpoints.index(row["endpoint"])
        path = FIG_DIR / f"failure_{rank:02d}_{row['inchikey'][:14]}.png"
        draw(row["smiles"], result["attribution"][:, ep],
             json.loads(row["alert_atom_indices"]), path,
             f"FAILURE  {row['endpoint']}  p={row['baseline_prob']:.2f}  "
             f"diffuseness {row['diffuseness']:.2f}  enrichment {row['enrichment']:.2f}x")
        figures.append({"kind": "failure", "path": str(path.relative_to(paths.ROOT)),
                        "inchikey": row["inchikey"], "alerts": row["alerts"],
                        "enrichment": row["enrichment"]})

    by_alert = (
        frame.assign(alert=frame["alerts"].str.split(","))
        .explode("alert")
        .groupby("alert")
        .agg(n=("inchikey", "size"), median_enrichment=("enrichment", "median"),
             top_hit_rate=("top_hits_alert", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    per_atom_by_alert = dict(
        zip(by_alert["alert"], by_alert["median_enrichment"])
    )

    stats = {
        "checkpoint": str(CHECKPOINT.relative_to(paths.ROOT)),
        "config": blob["config"],
        "tox_split": blob["tox_split"],
        "seed": blob["seed"],
        "n_test_molecules": int(len(test_idx)),
        "n_with_alerts": int(len(frame)),
        "top_k": TOP_K,
        "top_hits_alert_rate": hit_rate,
        "fraction_enriched_above_1x": enriched,
        "median_enrichment": float(frame["enrichment"].median()),
        "median_fraction_of_mass_on_alert": float(frame["fraction_of_mass_on_alert"].median()),
        "n_diffuse": int(len(diffuse)),
        "diffuse_threshold": DIFFUSE_THRESHOLD,
        "n_off_target": int(len(off_target)),
        "per_alert": by_alert.to_dict(orient="records"),
        "control": {
            "n_null_draws": N_NULL_DRAWS,
            "n_molecules": int(len(control)),
            "beats_null_rate": float(control["beats_null"].mean()),
            "median_percentile_vs_null": float(control["percentile_vs_null"].median()),
            "per_alert": control_by_alert.to_dict(orient="records"),
        },
        "figures": figures,
    }
    (paths.RESULTS / "attribution.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )

    L = [
        "# Atom-level attribution for the toxicity model",
        "",
        f"Method: for every heavy atom, its feature row is replaced with the "
        f"unknown-atom vector (all features zero) and inference is re-run; the "
        f"attribution is the drop in predicted probability. One vector per molecule "
        f"per endpoint, ordered to match the canonical-SMILES atom order. Model: "
        f"configuration A (GIN from scratch, toxicity only), seed {blob['seed']}, "
        f"evaluated on the {blob['tox_split']}-split test fold. Generated by "
        f"`src/22_attribution.py`.",
        "",
        "There is no ground truth for atom attribution, so this is validated "
        "qualitatively: does the signal concentrate on substructures that "
        "toxicologists already treat as alerts?",
        "",
        "## Coverage",
        "",
        "| quantity | value |",
        "| --- | --- |",
        f"| test molecules | {len(test_idx)} |",
        f"| with at least one structural alert | {len(frame)} |",
        f"| top-{TOP_K} atoms include an alert atom | {hit_rate:.1%} of them |",
        f"| attribution enriched on the alert (>1x uniform) | {enriched:.1%} |",
        f"| median enrichment | {frame['enrichment'].median():.2f}x |",
        f"| median share of positive attribution on alert atoms | "
        f"{frame['fraction_of_mass_on_alert'].median():.1%} |",
        "",
        "## By alert type",
        "",
        "| alert | molecules | median enrichment | top-k hit rate |",
        "| --- | --- | --- | --- |",
    ]
    for _, r in by_alert.iterrows():
        L.append(f"| {r['alert']} | {int(r['n'])} | {r['median_enrichment']:.2f}x "
                 f"| {r['top_hit_rate']:.1%} |")
    L += [
        "",
        "## Control: is a low enrichment the model's fault or the probe's?",
        "",
        f"Single-atom masking understates a substructure whose signal spans several "
        f"atoms. As a control, each alert was masked **as a whole** and compared with "
        f"{N_NULL_DRAWS} seeded random atom sets of the same size drawn from outside "
        f"it. The alert produces a larger probability drop than the average random "
        f"set in **{control['beats_null'].mean():.1%}** of {len(control)} molecules "
        f"(median percentile against the null: "
        f"{control['percentile_vs_null'].median():.2f}).",
        "",
        "| alert | molecules | median percentile vs null | beats null |",
        "| --- | --- | --- | --- |",
    ] + [
        f"| {r['alert']} | {int(r['n'])} | {r['median_percentile']:.2f} "
        f"| {r['beats_null_rate']:.0%} |"
        for _, r in control_by_alert.iterrows()
    ] + [
        "",
        "Where the two probes disagree, the per-atom number is the less reliable of "
        "the two for small groups: masking one atom of a three-atom epoxide leaves a "
        "near-identical environment, while masking all three removes a ring. "
        "Disagreements (per-atom enrichment > 1.1 but group percentile <= 0.6, or the "
        "reverse): "
        + (", ".join(
            f"{r['alert']} ({per_atom_by_alert[r['alert']]:.2f}x per atom vs "
            f"{r['median_percentile']:.2f} percentile per group, n={int(r['n'])})"
            for _, r in control_by_alert.iterrows()
            if r["alert"] in per_atom_by_alert
            and (per_atom_by_alert[r["alert"]] > 1.1) != (r["median_percentile"] > 0.6)
        ) or "none") + ".",
        "",
        "## Annotated examples",
        "",
        "Red = removing the atom lowered the prediction (it pushed toxicity up); "
        "blue = the opposite. Alert atoms are drawn with a larger highlight radius.",
        "",
    ]
    for fig in figures:
        if fig["kind"] == "success":
            L.append(f"![{fig['inchikey']}]({fig['path']}) ")
            L.append("")
            L.append(f"- `{fig['inchikey']}` — alerts: {fig['alerts']}, "
                     f"enrichment {fig['enrichment']:.2f}x")
            L.append("")
    L += [
        "## Where it fails",
        "",
        f"- **Diffuse attribution**: {len(diffuse)} of {len(frame)} molecules have a "
        f"normalised attribution entropy above {DIFFUSE_THRESHOLD}, i.e. the signal is "
        f"spread almost uniformly over every atom and carries no localised explanation.",
        f"- **Off-target attribution**: {len(off_target)} molecules put the top-{TOP_K} "
        f"atoms outside the alert AND concentrate less than half the uniform share of "
        f"attribution mass on it.",
        "",
    ]
    for fig in figures:
        if fig["kind"] == "failure":
            L.append(f"![{fig['inchikey']}]({fig['path']}) ")
            L.append("")
            L.append(f"- `{fig['inchikey']}` — alerts: {fig['alerts']}, "
                     f"enrichment {fig['enrichment']:.2f}x")
            L.append("")
    L += [
        "Per-molecule numbers for every alert-bearing test molecule are in "
        "`results/attribution_per_molecule.csv`.",
        "",
    ]
    (paths.RESULTS / "attribution.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"top-{TOP_K} hit rate {hit_rate:.1%}; median enrichment "
          f"{frame['enrichment'].median():.2f}x; {len(diffuse)} diffuse, "
          f"{len(off_target)} off-target")
    print(f"Wrote {paths.RESULTS / 'attribution.md'}, attribution.json, "
          f"attribution_per_molecule.csv and {len(figures)} figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
