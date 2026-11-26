# Trained models

Committed so that a fresh clone can predict without training anything and
without downloading any model weights. Build or rebuild this directory with:

```bash
python src/38_export_models.py
```

| name | task | input | output |
| --- | --- | --- | --- |
| `toxicity_gin` | 12-endpoint toxicity | SMILES | 12 probabilities |
| `multitask_gin` | shared encoder, both heads | SMILES (+ protein) | 12 probabilities, pKd |
| `affinity_frozen_mlp` | drug–target affinity | SMILES + UniProt accession | pKd-scale affinity |

`protein_embeddings.npz` holds mean-pooled ESM-2 650M vectors for the 229 KIBA
proteins. It is 1.1 MB but needs a 2.5 GB model download to regenerate, which is
why it ships here rather than being fetched.

`registry.json` records each file's sha256, architecture and measured metrics.
`load_model` verifies the hash, so a corrupted download fails loudly.

## Using them

```python
from tox21lib.model_io import load_model

tox = load_model("toxicity_gin")
print(tox.predict_frame(["CCO", "O=[N+]([O-])c1ccccc1"]))   # 12 columns, named

aff = load_model("affinity_frozen_mlp")
print(aff.predict(["CC(=O)Oc1ccccc1C(=O)O"], ["P00533"]))    # EGFR
print(len(aff.known_proteins))                                # 229
```

Unparsable SMILES return a row of `NaN` rather than being dropped, so output
rows always align with input rows. Asking for a protein with no committed
embedding raises instead of substituting a zero vector — a zero vector is a
valid-looking input that yields a plausible-looking wrong answer.

## Replacing a model

Nothing in the calling code names a file. Either pass a path:

```python
model = load_model("path/to/your_model.pt")
```

or add an entry to `registry.json` and keep calling it by name. A replacement
must carry the same self-describing envelope:

```python
torch.save({
    "format_version": 1,
    "task": "toxicity",              # or "affinity" / "multitask"
    "architecture": {...},           # whatever your loader needs
    "state_dict": model.state_dict(),
    "endpoints": [...],              # toxicity only: the output column order
    "metadata": {"metrics": {...}, "input_contract": "..."},
}, "your_model.pt")
```

Loading checks `format_version`, the presence of `task` / `architecture` /
`state_dict`, that the endpoint names match the number of outputs, and that the
protein embedding width matches what the model expects. A replacement that gets
the endpoint order wrong is rejected at load time rather than silently writing
probabilities into the wrong columns.

To register a custom architecture, add a `_build_*` function in
`src/tox21lib/model_io.py` and dispatch on your `task` string.

## What is *not* here

The six ablation checkpoints (`config{A–F}_seed0.pt`, 74 MB) stay under
`data/processed/checkpoints/` and are gitignored. Four of them exist only to
support ablation tables and are regenerable with
`src/20_shared_encoder_ablation.py`.

The random-forest baselines are not shipped either: they refit in about ten
seconds (`src/04_baseline_rf.py`) and a pickled forest is both larger than its
training script and tied to one scikit-learn version.

## Honest scope

These models were measured under distribution shift and the numbers are in
`FINAL_REPORT.md`. Three limits matter before using them for anything:

- **Scores are not calibrated probabilities.** They over-predict ~1.7× on
  scaffold splits (ECE 0.0645). Use them to rank, not to quote a risk.
- **They fail outside their training chemistry.** The toxicity model confidently
  predicts 0.936 for cyclohexanol, a 7-heavy-atom molecule below the pipeline's
  own 10-atom training filter.
- **`affinity_frozen_mlp` is strong on seen proteins and weak on unseen ones**
  (Pearson 0.851 random vs 0.500 cold-target). On an unseen target it is beaten
  by a model that ignores the protein entirely.
