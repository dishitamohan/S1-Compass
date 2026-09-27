# S1 Compass Decision Studio

Local model training and decision scoring. See [the Qwen3 base-model plan](QWEN3_BASE_MODEL_PLAN.md) for the selectable model implementation roadmap.

## Registered base models

```bash
.venv/bin/python -m s1_compass.models list
```

Download an immutable Hugging Face revision into the registry's local artifact path:

```bash
.venv/bin/python -m s1_compass.models download qwen3-0.6b --revision <40-character-commit>
```

To hash-lock a model already present on disk without downloading it:

```bash
.venv/bin/python -m s1_compass.models lock modernbert-base --revision <40-character-commit>
```

Verify a previously locked snapshot:

```bash
.venv/bin/python -m s1_compass.models verify qwen3-0.6b
```

Snapshot locks bind the repository ID, immutable commit, architecture, hidden size, and each local file hash. Downloaded weights and generated run artifacts stay local and are ignored by Git.

## Start the local client and model

```bash
./start.sh
```

Open <http://127.0.0.1:8001/>. See [client instructions](client/README.md) for serving details.

## Python runtime

The supported project runtime is Python 3.11 or 3.12 on Apple Silicon. Create a clean environment and install the pinned project stack:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[reference]'
```

The dependency set pins `huggingface-hub==0.30.2`, compatible with Transformers 4.51.3 and tokenizers 0.21.4.

Qwen3 snapshot download requires a network connection and a 40-character immutable Hugging Face commit:

```bash
.venv/bin/python -m s1_compass.models download qwen3-0.6b --revision <40-character-commit>
```

The Qwen3 path uses MLX-LM's Qwen3 backbone with the project-specific per-option hidden-state scorer. Qwen training and activation remain gated until the reference hidden-state, decision agreement, option-independence, and adapter-gradient parity command passes. A base snapshot is not a trained candidate. To train ModernBERT, use `.venv/bin/python -m s1_compass.pipeline --base-model modernbert-base --rows artifacts/smoke/rows.jsonl --annotations artifacts/smoke/annotations.json --small-data --output artifacts/modernbert-base/runs/smoke`.

### Qwen3 parity, training, and serving

After installing the supported Python runtime and hash-locked Qwen snapshot, run the CPU Transformers reference and MLX parity gate. This compares short, padded, long, Unicode, reordered, and eight-option cases; it also checks option independence and an adapter-plus-head backward pass. Training and serving refuse Qwen without a passing sidecar:

```bash
.venv/bin/python -m s1_compass.models parity
```

Run the synthetic smoke pipeline as a benchmark candidate:

```bash
.venv/bin/python -m s1_compass.pipeline \
  --base-model qwen3-0.6b \
  --snapshot artifacts/qwen3-0.6b \
  --rows artifacts/smoke/rows.jsonl \
  --annotations artifacts/smoke/annotations.json \
  --output artifacts/runs/qwen3-0.6b/smoke \
  --small-data --epochs 1 --max-steps 1
```

The Qwen profile is LoRA rank 8, scale 16 on query and value projections in the final four transformer blocks, plus the shared scalar head. It stores only trainable adapter/head weights in `candidate.npz`. Synthetic output remains benchmark-only and cannot be promoted.

Run only that Qwen candidate:

```bash
.venv/bin/python -m s1_compass.serve \
  --base-model qwen3-0.6b \
  --release artifacts/runs/qwen3-0.6b/smoke \
  --snapshot artifacts/qwen3-0.6b \
  --benchmark-mode
```

Or run the local client with both explicitly registered candidates (after both releases exist):

```bash
BASE_MODEL=modernbert-base \
CANDIDATES='modernbert-base=artifacts/smoke/run,qwen3-0.6b=artifacts/runs/qwen3-0.6b/smoke' \
./start.sh
```

Artifacts live under `artifacts/qwen3-0.6b/`. The API loads one resident candidate at a time. Switching candidates is serialized; failed activation attempts restore the previous model and return an error. `GET /v1/models` reports installed trained releases; a snapshot by itself is unavailable in the picker.
