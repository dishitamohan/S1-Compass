# S1 Compass: selectable base models and Qwen3-0.6B training

Implementation plan for GPT Luna. This document specifies work to do; it does not claim that Qwen support or a trained Qwen candidate already exists.

## Goal and acceptance result

Keep the current ModernBERT route working. Add `Qwen/Qwen3-0.6B` as a second supported base model for the same `DecisionRow` training corpus and `/v1/decide` response contract. A user can select an installed, trained candidate in the browser; the selected model must be identified in the response. The training command must also take an explicit base-model choice. A Qwen snapshot alone is **not** a trained S1 Compass candidate and must never appear as one in the picker.

Done means: both backbones can train, calibrate, evaluate, save, reload, serve, and render decisions through the existing client; each candidate is bound to its own tokenizer, preprocessing version, base snapshot, weights, and calibration; switching models cannot silently serve a different candidate or mix artifacts. Existing ModernBERT smoke artifacts remain readable. Keep synthetic-data releases in benchmark mode and subject real releases to the existing promotion gates.

## Facts and constraints to use

* The repository currently pins Python `>=3.9,<3.10`, MLX 0.21.1, tokenizers 0.21.4, and Transformers 4.48.0 in `pyproject.toml`. The ModernBERT tokenizer, 768-wide head, backbone loader, public model ID, pipeline, and serving startup are hard coded. The browser client and `start.sh` assume one ModernBERT candidate.
* [Qwen's model card](https://huggingface.co/Qwen/Qwen3-0.6B) identifies Qwen3-0.6B as a causal LM with 28 layers, 16 query heads, 8 key/value heads, and 32,768 advertised context tokens. It warns that Transformers earlier than 4.51.0 do not recognize `qwen3`. The [published config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json) reports hidden size 1024, grouped query attention, and tied token embeddings. Pin one immutable revision and record its actual config rather than assuming `main` remains unchanged.
* [Transformers documents `Qwen3Model`](https://huggingface.co/docs/transformers/model_doc/qwen3) as the base model outputting hidden states. [MLX-LM has a Qwen3 implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py) and a [fine-tuning entry point](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py), but its generic causal-LM training objective does not implement this project's option-level classification and calibration. Reuse its supported model conversion or layer code only after proving that hidden states and gradients meet this plan's contract.
* Qwen is causal. The current ModernBERT sequence packs *all* options and mean-pools their spans after bidirectional attention. Reusing that sequence would make earlier options unable to see later options and later options able to see earlier ones. For Qwen, score each option in an independent sequence sharing an identical instruction/state prefix. Never generate free text as the decision output. The same scalar ranking and temperature calibration contract remains.

## User-visible behavior

1. Training CLI accepts `--base-model modernbert-base|qwen3-0.6b`, defaulting to `modernbert-base` for compatibility, plus `--snapshot`, `--rows`, `--annotations`, and `--output`. Resolve a missing snapshot from the chosen model's manifest. Reject a selector/snapshot mismatch before partitioning or loading weights. Record the selected ID in all run artifacts and logs.
2. The browser shows a **Base model** picker populated from the running API's installed candidate registry. Initial selection is the active server model. Each option shows a friendly name, checkpoint status, and whether it is a benchmark candidate. A model with no trained candidate is listed as unavailable or omitted with an explanation; choosing it never falls back to ModernBERT.
3. Choosing another available model activates its saved candidate. Show `Loading…` until the API confirms the active model and `Ready` afterward. Disable **Run decision** during activation. On failure, restore the previous selection and show the server error. Keep the JSON request format unchanged: model choice is a client/server setting, not a new request field.
4. The response card displays the actual `model` returned by the API, mapped to a friendly base-model label. Remove the hard-coded `ModernBERT` text from result and model cards. The model picker and result must agree even after a reload or another tab changes the active model.

## Artifact and API contracts

Create a small model registry with stable IDs, display names, architecture (`encoder` or `causal`), local snapshot path, local release path, availability, and active status. Persist configuration in a project-local file or explicit launcher arguments; do not discover arbitrary files or accept arbitrary client paths. Keep snapshots under separate directories, for example `artifacts/modernbert-base/` and `artifacts/qwen3-0.6b/`, and releases under distinct run directories. Preserve `artifacts/smoke/run/` as the legacy ModernBERT default.

Each new candidate writes a manifest such as `release.json` with at least `format_version`, `base_model_id`, `base_revision`, `architecture`, `hidden_size`, `snapshot_lock_sha256`, `tokenizer_sha256`, `preprocessing_version`, `checkpoint_sha256`, `model_sha`, `calibration_version`, and evaluation report hash. Loaders validate these fields against disk before loading tensors; reject missing, mismatched, or cross-model artifacts. ModernBERT legacy loading is a clearly marked compatibility branch. Include base ID and preprocessing version in the model SHA input. New public model IDs use `s1-compass-{base-model-id}:{sha256}`; relax schema validation to an allowlisted base ID plus 64 lowercase hex digits and preserve the legacy ModernBERT prefix. Do not loosen this to an arbitrary string.

Add `GET /v1/models` returning available installed candidates and `active_model_id`, and `POST /v1/models/active` with `{"model_id":"qwen3-0.6b"}`. Activation is local-only and returns the active model's metadata after successful load. Serialize activation and inference: no request is scored with a half-loaded model. Keep one resident model at a time to fit the current memory target; load the replacement fully, validate it, atomically swap the active service, then release the old instance. If memory cannot hold both briefly, stop accepting decisions, unload the old service, load the new one, and restore the old model on failure. Return a distinct busy/loading status to the client and document it. Do not persist selection in browser storage as a substitute for server state. Preserve existing CORS behavior for the separate local client and add support for these endpoints.

## Implementation sequence

### 1. Freeze compatibility and choose a supported runtime

* Capture the existing smoke request/response, model ID, calibration, checkpoint hashes, and launch command as a baseline. Record current versions from the environment.
* Resolve the Python/MLX/Transformers compatibility conflict in `pyproject.toml` and `requirements.lock.txt`. Qwen3 requires Transformers 4.51.0 or newer for the official reference loader. Choose and pin one tested Apple Silicon Python + MLX + MLX-LM/Transformers combination; do not simply upgrade one package while retaining an incompatible `<3.10` Python pin. Keep Transformers/PyTorch in a reference or conversion extra if MLX runs inference and training. Document a clean virtual environment setup and recheck ModernBERT loading on that runtime.
* Specify whether Qwen uses an adapted MLX-LM Qwen3 base model or a strict local port. Do a short forward/gradient spike before committing to either. The chosen path must expose per-token hidden states and support training the option head and desired backbone parameters. A text-generation-only API is insufficient.

### 2. Download and lock the Qwen snapshot

* Add a deterministic snapshot command taking model ID and immutable 40-character revision. Download config, tokenizer files, safetensors/index, and all files required by the chosen loader. Write a lock with revision, individual file hashes, aggregate weight hash, library versions, and architecture. Verify every file at startup; work offline after download. Reject model ID, architecture, token IDs, or weight names that do not match the pinned manifest.
* Use Qwen's tokenizer and its actual special tokens. Define a local padding policy and attention mask without changing the vocabulary or resizing embeddings. Do not borrow ModernBERT CLS/SEP IDs. If using a chat template, freeze the exact template and `enable_thinking=False` behavior as part of preprocessing; otherwise use a plain, versioned classification prompt. Pick one and keep it identical in training, validation, calibration, and serving. No generation or `<think>` content is part of scoring.
* Save a tokenizer golden fixture with token IDs and masks for an ordinary row, Unicode text, long state, two and eight options, and reordered options. Compare with the pinned reference tokenizer.

### 3. Introduce backbone and tokenizer interfaces

* Define a registry/factory around a model spec; route snapshot verification, tokenizer loading, encoding, backbone loading, hidden width, checkpoint serialization, and public ID construction through it. Use explicit types rather than scattered `if model_id == ...` checks. Keep the current ModernBERT implementation as the encoder plugin.
* Generalize `s1_compass.model.heads.Head` from fixed width 768 to `spec.hidden_size`, and instantiate it from saved candidate metadata on training and serving. Preserve a single bias-free scalar head shared across options. Do not change the request/answer schemas, confidence policy, or option-ID mapping.
* Refactor `s1_compass.data.dataset.partition` and `batches` to accept the chosen tokenizer/encoder, record preprocessing version and lock hashes in the split manifest, and reject use of a split with the wrong tokenizer. Partition membership can remain deterministic from row IDs and annotations; model-specific encoded batches must be rebuilt or cached separately.

### 4. Implement causal option scoring for Qwen

* For each `(state_text, question, option)`, construct an independent causal sequence `instruction + state + option description` using the versioned delimiters/template. The option description must be the final scored text segment. The **same** instruction/state tokens and truncation must be used for every option of a question; allocate state capacity using the longest option so options are compared on equal evidence. Exclude option IDs and other options' descriptions from the model input. Limit the final sequence to the chosen training length, initially 512 unless profiling justifies another limit; retain head and tail of truncated state, with an explicit minimum evidence budget. Reject unencodable fixed content and empty option spans.
* Flatten valid options to sequences, pad and mask them, run the causal backbone, FP32 mean-pool each option's description hidden states, apply the shared scalar head, then reshape to `[question, option]` logits. Use the existing masked cross entropy, stable softmax, lexical tie rule, option-order permutation/remapping, and temperature fitting. Gradients must flow through the Qwen backbone if full fine-tuning is selected, or through declared adapters plus head for the resource-limited path. Record the exact trainable parameter set in metadata.
* Start with a memory-safe LoRA/adapters-plus-head profile for Qwen on the available Mac. Make full fine-tuning an explicit mode only if a measured peak memory budget permits it; never silently swap one training mode for another. Train with bounded token batches and gradient accumulation; `20 questions × 8 options × 512 tokens` must be budgeted separately from the current one-sequence-per-question ModernBERT route. Keep checkpoint selection by validation Brier at `T=1`, deterministic seed, finite-loss checks, gradient clipping, and exact resume metadata.
* Validate hidden states against a pinned Transformers reference at short/padded/long inputs, using the same attention mask and positions. Require a documented numerical tolerance appropriate to the chosen dtype and test decision-level agreement. The current ModernBERT FP32 parity record cannot certify Qwen.

### 5. Wire training, calibration, evaluation, and release loading

* Update `s1_compass.pipeline.run`, `s1_compass.train.loop.fit`, `s1_compass.model.scorer`, and `s1_compass.serve.api.load_release` to use model specs. Keep a per-candidate output directory. Save weights/adapters, head weights, optimizer/resume state, training config, snapshot lock, calibration, and evaluation with hashes. Never reuse ModernBERT's `candidate.npz`, calibration, or model SHA for Qwen.
* Calibration must use held-out calibration rows encoded with the selected model's tokenizer and template. Refit temperature for each Qwen checkpoint, then evaluate on untouched test rows. Existing numeric quality gates remain until actual held-out Qwen results justify a documented change. Do not mark a synthetic smoke run promotable.
* Update `s1_compass.serve.__main__` and `start.sh` so a configured registry is checked at startup. Preserve the one-command ModernBERT default. Give an explicit command to train Qwen and an explicit command to launch a server with both installed candidates; log active model ID, revision, candidate hash, and activation events. Do not let the launcher decide a model from whatever files happen to be present.

### 6. Implement model discovery and picker in the client

* Add the base-model picker near the API connection indicator in `client/index.html`, matching the existing visual style in `client/styles.css`. Fetch `/v1/models` after connection and after activation. Keep a plain explanation when only ModernBERT is installed. Show the backend's active model as source of truth; use the `model` field on each decision result for the displayed answer identity.
* Add `activateModel` in `client/app.js`: disable run controls, POST the chosen ID, handle loading/error states, refetch active metadata, and clear or label the previous result to avoid attributing it to the new model. Handle a stale tab by refreshing model metadata before a run or when the server reports an active-model mismatch. Do not clear the user's context, questions, or pasted JSON when switching.
* Keep the existing JSON input mode and editable option UI intact. Document that the model choice is outside the request body.

### 7. Verify and report results

* Check ModernBERT regression: tokenizer golden, existing smoke candidate load, same typed output contract, startup, and model picker default.
* Check Qwen: snapshot hash rejection, tokenizer golden, hidden-state/reference parity, forward/backward on a tiny corpus, reproducible option permutation, option independence, checkpoint restore, calibration binding, API output schema, activation rollback, and model identity after switching. Include a probe where one option's text changes and every other option logit remains unchanged before softmax; probabilities may change due to normalization.
* Run an end-to-end Qwen smoke pipeline with the existing synthetic fixtures, then run the full pipeline only when a production-size labeled corpus is available. Save candidate location, training steps, validation Brier, calibration temperature, test metrics, peak memory, and per-model request latency. Check the current 500 ms request deadline and benchmark gates with Qwen separately; if Qwen cannot meet them, report that clearly and keep it a benchmark candidate rather than relaxing policy implicitly.
* Update `client/README.md` and the top-level operational documentation with setup, both model IDs, snapshot/download commands, training and serving commands, picker behavior, artifact locations, and recovery from a failed activation.

## Suggested work order and handoff checkpoints for Luna

Implement in small, reviewable phases: **A** runtime/snapshot/registry, **B** Qwen tokenizer and scoring, **C** training and artifacts, **D** serving and picker, **E** end-to-end evaluation and docs. At the end of each phase, provide changed files, the exact command used, result, and any blocker. Do not claim production readiness from a synthetic smoke run. If the chosen Qwen backend cannot expose trainable hidden states within the memory budget, stop that phase with measured evidence and select a documented alternative before changing the product contract.

## Source references

* [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
* [Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
* [Transformers Qwen3 model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3)
* [MLX-LM Qwen3 implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py)
* [MLX-LM fine-tuning implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py)
