# S1 Compass Decision Studio

A dependency-free browser client for trying the local S1 Compass model. The API runs as a separate process; this directory contains only static HTML, CSS, and JavaScript.

## One-command start and restart

From the project root, run:

```bash
./start.sh
```

This restarts the project's local API and client servers, streams their logs in the terminal, and saves complete logs under `logs/`. Open <http://127.0.0.1:8001>. Press Ctrl-C to stop both servers.

The script refuses to stop a process on either port unless it can verify it belongs to this project.

## Start the model API

To start only the model API manually, from the project root:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/s1-compass-pycache .venv/bin/python -m s1_compass.serve \
  --release artifacts/smoke/run \
  --snapshot artifacts/modernbert-base \
  --benchmark-mode
```

`--benchmark-mode` is required for the current smoke candidate because its evaluation report correctly blocks production promotion. The API listens on `http://127.0.0.1:8000`.

## Start the client

To start only the client manually, from the project root:

```bash
python3 -m http.server 8001 --directory client
```

Open <http://127.0.0.1:8001>. The client sends requests to the local API and stores the API address in browser local storage. Requests are not sent to a hosted service.

The tester supports up to 20 choice, boolean, or ordinal questions, each with 2–8 options. It shows typed answers, confidence, probabilities, latency, and low-confidence escalation responses. The current candidate is a one-update synthetic-data smoke model and is for local testing only.

## Base-model picker

The picker reads explicitly installed trained candidates from `GET /v1/models`. Selection is activated through `POST /v1/models/active`; it is server state and is not added to the DecisionRow JSON request. A downloaded base snapshot by itself is not a trained candidate and cannot be selected. The result card identifies the public model ID returned by the API. Qwen remains unavailable until the MLX/Transformers parity gate passes and a trained release is installed.

When both trained releases are installed, start the client with an explicit candidate registry:

```bash
BASE_MODEL=modernbert-base \
CANDIDATES='modernbert-base=artifacts/smoke/run,qwen3-0.6b=artifacts/runs/qwen3-0.6b/smoke' \
./start.sh
```

To train the Qwen benchmark candidate, first run `.venv/bin/python -m s1_compass.models parity`, then follow the Qwen3 smoke command in the top-level README. Candidate selection is server state, so another tab’s activation is reflected when the client refreshes its model list before a decision.
