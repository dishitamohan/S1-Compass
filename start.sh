#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
CLIENT_PORT="${CLIENT_PORT:-8001}"
CLIENT_ORIGINS="${CLIENT_ORIGINS:-http://127.0.0.1:8001,http://localhost:8001}"
RELEASE_DIR="${RELEASE_DIR:-$ROOT/artifacts/smoke/run}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$ROOT/artifacts/modernbert-base}"
BASE_MODEL="${BASE_MODEL:-modernbert-base}"
CANDIDATES="${CANDIDATES:-}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
RUN_DIR="$ROOT/.run"
API_LOG="$LOG_DIR/api.log"
CLIENT_LOG="$LOG_DIR/client.log"
SUPERVISOR_LOG="$LOG_DIR/supervisor.log"

mkdir -p "$LOG_DIR" "$RUN_DIR"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python environment: $PYTHON" >&2
  echo "Create it and install the project dependencies before running ./start.sh." >&2
  exit 1
fi
if ! command -v lsof >/dev/null || ! command -v curl >/dev/null; then
  echo "start.sh requires lsof and curl to check ports and service health." >&2
  exit 1
fi

log() {
  local line="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  printf '%s\n' "$line" | tee -a "$SUPERVISOR_LOG"
}

listener_pids() {
  lsof -nP -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null || true
}

stop_project_listener() {
  local port="$1" kind="$2" mode="${3:-stop}" pid command cwd matches
  matches="$(listener_pids "$port")"
  [[ -z "$matches" ]] && return 0
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    cwd="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1)"
    if [[ "$cwd" != "$ROOT" ]]; then
      log "Refusing to stop PID $pid on port $port; its working directory is '$cwd'."
      return 1
    fi
    if [[ "$kind" == api && "$command" == *"-m s1_compass.serve"* ]]; then
      :
    elif [[ "$kind" == api && "$command" == *"-m http.server"* && "$command" == *" $API_PORT "* && "$command" == *"--directory $ROOT/client"* ]]; then
      log "Found the old client server occupying the model API port; replacing it with the API."
    elif [[ "$kind" == client && "$command" == *"-m http.server"* && "$command" == *" $CLIENT_PORT "* && "$command" == *"--directory $ROOT/client"* ]]; then
      :
    else
      log "Refusing to stop PID $pid on port $port; it is not a recognized S1 Compass $kind server."
      log "Process: $command"
      return 1
    fi
    [[ "$mode" == "check" ]] && continue
    log "Stopping existing S1 Compass $kind server (PID $pid, port $port)."
    kill -TERM "$pid"
    for _ in {1..50}; do
      matches="$(listener_pids "$port")"
      [[ " $matches " != *" $pid "* ]] && break
      sleep 0.1
    done
    matches="$(listener_pids "$port")"
    if [[ " $matches " == *" $pid "* ]]; then
      log "PID $pid did not stop within five seconds; sending SIGKILL."
      kill -KILL "$pid"
    fi
  done <<< "$matches"
}

# Validate both listeners before terminating either one. This prevents a failed
# restart from leaving the API down because the client was not recognized.
stop_project_listener "$API_PORT" api check
stop_project_listener "$CLIENT_PORT" client check
stop_project_listener "$API_PORT" api
stop_project_listener "$CLIENT_PORT" client

API_PID=""
CLIENT_PID=""
TAIL_PID=""
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  [[ -n "$TAIL_PID" ]] && kill "$TAIL_PID" 2>/dev/null || true
  for pid in "${API_PID:-}" "${CLIENT_PID:-}"; do
    [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
  done
  [[ -n "$API_PID" ]] && wait "$API_PID" 2>/dev/null || true
  [[ -n "$CLIENT_PID" ]] && wait "$CLIENT_PID" 2>/dev/null || true
  rm -f "$RUN_DIR/api.pid" "$RUN_DIR/client.pid"
  log "Both S1 Compass servers stopped (exit $exit_code)."
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ ! -d "$RELEASE_DIR" || ! -d "$SNAPSHOT_DIR" ]]; then
  log "Release or backbone directory is missing. Check RELEASE_DIR and SNAPSHOT_DIR."
  exit 1
fi

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/private/tmp/s1-compass-pycache}"
log "Starting S1 Compass API on $API_HOST:$API_PORT."
IFS=',' read -r -a origin_list <<< "$CLIENT_ORIGINS"
origin_args=()
for origin in "${origin_list[@]}"; do
  [[ -n "$origin" ]] && origin_args+=(--client-origin "$origin")
done
candidate_args=()
if [[ -n "$CANDIDATES" ]]; then
  IFS=',' read -r -a candidate_list <<< "$CANDIDATES"
  for candidate in "${candidate_list[@]}"; do
    candidate_args+=(--candidate "$candidate")
  done
fi
if [[ -n "$CANDIDATES" ]]; then
  "$PYTHON" -m s1_compass.serve \
    --release "$RELEASE_DIR" \
    --snapshot "$SNAPSHOT_DIR" \
    --base-model "$BASE_MODEL" \
    --benchmark-mode \
    --host "$API_HOST" \
    --port "$API_PORT" \
    "${candidate_args[@]}" \
    "${origin_args[@]}" \
    >>"$API_LOG" 2>&1 &
else
  "$PYTHON" -m s1_compass.serve \
    --release "$RELEASE_DIR" \
    --snapshot "$SNAPSHOT_DIR" \
    --base-model "$BASE_MODEL" \
    --benchmark-mode \
    --host "$API_HOST" \
    --port "$API_PORT" \
    "${origin_args[@]}" \
    >>"$API_LOG" 2>&1 &
fi
API_PID=$!
printf '%s\n' "$API_PID" > "$RUN_DIR/api.pid"

log "Starting S1 Compass client on $CLIENT_HOST:$CLIENT_PORT."
"$PYTHON" -m http.server "$CLIENT_PORT" --bind "$CLIENT_HOST" --directory "$ROOT/client" \
  >>"$CLIENT_LOG" 2>&1 &
CLIENT_PID=$!
printf '%s\n' "$CLIENT_PID" > "$RUN_DIR/client.pid"

log "Waiting for the model API and client to respond."
ready=0
for _ in {1..90}; do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    log "API process exited during startup. See $API_LOG."
    tail -n 60 "$API_LOG" >&2 || true
    exit 1
  fi
  if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
    log "Client process exited during startup. See $CLIENT_LOG."
    tail -n 60 "$CLIENT_LOG" >&2 || true
    exit 1
  fi
  if curl -fsS --max-time 1 "http://127.0.0.1:$CLIENT_PORT/" >/dev/null 2>&1 \
    && curl -fsS --max-time 1 -X OPTIONS "http://127.0.0.1:$API_PORT/v1/decide" \
      -H "Origin: http://127.0.0.1:$CLIENT_PORT" \
      -H 'Access-Control-Request-Method: POST' \
      -H 'Access-Control-Request-Headers: content-type' >/dev/null 2>&1 \
    && curl -fsS --max-time 1 "http://127.0.0.1:$API_PORT/v1/models" \
      -H "Origin: http://127.0.0.1:$CLIENT_PORT" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  log "Startup health check timed out. See $API_LOG and $CLIENT_LOG."
  tail -n 60 "$API_LOG" "$CLIENT_LOG" >&2 || true
  exit 1
fi

log "Both servers are ready. Open http://127.0.0.1:$CLIENT_PORT/ in any browser on this Mac."
log "Complete logs: $API_LOG, $CLIENT_LOG, $SUPERVISOR_LOG. Press Ctrl-C to stop both servers."

tail -n 0 -F "$API_LOG" "$CLIENT_LOG" &
TAIL_PID=$!
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$CLIENT_PID" 2>/dev/null; do
  sleep 1
done
log "A server process exited; shutting down its partner. Review logs before restarting."
exit 1
