#!/usr/bin/env bash
set -euo pipefail

NAME="${1:-}"
if [[ -z "${NAME}" ]]; then
  echo "Usage: $0 <deployment-name>"
  exit 1
fi

BASE_URL="${BASE_URL:-http://localhost:7861}"
TIMEOUT="${TIMEOUT:-180}"

info() { echo "[smoke] $*"; }

info "Waiting for ${BASE_URL} to be ready (timeout ${TIMEOUT}s)..."
start_ts=$(date +%s)
while true; do
  if curl -fsS "${BASE_URL}/healthz" >/dev/null 2>&1; then
    info "Health OK"
    break
  fi
  now=$(date +%s)
  if (( now - start_ts > TIMEOUT )); then
    echo "Timed out waiting for ${BASE_URL}/healthz" >&2
    exit 1
  fi
  sleep 2
done

set -x
curl -fsS "${BASE_URL}/healthz" | head -c 500
# Adjust the endpoint and body to your actual API
if curl -fsS -X POST "${BASE_URL}/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"hello"}' | head -c 500; then
  :
else
  echo "Chat endpoint check failed" >&2
  exit 1
fi
set +x

info "Smoke tests passed for ${NAME}"

