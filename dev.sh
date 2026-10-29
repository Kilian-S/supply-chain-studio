#!/usr/bin/env bash
# Start the API (port 8000) and the web app (http://localhost:5173) in this terminal tab.
# Ctrl+C or closing the tab stops both, helpers included (see scripts/lifecycle.sh).
# NO_OPEN=1 skips opening the browser.
cd "$(dirname "$0")" || exit 1
source scripts/lifecycle.sh
lifecycle_init "$0" "$@"

prefix() { while IFS= read -r line; do echo "[$1] $line"; done; }

(cd backend && exec uv run uvicorn scgame.api:app --port 8000 --reload) 2>&1 | prefix api &
API_JOB=$!
(
  cd frontend || exit 1
  [ -d node_modules ] || npm install
  if [ -n "${NO_OPEN:-}" ]; then exec npx vite --port 5173 --strictPort; else exec npx vite --port 5173 --strictPort --open; fi
) 2>&1 | prefix web &
WEB_JOB=$!

lifecycle_wait "$API_JOB" "$WEB_JOB"
