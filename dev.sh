#!/usr/bin/env bash
# Start the API (port 8000) and the web app (http://localhost:5173). Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"
(cd backend && uv run uvicorn scgame.api:app --port 8000) &
API=$!
trap 'kill $API 2>/dev/null' EXIT
cd frontend
[ -d node_modules ] || npm install
npx vite --port 5173 --open
