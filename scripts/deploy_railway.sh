#!/usr/bin/env bash
# Deploy CheeseBlend to Railway: project + service, the Muse key as a service secret, upload, public URL.
# Needs a signed-in Railway CLI (`railway login`) and MODEL_API_KEY in .env. Run: bash scripts/deploy_railway.sh
set -euo pipefail
WORKSPACE="${1:-ea88cc78-1953-4a9a-9175-912a959dffc2}"
cd "$(dirname "$0")/.."
# Optional: a Railway token in .env (RAILWAY_API_TOKEN=...) replaces the interactive `railway login`.
tok=$(grep '^RAILWAY_API_TOKEN=' .env | cut -d= -f2- | tr -d '\r\n' || true)
[ -n "$tok" ] && export RAILWAY_API_TOKEN="$tok"

railway status >/dev/null 2>&1 || railway init --name cheeseblend --workspace "$WORKSPACE"
railway add --service cheeseblend >/dev/null 2>&1 || true  # already exists on a re-run
railway service cheeseblend
# Railpack ignores railway.json's startCommand and would run `python app.py`; pin it.
railway variable set 'RAILPACK_START_CMD=streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true' \
  --service cheeseblend --skip-deploys
# The key goes in through stdin into Railway's secret store: never on the command line, never in the repo.
grep '^MODEL_API_KEY=' .env | cut -d= -f2- | tr -d '\r\n' |
  railway variable set MODEL_API_KEY --stdin --service cheeseblend --skip-deploys
railway up --service cheeseblend --detach  # respects .gitignore, so .env and data/raw stay local
railway domain --service cheeseblend
