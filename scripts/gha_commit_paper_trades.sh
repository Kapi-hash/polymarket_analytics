#!/usr/bin/env bash
# Commit and push paper-trade journal artifacts from GitHub Actions.
# Exits 0 when there is nothing new to commit (empty diff is not a failure).
#
# Usage: ./scripts/gha_commit_paper_trades.sh [branch]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BRANCH="${1:-${GITHUB_REF_NAME:-main}}"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

mkdir -p data
PATHS=(
  data/paper_trades.json
  data/paper_trader.log
)

EXISTING=()
for p in "${PATHS[@]}"; do
  if [[ -f "$p" ]]; then
    EXISTING+=("$p")
  fi
done

if [[ ${#EXISTING[@]} -eq 0 ]]; then
  echo "No paper-trade artifacts found; nothing to commit."
  exit 0
fi

git add -- "${EXISTING[@]}"

if git diff --staged --quiet; then
  echo "No paper trade changes to commit (empty diff)."
  exit 0
fi

git commit -m "$(cat <<'EOF'
chore: update paper trade journal [skip ci]

Automated GitHub Actions paper-trader snapshot.
EOF
)"

git push origin "HEAD:${BRANCH}"
echo "Pushed paper trade journal update to ${BRANCH}."
