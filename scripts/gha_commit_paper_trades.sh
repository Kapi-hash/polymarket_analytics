#!/usr/bin/env bash
# Commit and push paper/swing journal artifacts from GitHub Actions.
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

EXISTING=()
while IFS= read -r -d '' f; do
  EXISTING+=("$f")
done < <(
  find data -maxdepth 1 -type f \( \
    -name 'paper_trades*.json' \
    -o -name 'swing_trades*.json' \
    -o -name 'paper_trader.log' \
    -o -name 'swing_trader.log' \
  \) -print0 2>/dev/null | sort -z
)

if [[ ${#EXISTING[@]} -eq 0 ]]; then
  echo "No paper/swing trade artifacts found; nothing to commit."
  exit 0
fi

git add -- "${EXISTING[@]}"

if git diff --staged --quiet; then
  echo "No paper trade changes to commit (empty diff)."
  exit 0
fi

git commit -m "$(cat <<'EOF'
chore: update multi-profile paper/swing journals [skip ci]

Automated GitHub Actions forward-test snapshot.
EOF
)"

git push origin "HEAD:${BRANCH}"
echo "Pushed journal updates to ${BRANCH}: ${EXISTING[*]}"
