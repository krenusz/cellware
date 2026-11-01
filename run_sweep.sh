#!/usr/bin/env bash
# Run a sweep file on GitHub Actions, wait, and copy only NEW result files into the repo.
#   ./run_sweep.sh sweeps/e9_v8.txt [--larry] [--lines 3-10]
# Existing files are never overwritten; logs and the full artifact stay in .sweep_runs/<run id>/.
set -euo pipefail
cd "$(dirname "$0")"
sweep=$1; shift
args=(-f "sweep=$sweep")
while [ $# -gt 0 ]; do
  case $1 in
    --larry) args+=(-f larry=true) ;;
    --lines) args+=(-f "lines=$2"); shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
branch=$(git rev-parse --abbrev-ref HEAD)
git fetch -q origin "$branch"
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$branch")" ]; then
  echo "push $branch first: the runners check out origin/$branch" >&2; exit 1
fi
before=$(gh run list --workflow sweep.yml --limit 1 --json databaseId -q '.[0].databaseId // 0')
gh workflow run sweep.yml --ref "$branch" "${args[@]}"
id=$before
while [ "$id" = "$before" ]; do sleep 3; id=$(gh run list --workflow sweep.yml --limit 1 --json databaseId -q '.[0].databaseId'); done
echo "run $id: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/actions/runs/$id"
gh run watch "$id" --interval 30 > /dev/null || true
out=.sweep_runs/$id
rm -rf "$out"; gh run download "$id" -n results -D "$out" || { echo "no results artifact" >&2; exit 1; }
new=0
while IFS= read -r f; do
  rel=${f#"$out"/}
  if [ ! -e "$rel" ]; then mkdir -p "$(dirname "$rel")"; cp "$f" "$rel"; echo "new: $rel"; new=$((new+1)); fi
done < <(find "$out/experiments" -type f -name '*.json' -o -type f -name '*.pt' | grep -v '^$')
echo "$new new result files; conclusion: $(gh run view "$id" --json conclusion -q .conclusion)"
