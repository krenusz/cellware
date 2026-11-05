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
    --guard) guard=$2; shift ;;  # cancel once fewer than N seeds can pass the gate
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
branch=$(git rev-parse --abbrev-ref HEAD)
git fetch -q origin "$branch"
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$branch")" ]; then
  echo "push $branch first: the runners check out origin/$branch" >&2; exit 1
fi
# lock dispatch + run-id lookup so concurrent launches cannot pick up each other's run
lock=.sweep_runs/.dispatch.lock
mkdir -p .sweep_runs
until mkdir "$lock" 2>/dev/null; do sleep 2; done
trap 'rmdir "$lock" 2>/dev/null || true' EXIT
before=$(gh run list --workflow sweep.yml --limit 1 --json databaseId -q '.[0].databaseId // 0')
gh workflow run sweep.yml --ref "$branch" "${args[@]}"
id=$before
while [ "$id" = "$before" ]; do sleep 3; id=$(gh run list --workflow sweep.yml --limit 1 --json databaseId -q '.[0].databaseId'); done
rmdir "$lock"
echo "run $id: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/actions/runs/$id"
[ -n "${guard:-}" ] && (cd experiments && uv run python -u sweep_guard.py "$id" "../$sweep" "$guard" > "../.sweep_runs/guard_$id.log" 2>&1 &)
gh run watch "$id" --interval 30 > /dev/null || true
out=.sweep_runs/$id
rm -rf "$out"; gh run download "$id" -n results -D "$out" || { echo "no results artifact" >&2; exit 1; }
new=0; diff=0
while IFS= read -r f; do
  rel=${f#"$out"/}
  if [ ! -s "$rel" ]; then  # missing or empty (e.g. left by a smoke test): take the run's file
    mkdir -p "$(dirname "$rel")"; cp "$f" "$rel"; echo "new: $rel"; new=$((new+1))
  elif ! diff -q --strip-trailing-cr "$f" "$rel" > /dev/null; then  # never overwrite silently: keep both and say so
    cp "$f" "$rel.run$id"; echo "DIFFERS (local kept, run copy at $rel.run$id): $rel"; diff=$((diff+1))
  fi
done < <(find "$out/experiments" -type f -name '*.json' -o -type f -name '*.pt' | grep -v '^$')
concl=$(gh run view "$id" --json conclusion -q .conclusion)
echo "$new new, ${diff} differing result files; conclusion: $concl"
# exit codes: 0 success, 3 cancelled (e.g. by the guard), 1 anything else
case $concl in success) exit 0 ;; cancelled) exit 3 ;; *) exit 1 ;; esac
