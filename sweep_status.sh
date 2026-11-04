#!/usr/bin/env bash
# Progress bar for the latest (or a given) sweep run: seeds finished, guard verdicts, elapsed.
#   ./sweep_status.sh [RUN_ID]
set -euo pipefail
id=${1:-$(gh run list --workflow sweep.yml --limit 1 --json databaseId -q '.[0].databaseId')}
read -r status created < <(gh run view "$id" --json status,createdAt -q '"\(.status) \(.createdAt)"')
read -r done total < <(gh run view "$id" --json jobs -q \
  '[.jobs[] | select(.name|startswith("run"))] | "\([.[] | select(.status=="completed")] | length) \(length)"')
width=30
fill=$(( total ? done * width / total : 0 ))
bar=$(printf '%*s' "$fill" '' | tr ' ' '#')$(printf '%*s' $((width - fill)) '' | tr ' ' '-')
mins=$(( ($(date -u +%s) - $(date -u -d "$created" +%s)) / 60 ))
guard=.sweep_runs/guard_$id.log
verdict=""
if [ -f "$guard" ]; then
  g=$(grep -c ': gated' "$guard" || true)
  f=$(grep -c ': FAIL' "$guard" || true)
  verdict=" | gate: $g pass, $f fail"
  grep -q 'cancelling' "$guard" && verdict="$verdict (guard cancelled)"
fi
echo "run $id [$bar] $done/$total seeds done | ${mins} min | $status$verdict"
