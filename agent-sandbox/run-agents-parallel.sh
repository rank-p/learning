#!/usr/bin/env bash
# run-agents-parallel.sh — launch multiple agents concurrently from a manifest
#
# Usage:
#   ./run-agents-parallel.sh agents.conf
#
# agents.conf format (tab or space separated, one agent per line):
#   <agent-id>  <aws-profile>  <workspace>  <command...>
#
# Example agents.conf:
#   agent-1  prod-account   /workspaces/project-a  python agent.py --task summarize
#   agent-2  staging-acct   /workspaces/project-b  python agent.py --task review
#   agent-3  dev-account    /workspaces/project-c  bash worker.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${1:-agents.conf}"

[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest not found: $MANIFEST" >&2; exit 1; }

declare -a PIDS=()
declare -A PID_TO_AGENT=()
LOG_DIR="${AGENT_SANDBOX_LOGS:-/tmp/agent-sandbox-logs}"
mkdir -p "$LOG_DIR"

cleanup() {
  echo "INFO: shutting down remaining agents..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait
}
trap cleanup EXIT INT TERM

while IFS= read -r line || [[ -n "$line" ]]; do
  # skip comments and blank lines
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  [[ -z "${line//[[:space:]]/}" ]]  && continue

  read -r agent_id aws_profile workspace cmd_rest <<< "$line"
  log_file="$LOG_DIR/${agent_id}.log"

  echo "INFO: starting $agent_id (profile=$aws_profile workspace=$workspace)"

  # shellcheck disable=SC2086
  "$SCRIPT_DIR/run-agent.sh" \
    --agent-id    "$agent_id" \
    --aws-profile "$aws_profile" \
    --workspace   "$workspace" \
    -- $cmd_rest \
    >"$log_file" 2>&1 &

  pid=$!
  PIDS+=("$pid")
  PID_TO_AGENT[$pid]="$agent_id"
  echo "INFO: $agent_id started (pid=$pid, log=$log_file)"

done < "$MANIFEST"

echo "INFO: all agents launched, waiting..."

# wait for each and report exit status
failed=0
for pid in "${PIDS[@]}"; do
  agent="${PID_TO_AGENT[$pid]}"
  if wait "$pid"; then
    echo "OK:   $agent (pid=$pid) finished successfully"
  else
    code=$?
    echo "FAIL: $agent (pid=$pid) exited with code $code"
    failed=$((failed + 1))
  fi
done

[[ $failed -eq 0 ]] && echo "INFO: all agents completed successfully"
exit $failed
