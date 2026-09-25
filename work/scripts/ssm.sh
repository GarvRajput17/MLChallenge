#!/usr/bin/env bash
# Run shell commands on the EC2 box through Systems Manager -- no SSH key needed.
#   ./ssm.sh run  "<command>"        run, wait, print stdout/stderr
#   ./ssm.sh bg   <name> "<command>" launch detached under tmux, returns immediately
#   ./ssm.sh log  <name> [lines]     tail that job's log
#   ./ssm.sh jobs                    list running tmux sessions
set -euo pipefail
export PATH=/usr/local/bin:$PATH
ID="${EC2_ID:-i-064c2fde9d13d9f8c}"
DIR="${EC2_DIR:-/home/ubuntu/mlchallenge}"

send() {   # $1 = shell command, $2 = timeout seconds
  local cid
  cid=$(aws ssm send-command --instance-ids "$ID" \
        --document-name AWS-RunShellScript \
        --parameters "commands=[$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")]" \
        --timeout-seconds "${2:-3600}" \
        --query "Command.CommandId" --output text)
  local status
  for _ in $(seq 1 "${POLL:-240}"); do
    status=$(aws ssm get-command-invocation --command-id "$cid" --instance-id "$ID" \
             --query Status --output text 2>/dev/null || echo Pending)
    case "$status" in Success|Failed|Cancelled|TimedOut) break;; esac
    sleep 5
  done
  aws ssm get-command-invocation --command-id "$cid" --instance-id "$ID" \
      --query "StandardOutputContent" --output text
  local err
  err=$(aws ssm get-command-invocation --command-id "$cid" --instance-id "$ID" \
        --query "StandardErrorContent" --output text)
  [ -n "$err" ] && [ "$err" != "None" ] && { echo "--- stderr ---"; echo "$err"; } || true
  echo "[$status]"
}

case "${1:-}" in
run)  send "$2" "${3:-3600}" ;;
bg)   send "mkdir -p $DIR/work/reports && cd $DIR/work && \
            tmux new-session -d -s '$2' '$3 2>&1 | tee reports/$2.log' && \
            echo launched '$2'" 120 ;;
log)  send "tail -n ${3:-40} $DIR/work/reports/$2.log" 120 ;;
jobs) send "tmux ls 2>/dev/null || echo '(no jobs)'" 60 ;;
*)    sed -n '2,7p' "$0"; exit 1 ;;
esac
