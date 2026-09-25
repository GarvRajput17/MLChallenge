#!/usr/bin/env bash
# Drive the EC2 work box from the laptop. Nothing heavy ever runs locally.
#
#   ./ec2.sh check                 verify SSH + report the box's specs
#   ./ec2.sh setup                 install python deps on the box
#   ./ec2.sh push                  rsync source code (fast, small)
#   ./ec2.sh push-data             rsync the 2.3 GB dataset (once; slow)
#   ./ec2.sh run <name> <cmd...>   launch inside tmux, detached and crash-safe
#   ./ec2.sh log <name>            tail that job's log
#   ./ec2.sh jobs                  list running tmux sessions
#   ./ec2.sh pull                  bring output/ and reports/ back
#
# Configure once:  export EC2_HOST=ubuntu@1.2.3.4   (or put it in ~/.ssh/config as 'mlbox')
set -euo pipefail

HOST="${EC2_HOST:-mlbox}"
KEY="${EC2_KEY:-$HOME/.ssh/ml_challenge_ec2}"
REMOTE_DIR="${EC2_DIR:-~/mlchallenge}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ssh_opts=(-i "$KEY" -o ServerAliveInterval=60 -o StrictHostKeyChecking=accept-new)
sh_() { ssh "${ssh_opts[@]}" "$HOST" "$@"; }

case "${1:-}" in
check)
    echo "== connecting to $HOST =="
    sh_ 'echo "host : $(hostname)"
         echo "cpu  : $(nproc) vCPU"
         echo "mem  : $(free -g 2>/dev/null | awk "/^Mem:/{print \$2\" GB\"}")"
         echo "disk : $(df -h --output=avail / | tail -1 | tr -d " ") free"
         echo "py   : $(python3 --version 2>&1)"
         echo "gpu  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
         echo "tmux : $(tmux -V 2>/dev/null || echo "NOT INSTALLED")"'
    ;;
setup)
    echo "== installing dependencies on $HOST =="
    sh_ "sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip tmux rsync htop"
    sh_ "mkdir -p $REMOTE_DIR"
    rsync -az -e "ssh ${ssh_opts[*]}" \
          "$LOCAL_ROOT/work/requirements.txt" "$HOST:$REMOTE_DIR/"
    sh_ "cd $REMOTE_DIR && pip3 install --quiet --break-system-packages -r requirements.txt && python3 -c 'import lightgbm,rapidfuzz,sparse_dot_topn,sklearn,pandas; print(\"deps ok\")'"
    ;;
push)
    echo "== syncing source -> $HOST:$REMOTE_DIR =="
    rsync -azv --delete -e "ssh ${ssh_opts[*]}" \
          --exclude '__pycache__' --exclude '*.pyc' \
          "$LOCAL_ROOT/work/src"    "$LOCAL_ROOT/work/scripts" \
          "$LOCAL_ROOT/work/requirements.txt" "$LOCAL_ROOT/work/README.md" \
          "$HOST:$REMOTE_DIR/work/"
    rsync -azv -e "ssh ${ssh_opts[*]}" \
          "$LOCAL_ROOT/student_resource/utils" \
          "$HOST:$REMOTE_DIR/student_resource/"
    ;;
push-data)
    echo "== syncing dataset (2.3 GB, once) =="
    rsync -azv --progress -e "ssh ${ssh_opts[*]}" \
          "$LOCAL_ROOT/student_resource/dataset" \
          "$HOST:$REMOTE_DIR/student_resource/"
    ;;
run)
    shift; name="$1"; shift
    echo "== launching '$name' on $HOST =="
    sh_ "mkdir -p $REMOTE_DIR/work/reports"
    sh_ "tmux new-session -d -s '$name' \
         \"cd $REMOTE_DIR/work && $* 2>&1 | tee reports/$name.log\"" \
        || { echo "session '$name' may already exist; use: ./ec2.sh jobs"; exit 1; }
    echo "detached. follow with:  ./ec2.sh log $name"
    ;;
log)
    sh_ "tail -f -n 60 $REMOTE_DIR/work/reports/$2.log"
    ;;
jobs)
    sh_ "tmux ls 2>/dev/null || echo '(no running jobs)'"
    ;;
pull)
    rsync -azv -e "ssh ${ssh_opts[*]}" \
          "$HOST:$REMOTE_DIR/work/output" "$HOST:$REMOTE_DIR/work/reports" \
          "$LOCAL_ROOT/work/"
    ;;
*)
    sed -n '2,20p' "$0"; exit 1;;
esac
