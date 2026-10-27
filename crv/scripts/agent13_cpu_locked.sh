#!/bin/bash
set -eo pipefail
ROOT=/exp/mu2e/data/users/ram2aq/agent_locks
LOCK=$ROOT/$(hostname -s)
mkdir -p "$ROOT"
mkdir "$LOCK" || { echo 'Host lock is held; not launching'; exit 3; }
printf 'crv-agent13 pid=%s %s\n' "$$" "$(date -Is)" > "$LOCK/task"
cleanup() {
    if grep -q "crv-agent13 pid=$$ " "$LOCK/task"; then
        rm "$LOCK/task"
        rmdir "$LOCK"
    fi
}
trap cleanup EXIT
HERE=$(cd "$(dirname "$0")" && pwd)
bash "$HERE/agent13_local_chain.sh" cpu "$1" "${2:-5000}" 0
