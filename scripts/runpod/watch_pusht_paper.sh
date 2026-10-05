#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FJEPA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$FJEPA_ROOT"
source ./env.sh

LOG="$FJEPA_HOME/logs/pusht_paper_latest.log"
if [ ! -e "$LOG" ]; then
  echo "No latest log found at $LOG" >&2
  exit 1
fi

tail -F "$LOG"
