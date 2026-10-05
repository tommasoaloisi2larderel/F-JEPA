#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-fjepa-pusht-paper}"
LOCAL_FAST_ROOT="${LOCAL_FAST_ROOT:-/tmp/fjepa-fast/stable-wm}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FJEPA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$FJEPA_ROOT"
source ./env.sh

mkdir -p "$FJEPA_HOME/logs"
LOG="$FJEPA_HOME/logs/pusht_paper_fast_$(date -u +%Y%m%dT%H%M%SZ).log"
LATEST="$FJEPA_HOME/logs/pusht_paper_latest.log"
ln -sfn "$LOG" "$LATEST"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 2
fi

tmux new-session -d -s "$SESSION" "bash -lc '
set -euo pipefail
cd \"$FJEPA_HOME\"
source ./env.sh
export LOCAL_DATASET_DIR=\"$LOCAL_FAST_ROOT\"
cd Flow-JEPA
source .venv/bin/activate
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
echo \"[$(date -u +%FT%TZ)] starting F-JEPA Push-T paper run with local staged dataset\" | tee -a \"$LOG\"
echo \"log=$LOG\" | tee -a \"$LOG\"
echo \"LOCAL_DATASET_DIR=$LOCAL_FAST_ROOT\" | tee -a \"$LOG\"
nvidia-smi | tee -a \"$LOG\"
python train.py --config-name pusht_paper 2>&1 | tee -a \"$LOG\"
status=\${PIPESTATUS[0]}
echo \"[$(date -u +%FT%TZ)] finished with status=\$status\" | tee -a \"$LOG\"
exit \$status
'"

cat <<EOF
Started tmux session: $SESSION
Log file: $LOG
Latest log symlink: $LATEST

Attach:
  tmux attach -t $SESSION

Tail log:
  tail -F $LATEST

GPU watch:
  watch -n 2 nvidia-smi
EOF
