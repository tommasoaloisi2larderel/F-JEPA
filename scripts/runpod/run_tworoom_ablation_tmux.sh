#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-fjepa-tworoom-ablation}"
EPOCHS="${EPOCHS:-20}"
NUM_EVAL="${NUM_EVAL:-50}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FJEPA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$FJEPA_ROOT"
source ./env.sh

mkdir -p "$FJEPA_HOME/logs" "$FJEPA_HOME/results/tworoom_ablation"
LOG="$FJEPA_HOME/logs/tworoom_ablation_$(date -u +%Y%m%dT%H%M%SZ).log"
LATEST="$FJEPA_HOME/logs/tworoom_ablation_latest.log"
SUMMARY="$FJEPA_HOME/results/tworoom_ablation/summary_$(date -u +%Y%m%dT%H%M%SZ).txt"
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
cd Flow-JEPA
source .venv/bin/activate
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

echo \"[$(date -u +%FT%TZ)] starting TwoRoom JEPA ablation\" | tee -a \"$LOG\"
echo \"log=$LOG\" | tee -a \"$LOG\"
echo \"summary=$SUMMARY\" | tee -a \"$LOG\"
echo \"epochs=$EPOCHS num_eval=$NUM_EVAL seed=$SEED batch_size=$BATCH_SIZE limit_train_batches=$LIMIT_TRAIN_BATCHES limit_val_batches=$LIMIT_VAL_BATCHES\" | tee -a \"$LOG\"
nvidia-smi | tee -a \"$LOG\"

printf \"TwoRoom JEPA ablation\nEPOCHS=%s\nNUM_EVAL=%s\nSEED=%s\nBATCH_SIZE=%s\nLIMIT_TRAIN_BATCHES=%s\nLIMIT_VAL_BATCHES=%s\n\n\" \"$EPOCHS\" \"$NUM_EVAL\" \"$SEED\" \"$BATCH_SIZE\" \"$LIMIT_TRAIN_BATCHES\" \"$LIMIT_VAL_BATCHES\" > \"$SUMMARY\"

train_overrides=(
  trainer.max_epochs=\"$EPOCHS\"
  loader.batch_size=\"$BATCH_SIZE\"
  seed=\"$SEED\"
)
if [ -n \"$LIMIT_TRAIN_BATCHES\" ]; then
  train_overrides+=(+trainer.limit_train_batches=\"$LIMIT_TRAIN_BATCHES\")
fi
if [ -n \"$LIMIT_VAL_BATCHES\" ]; then
  train_overrides+=(+trainer.limit_val_batches=\"$LIMIT_VAL_BATCHES\")
fi

configs=(
  tworoom_flow:flow-jepa-tworoom-flow
  tworoom_deterministic:flow-jepa-tworoom-deterministic
  tworoom_residual_flow:flow-jepa-tworoom-residual-flow
  tworoom_learned_source_flow:flow-jepa-tworoom-learned-source-flow
)

for item in \"\${configs[@]}\"; do
  config=\"\${item%%:*}\"
  run_name=\"\${item##*:}\"
  ckpt=\"$STABLEWM_HOME/checkpoints/\$run_name/weights_epoch_${EPOCHS}.pt\"
  echo \"[$(date -u +%FT%TZ)] training \$config -> \$run_name\" | tee -a \"$LOG\"
  python train.py --config-name \"\$config\" \
    \"\${train_overrides[@]}\" \
    2>&1 | tee -a \"$LOG\"
  train_status=\${PIPESTATUS[0]}
  if [ \"\$train_status\" -ne 0 ]; then
    echo \"training failed for \$config with status=\$train_status\" | tee -a \"$LOG\"
    exit \"\$train_status\"
  fi
  test -f \"\$ckpt\"

  clean_dir=\"$FJEPA_HOME/results/tworoom_ablation/\$run_name/clean\"
  noisy_dir=\"$FJEPA_HOME/results/tworoom_ablation/\$run_name/noisy\"
  mkdir -p \"\$clean_dir\" \"\$noisy_dir\"

  echo \"[$(date -u +%FT%TZ)] clean eval \$run_name\" | tee -a \"$LOG\"
  python eval.py --config-name=tworoom \
    policy=\"\$ckpt\" \
    seed=\"$SEED\" \
    flow_seed=\"$SEED\" \
    eval.num_eval=\"$NUM_EVAL\" \
    output.dir=\"\$clean_dir\" \
    output.filename=results_clean.txt \
    output.save_video=false \
    2>&1 | tee -a \"$LOG\"
  clean_status=\${PIPESTATUS[0]}
  if [ \"\$clean_status\" -ne 0 ]; then
    echo \"clean eval failed for \$run_name with status=\$clean_status\" | tee -a \"$LOG\"
    exit \"\$clean_status\"
  fi

  echo \"[$(date -u +%FT%TZ)] noisy eval \$run_name\" | tee -a \"$LOG\"
  python eval.py --config-name=tworoom \
    policy=\"\$ckpt\" \
    seed=\"$SEED\" \
    flow_seed=\"$SEED\" \
    eval.num_eval=\"$NUM_EVAL\" \
    perturbation.enabled=true \
    output.dir=\"\$noisy_dir\" \
    output.filename=results_noisy.txt \
    output.save_video=false \
    2>&1 | tee -a \"$LOG\"
  noisy_status=\${PIPESTATUS[0]}
  if [ \"\$noisy_status\" -ne 0 ]; then
    echo \"noisy eval failed for \$run_name with status=\$noisy_status\" | tee -a \"$LOG\"
    exit \"\$noisy_status\"
  fi

  {
    echo \"==== \$run_name ====\"
    echo \"clean:\"
    grep -A2 \"==== RESULTS ====\" \"\$clean_dir/results_clean.txt\" | tail -n 2
    echo \"noisy:\"
    grep -A2 \"==== RESULTS ====\" \"\$noisy_dir/results_noisy.txt\" | tail -n 2
    echo
  } >> \"$SUMMARY\"
done

echo \"[$(date -u +%FT%TZ)] finished TwoRoom ablation\" | tee -a \"$LOG\"
cat \"$SUMMARY\" | tee -a \"$LOG\"
'"

cat <<EOF
Started tmux session: $SESSION
Log file: $LOG
Latest log symlink: $LATEST
Summary file: $SUMMARY

Attach:
  tmux attach -t $SESSION

Tail log:
  tail -F $LATEST
EOF
