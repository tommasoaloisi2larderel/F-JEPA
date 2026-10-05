#!/usr/bin/env bash
# LE VRAI TEST (plus tard) : leur train.py tel quel, encodeur appris de zéro, comme le papier sur TwoRoom.
# Lent : leur chargeur décompresse les images à chaque epoch sur le processeur (plusieurs heures).
# Usage : bash scripts/train_full.sh [EPOCHS]    (20 par défaut, comme le papier)
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$VENV/bin/activate"

EPOCHS="${1:-20}"
mkdir -p "$REPO_DIR/logs"
cd "$FJEPA_DIR"

# loss.sigreg.weight=0.1 : valeur du papier pour TwoRoom (0.09 n'est utilisé que pour Push-T).
python train.py data=tworoom \
  output_model_name="$RUN_NAME_FULL" \
  trainer.max_epochs="$EPOCHS" \
  loss.sigreg.weight=0.1 \
  2>&1 | tee "$REPO_DIR/logs/train_full.log"

echo "Poids sauvegardés dans : $STABLEWM_HOME/checkpoints/$RUN_NAME_FULL/"
