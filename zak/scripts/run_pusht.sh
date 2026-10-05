#!/usr/bin/env bash
# Expérience Push-T de bout en bout, sans surveillance, sur un pod à plusieurs GPU :
#   1. vérifie qu'on peut écrire sur GitHub, AVANT les heures de calcul ;
#   2. calcule une seule fois les vecteurs de toutes les images (encodeur LeWM gelé, partagé par tous) ;
#   3. entraîne les 6 modèles en même temps, répartis sur les GPU :
#        étape 0 : LeWM réentraîné | étape 1 : D | étape 2 : C et B | étape 3 : A (Flow-JEPA, 2 graines) ;
#   4. évalue tout en parallèle (protocole de l'article, 200 épisodes, poids de l'epoch 10) ;
#   5. trace les courbes de perte, puis publie runs/pusht-chain/ sur GitHub (avec les modèles de référence) ;
#   6. si tout a réussi : supprime le pod ; sinon : l'arrête seulement (rien n'est perdu, on peut reprendre).
# Usage : bash scripts/run_pusht.sh [EPOCHS]    (20 par défaut)
# Avant : bash scripts/setup.sh pusht
set -uo pipefail  # pas de -e : si une étape échoue, on publie quand même les logs, puis on arrête le pod
source "$(dirname "$0")/env.sh"
source "$(dirname "$0")/pod_utils.sh"
source "$VENV/bin/activate"

EPOCHS="${1:-20}"
EXP=pusht-chain
OUT="$REPO_DIR/runs/$EXP"
# Un entraînement = modèle:graine, dans l'ordre de la chaîne (voir train_frozen_fast.py).
TRAININGS=(lewm:0 ar_det:0 ar_flow:0 joint_det:0 joint_flow:0 joint_flow:1)
read -r -a EVAL_ARGS <<< "${EVAL_OPTIONS:-}"  # options en plus pour eval_jobs.py (tests sur Mac)
N_GPU=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")
RUNS=()
for t in "${TRAININGS[@]}"; do RUNS+=("${t%%:*}-pusht-s${t##*:}"); done
mkdir -p "$OUT/train_logs" "$REPO_DIR/logs"

# Copie les résultats dans runs/pusht-chain/, puis commit et push. $1 = message de commit.
publish() {
  for run in "${RUNS[@]}"; do
    cp "$REPO_DIR/logs/train_$run.log" "$REPO_DIR/logs/history_$run.json" "$OUT/train_logs/" 2> /dev/null
  done
  cp "$REPO_DIR/results/$EXP/summary.md" "$REPO_DIR/results/$EXP/summary.json" "$OUT/" 2> /dev/null
  git_publish "runs/$EXP" "$1"
}

# 1. Identité git, puis test d'écriture sur GitHub
git_identity
echo "démarré le $(date -u '+%Y-%m-%d %H:%M') UTC, $EPOCHS epochs, ${#TRAININGS[@]} entraînements sur $N_GPU GPU" > "$OUT/STATUS.txt"
if ! publish "Expérience $EXP : démarrage"; then
  push_error_help
  exit 1
fi
echo "Push vers GitHub : OK."

# 2. Les vecteurs de toutes les images, une seule fois (sinon les 6 entraînements les calculeraient en même temps)
python "$REPO_DIR/scripts/train_frozen_fast.py" --env pusht --encode-only
STATUS=$?

# 3. Les 6 entraînements en même temps, répartis sur les GPU (chacun reprend à sa dernière epoch s'il a été coupé)
if [ "$STATUS" -eq 0 ]; then
  PIDS=()
  for i in "${!TRAININGS[@]}"; do
    model="${TRAININGS[$i]%%:*}"
    seed="${TRAININGS[$i]##*:}"
    gpu=$((i % N_GPU))
    CUDA_VISIBLE_DEVICES=$gpu python "$REPO_DIR/scripts/train_frozen_fast.py" --env pusht --model "$model" \
      --seed "$seed" --epochs "$EPOCHS" --save-every 5 --run-name "${RUNS[$i]}" \
      > "$REPO_DIR/logs/stdout_${RUNS[$i]}.log" 2>&1 &
    PIDS+=($!)
    echo "lancé : ${RUNS[$i]} sur le GPU $gpu   (suivre : tail -f logs/train_${RUNS[$i]}.log)"
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
  done
fi

# 4. Toutes les évaluations, en parallèle
if [ "$STATUS" -eq 0 ]; then
  python "$REPO_DIR/scripts/eval_jobs.py" --env pusht --exp "$EXP" --runs "${RUNS[@]}" --final-epoch "$EPOCHS" \
    ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} || STATUS=1
fi

# 5. Courbes de perte, modèles de référence (LeWM réentraîné et A, graine 0 : environ 165 Mo), publication.
#    Pas les modèles de test (les autres variantes) : inutiles une fois évalués.
python "$REPO_DIR/scripts/plot_runs.py" --runs "${RUNS[@]}" --out "$OUT/courbes.png"
for run in "${RUNS[@]}"; do
  ckpt="$STABLEWM_HOME/checkpoints/$run"
  if [[ "$run" == lewm-pusht-s0 || "$run" == joint_flow-pusht-s0 ]] && [ -f "$ckpt/weights_epoch_$EPOCHS.pt" ]; then
    mkdir -p "$OUT/models/$run"
    cp "$ckpt/weights_epoch_$EPOCHS.pt" "$ckpt/config.json" "$OUT/models/$run/"
  fi
done
if [ "$STATUS" -eq 0 ]; then RESULT="terminé"; else RESULT="ÉCHEC (voir runs/$EXP/train_logs et logs/ sur le pod)"; fi
echo "$RESULT le $(date -u '+%Y-%m-%d %H:%M') UTC" >> "$OUT/STATUS.txt"
PUSHED=0
publish "Expérience $EXP : $RESULT" && PUSHED=1 || echo "ERREUR : push impossible. Les résultats restent sur le pod, dans $OUT"

# 6. Suppression du pod si tout a réussi, sinon simple arrêt (rien n'est perdu)
if [ "$STATUS" -eq 0 ] && [ "$PUSHED" -eq 1 ]; then
  delete_pod
else
  stop_pod
fi
