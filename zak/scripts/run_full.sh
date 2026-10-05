#!/usr/bin/env bash
# Le vrai test, du début à la fin, sans surveillance :
#   1. vérifie qu'on peut écrire sur GitHub, AVANT les heures de calcul ;
#   2. entraîne (train_e2e.py, avec torch.compile si le benchmark l'a conseillé) ;
#   3. évalue sur 50 épisodes (protocole de l'article) et sur 200 épisodes, en même temps ;
#   4. commit + push des résultats et du modèle final dans runs/<run>/ ;
#   5. si tout a réussi (entraînement, évaluations, push) : supprime le pod (plus rien n'est facturé) ;
#      sinon : l'arrête seulement (le disque /workspace est conservé pour récupérer ou reprendre).
# Usage : bash scripts/run_full.sh [EPOCHS]    (20 par défaut)
# Avant : python scripts/train_e2e.py --benchmark
set -uo pipefail  # pas de -e : si une étape échoue, on publie quand même les logs, puis on arrête le pod
source "$(dirname "$0")/env.sh"
source "$(dirname "$0")/pod_utils.sh"
source "$VENV/bin/activate"

EPOCHS="${1:-20}"
RUN="$RUN_NAME_FULL"
OUT="$REPO_DIR/runs/$RUN"
CKPT_DIR="$STABLEWM_HOME/checkpoints/$RUN"
EPISODES="${EVAL_EPISODES:-50 200}"  # nombres d'épisodes évalués (réglables pour les tests sur Mac)
read -r -a EVAL_EXTRA <<< "${EVAL_OPTIONS:-}"  # options Hydra en plus pour l'évaluation (tests sur Mac)
mkdir -p "$OUT" "$REPO_DIR/logs"

# Copie les résultats dans runs/<run>/, puis commit et push. $1 = message de commit.
publish() {
  cp "$REPO_DIR/logs/train_e2e_$RUN.log" "$REPO_DIR/logs/history_$RUN.json" "$REPO_DIR/logs/benchmark.json" "$OUT/" 2> /dev/null
  for n in $EPISODES; do
    src="$REPO_DIR/results/$RUN/ep$n"
    [ -d "$src" ] || continue
    mkdir -p "$OUT/eval_ep$n"
    cp "$src/summary.txt" "$OUT/eval_ep$n/" 2> /dev/null
    for d in "$src"/*/; do  # un dossier par configuration : on garde ses résultats, pas les vidéos
      [ -f "$d/tworoom_results.txt" ] && cp "$d/tworoom_results.txt" "$OUT/eval_ep$n/$(basename "$d").txt"
    done
  done
  git_publish "runs/$RUN" "$1"
}

# 1. Identité git, puis test d'écriture sur GitHub
git_identity
echo "démarré le $(date -u '+%Y-%m-%d %H:%M') UTC, $EPOCHS epochs" > "$OUT/STATUS.txt"
if ! publish "Run $RUN : démarrage"; then
  push_error_help
  exit 1
fi
echo "Push vers GitHub : OK. Lancement de l'entraînement."

# 2. Entraînement
COMPILE=""
grep -q '"compile": true' "$REPO_DIR/logs/benchmark.json" 2> /dev/null && COMPILE="--compile"
python "$REPO_DIR/scripts/train_e2e.py" --epochs "$EPOCHS" --run-name "$RUN" $COMPILE
STATUS=$?

# 3. Évaluations : 50 épisodes (l'article) et 200 épisodes (plus précis), en même temps
if [ "$STATUS" -eq 0 ]; then
  PIDS=()
  for n in $EPISODES; do
    bash "$REPO_DIR/scripts/eval_small.sh" "$n" "$RUN" ${EVAL_EXTRA[@]+"${EVAL_EXTRA[@]}"} \
      > "$REPO_DIR/logs/eval_ep$n.log" 2>&1 &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
  done
  cat "$REPO_DIR/results/$RUN"/ep*/summary.txt 2> /dev/null
fi

# 4. Modèle final (environ 92 Mo) dans le repo, puis publication
if [ -f "$CKPT_DIR/weights_epoch_$EPOCHS.pt" ]; then
  mkdir -p "$OUT/model"
  cp "$CKPT_DIR/weights_epoch_$EPOCHS.pt" "$CKPT_DIR/config.json" "$OUT/model/"
fi
if [ "$STATUS" -eq 0 ]; then RESULT="terminé"; else RESULT="ÉCHEC (voir train_e2e_$RUN.log et logs/ sur le pod)"; fi
echo "$RESULT le $(date -u '+%Y-%m-%d %H:%M') UTC" >> "$OUT/STATUS.txt"
PUSHED=0
publish "Run $RUN : $RESULT" && PUSHED=1 || echo "ERREUR : push impossible. Les résultats restent sur le pod, dans $OUT"

# 5. Suppression du pod si tout a réussi, sinon simple arrêt (rien n'est perdu)
if [ "$STATUS" -eq 0 ] && [ "$PUSHED" -eq 1 ]; then
  delete_pod
else
  stop_pod
fi
