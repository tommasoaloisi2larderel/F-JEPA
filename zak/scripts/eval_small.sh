#!/usr/bin/env bash
# Évaluation par planification (MPC + CEM) sur TwoRoom, sur les mêmes épisodes pour tous :
# politique aléatoire, LeWM et notre Flow-JEPA, avec images propres puis bruitées.
# Usage : bash scripts/eval_small.sh [NUM_EVAL] [RUN] [options Hydra en plus...]
#   NUM_EVAL : nombre d'épisodes (50 par défaut, comme le papier)
#   RUN      : run Flow-JEPA à évaluer ($RUN_NAME par défaut : le test encodeur gelé)
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$VENV/bin/activate"

NUM_EVAL="${1:-50}"
RUN="${2:-$RUN_NAME}"
EXTRA=("${@:3}")
RESULTS="$REPO_DIR/results/$RUN/ep$NUM_EVAL"
# Chaque évaluation n'a besoin que d'un cœur : on évite que les 5 se disputent tous les cœurs.
export OMP_NUM_THREADS=2
CONFIG_DIR="$FJEPA_DIR/config/eval"
# Dernier checkpoint du run (sort -V : tri numérique, epoch_10 après epoch_9).
FJEPA_CKPT=$(ls "$STABLEWM_HOME/checkpoints/$RUN"/weights_epoch_*.pt | sort -V | tail -1)
echo "Checkpoint Flow-JEPA : $FJEPA_CKPT"

cd "$FJEPA_DIR"

# Le simulateur n'utilise qu'un cœur du processeur par évaluation.
# On lance donc les 5 évaluations en même temps (&), chacune avec son propre log.
# $1 = nom du dossier de résultats, $2 = script, $3 = policy, $4 = bruit, $5 = vidéos (true/false)
PIDS=()
evaluate() {
  python "$2" --config-path="$CONFIG_DIR" --config-name=tworoom \
    policy="$3" \
    perturbation.enabled="$4" \
    output.save_video="$5" \
    eval.num_eval="$NUM_EVAL" \
    output.dir="$RESULTS/$1" \
    hydra.run.dir="$RESULTS/$1/hydra" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    > "$RESULTS/$1.log" 2>&1 &
  PIDS+=($!)
  echo "lancé : $1   (log : results/$RUN/ep$NUM_EVAL/$1.log)"
}

mkdir -p "$RESULTS"
START=$(date +%s)
# Vidéos seulement pour Flow-JEPA : elles montrent l'agent ; inutile de payer leur écriture pour les autres.
evaluate random      eval.py                          random        false false
evaluate lewm_clean  "$REPO_DIR/scripts/eval_lewm.py" "$LEWM_DIR"   false false
evaluate lewm_noise  "$REPO_DIR/scripts/eval_lewm.py" "$LEWM_DIR"   true  false
evaluate fjepa_clean eval.py                          "$FJEPA_CKPT" false true
evaluate fjepa_noise eval.py                          "$FJEPA_CKPT" true  true

echo "Les 5 évaluations tournent en parallèle. Suivre l'une d'elles : tail -f results/$RUN/ep$NUM_EVAL/fjepa_noise.log"
FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=$((FAILED + 1))
done

# Résumé affiché et enregistré dans summary.txt
{
  echo "== Résumé : taux de succès en %, sur $NUM_EVAL épisodes ($RUN), en $(( ($(date +%s) - START) / 60 )) min"
  for name in random lewm_clean lewm_noise fjepa_clean fjepa_noise; do
    rate=$(grep -so "'success_rate': [0-9.]*" "$RESULTS/$name/tworoom_results.txt" | tail -1 | cut -d' ' -f2 || true)
    printf "%-12s %s\n" "$name" "${rate:-ÉCHEC (voir results/$RUN/ep$NUM_EVAL/$name.log)}"
  done
} | tee "$RESULTS/summary.txt"
[ "$FAILED" -eq 0 ] || { echo "$FAILED évaluation(s) en échec."; exit 1; }
