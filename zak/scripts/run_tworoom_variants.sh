#!/usr/bin/env bash
# Expérience TwoRoom « variantes » (encodeur appris), du début à la fin, sans surveillance :
#   1. vérifie qu'on peut écrire sur GitHub, AVANT les heures de calcul ;
#   2. run_variants.py : entraîne les variantes pas encore évaluées, un entraînement par GPU, et évalue
#      chaque modèle dès qu'il est fini (plus LeWM officiel, le hasard et Flow-JEPA, pour comparer) ;
#   3. publie runs/tworoom-variants/ sur GitHub (tableau, courbes, logs ; pas les modèles de test) ;
#   4. si tout a réussi : supprime le pod ; sinon : l'arrête seulement (rien n'est perdu, on peut reprendre).
# Usage : bash scripts/run_tworoom_variants.sh [EPOCHS]    (20 par défaut)
# Avant : bash scripts/setup.sh tworoom
set -uo pipefail  # pas de -e : si une étape échoue, on publie quand même les logs, puis on arrête le pod
source "$(dirname "$0")/env.sh"
source "$(dirname "$0")/pod_utils.sh"
source "$VENV/bin/activate"

EPOCHS="${1:-20}"
EXP=tworoom-variants
OUT="$REPO_DIR/runs/$EXP"
read -r -a OPTIONS <<< "${VARIANTS_OPTIONS:-}"  # options en plus pour run_variants.py (tests sur Mac)
N_GPU=$(python -c "import torch; print(torch.cuda.device_count())")
mkdir -p "$OUT"

# 1. Identité git, puis test d'écriture sur GitHub
git_identity
echo "démarré le $(date -u '+%Y-%m-%d %H:%M') UTC, $EPOCHS epochs, $N_GPU GPU" > "$OUT/STATUS.txt"
if ! git_publish "runs/$EXP" "Expérience $EXP : démarrage"; then
  push_error_help
  exit 1
fi
echo "Push vers GitHub : OK."

# 2. Entraînements et évaluations, répartis sur les GPU (voir run_variants.py)
python "$REPO_DIR/scripts/run_variants.py" --epochs "$EPOCHS" ${OPTIONS[@]+"${OPTIONS[@]}"}
STATUS=$?

# 3. Publication (run_variants.py a déjà copié résultats, courbes et logs dans runs/tworoom-variants/)
if [ "$STATUS" -eq 0 ]; then RESULT="terminé"; else RESULT="ÉCHEC (voir runs/$EXP/run_variants.log)"; fi
echo "$RESULT le $(date -u '+%Y-%m-%d %H:%M') UTC" >> "$OUT/STATUS.txt"
PUSHED=0
git_publish "runs/$EXP" "Expérience $EXP : $RESULT" && PUSHED=1 || echo "ERREUR : push impossible. Les résultats restent sur le pod, dans $OUT"

# 4. Suppression du pod si tout a réussi, sinon simple arrêt (rien n'est perdu)
if [ "$STATUS" -eq 0 ] && [ "$PUSHED" -eq 1 ]; then
  delete_pod
else
  stop_pod
fi
