# Chemins et noms partagés. Chargé par les autres scripts avec `source`.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_DIR/.venv"

# Dossier des données et checkpoints. /workspace est le disque persistant de RunPod.
# (Les scripts Python utilisent la même valeur par défaut.)
export STABLEWM_HOME="${STABLEWM_HOME:-/workspace/stable-wm}"
# train.py lit cette variable pour trouver les datasets.
export LOCAL_DATASET_DIR="$STABLEWM_HOME"

# Code officiel Flow-JEPA, figé sur un commit précis pour que le test soit reproductible.
FJEPA_DIR="$REPO_DIR/third_party/Flow-JEPA"
FJEPA_COMMIT=ab73e7c50435e3a1fffb4d3f6d175445e57a2f4e

# Checkpoint LeWM officiel (sert d'encodeur gelé et de baseline).
LEWM_DIR="$STABLEWM_HOME/checkpoints/lewm-tworooms"

# Noms des runs Flow-JEPA. Les poids vont dans $STABLEWM_HOME/checkpoints/<nom>/.
RUN_NAME=fjepa-tworoom-frozen     # test rapide, encodeur gelé (nom par défaut de train_frozen_fast.py)
RUN_NAME_FULL=fjepa-tworoom-full  # vrai test, encodeur appris (nom par défaut de train_e2e.py)

# Auteur des commits faits automatiquement par le pod (run_full.sh).
GIT_NAME=EloZiko
GIT_EMAIL=bougglib@gmail.com
