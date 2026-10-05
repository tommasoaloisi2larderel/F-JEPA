#!/usr/bin/env bash
# Installation unique sur le pod RunPod :
# outils système, environnement Python, code Flow-JEPA, dataset et checkpoint LeWM de l'environnement.
# Usage : bash scripts/setup.sh [tworoom|pusht]    (tworoom par défaut)
set -euo pipefail
source "$(dirname "$0")/env.sh"
ENV_NAME="${1:-tworoom}"
case "$ENV_NAME" in
  tworoom) LEWM_REPO=lewm-tworooms ;;
  pusht) LEWM_REPO=lewm-pusht ;;
  *) echo "Environnement inconnu : $ENV_NAME (tworoom ou pusht)"; exit 1 ;;
esac
LEWM_DIR="$STABLEWM_HOME/checkpoints/$LEWM_REPO"

echo "== 1/5 Outils système"
# zstd : décompresser le dataset. libegl1/libgl1 : rendu des environnements.
apt-get update -qq
apt-get install -y -qq zstd libegl1 libgl1 git curl > /dev/null

echo "== 2/5 Environnement Python"
if ! command -v uv > /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
# Cache des paquets sur le disque persistant : le disque du conteneur est petit.
export UV_CACHE_DIR="$REPO_DIR/.uv-cache"
[ -d "$VENV" ] || uv venv --python 3.10 "$VENV"
source "$VENV/bin/activate"
# PyTorch compilé pour CUDA 12.6 : la version par défaut (CUDA 13) exige un driver NVIDIA récent
# que beaucoup de machines RunPod n'ont pas. Attention : pas compatible avec les GPU Blackwell (RTX 5090, B200).
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# transformers<5 : la v5 a renommé les poids du ViT, le checkpoint LeWM ne se charge plus.
# On n'installe que les environnements nécessaires à TwoRoom et Push-T (pygame, pymunk, shapely, opencv).
# imageio[ffmpeg] : écriture des vidéos d'évaluation (déjà dans [format], écrit ici pour être explicite).
uv pip install \
  "stable-worldmodel[train,format]==0.1.1" \
  "transformers<5" \
  "imageio[ffmpeg]" \
  pygame pymunk shapely opencv-python-headless scikit-learn matplotlib
# On s'arrête tout de suite si PyTorch ne sait pas calculer sur ce GPU, avant les gros téléchargements.
# Un vrai calcul, pas seulement « le GPU est visible » : sur une carte trop récente (Blackwell),
# PyTorch voit le GPU mais n'a pas le code pour calculer dessus.
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch ne voit pas le GPU'; \
x = torch.ones(8, device='cuda'); assert (x * 2).sum().item() == 16; \
print('GPU :', torch.cuda.get_device_name(), '- calcul de test OK')"

echo "== 3/5 Code Flow-JEPA"
[ -d "$FJEPA_DIR" ] || git clone https://github.com/HuoYanchen/Flow-JEPA.git "$FJEPA_DIR"
git -C "$FJEPA_DIR" checkout -q "$FJEPA_COMMIT"

echo "== 4/5 Dataset $ENV_NAME"
mkdir -p "$STABLEWM_HOME/datasets"
TMP="$STABLEWM_HOME/datasets/.partiel"
if [ "$ENV_NAME" = pusht ] && [ ! -f "$STABLEWM_HOME/datasets/pusht_expert_train.h5" ]; then
  # Push-T : 13 Go à télécharger, environ 36 Go sur le disque. Un fichier .h5 compressé (pas une archive tar).
  rm -rf "$TMP" && mkdir -p "$TMP"
  curl -L --fail https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst \
    | zstd -dc > "$TMP/pusht_expert_train.h5"
  mv "$TMP/pusht_expert_train.h5" "$STABLEWM_HOME/datasets/pusht_expert_train.h5" && rm -rf "$TMP"
fi
if [ "$ENV_NAME" = tworoom ] && [ ! -f "$STABLEWM_HOME/datasets/tworoom.h5" ]; then
  # TwoRoom : 3,4 Go à télécharger, 12,8 Go sur le disque.
  # Téléchargement et décompression en flux : l'archive n'est jamais écrite sur le disque.
  # --no-same-owner : certains disques RunPod refusent de changer le propriétaire du fichier.
  # Décompression dans un dossier temporaire : un fichier à moitié écrit ne passe jamais pour complet.
  rm -rf "$TMP" && mkdir -p "$TMP"
  curl -L --fail https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst \
    | zstd -dc | tar -x --no-same-owner -C "$TMP"
  mv "$TMP/tworoom.h5" "$STABLEWM_HOME/datasets/tworoom.h5" && rm -rf "$TMP"
fi

echo "== 5/5 Checkpoint LeWM ($LEWM_REPO)"
mkdir -p "$LEWM_DIR"
for f in config.json weights.pt; do
  [ -f "$LEWM_DIR/$f" ] || curl -L --fail -o "$LEWM_DIR/$f" \
    "https://huggingface.co/quentinll/$LEWM_REPO/resolve/main/$f"
done

echo "Installation terminée ($ENV_NAME)."

