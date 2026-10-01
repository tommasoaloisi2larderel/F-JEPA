#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

DATASET="${1:-}"
DATASET_DIR="$STABLEWM_HOME"

usage() {
  cat <<'EOF'
Usage:
  ./download_dataset.sh tworoom
  ./download_dataset.sh pusht
  ./download_dataset.sh reacher
  ./download_dataset.sh cube
  ./download_dataset.sh all

Downloads stream directly from Hugging Face into stable-wm without
keeping duplicate archive copies.
EOF
}

download_tworoom() {
  local out="$DATASET_DIR/tworoom.h5"
  [[ -f "$out" ]] && { echo "Already present: $out"; return; }
  curl -L "https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst" \
    | tar --zstd -xvf - -C "$DATASET_DIR"
}

download_pusht() {
  local out="$DATASET_DIR/pusht_expert_train.h5"
  [[ -f "$out" ]] && { echo "Already present: $out"; return; }
  curl -L "https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst" \
    | zstd -d -o "$out"
}

download_reacher() {
  local out="$DATASET_DIR/reacher.h5"
  [[ -f "$out" ]] && { echo "Already present: $out"; return; }
  curl -L "https://huggingface.co/datasets/quentinll/lewm-reacher/resolve/main/reacher.tar.zst" \
    | tar --zstd -xvf - -C "$DATASET_DIR"
}

download_cube() {
  local out="$DATASET_DIR/cube_single_expert.h5"
  [[ -f "$out" ]] && { echo "Already present: $out"; return; }
  curl -L "https://huggingface.co/datasets/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst" \
    | tar --zstd -xvf - -C "$DATASET_DIR"
}

mkdir -p "$DATASET_DIR"

case "$DATASET" in
  tworoom) download_tworoom ;;
  pusht) download_pusht ;;
  reacher) download_reacher ;;
  cube) download_cube ;;
  all)
    download_tworoom
    download_pusht
    download_reacher
    download_cube
    ;;
  ""|-h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown dataset: $DATASET" >&2
    usage >&2
    exit 2
    ;;
esac

echo
echo "Dataset directory:"
find "$DATASET_DIR" -maxdepth 1 -type f -name '*.h5' -print
