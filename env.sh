#!/usr/bin/env bash

export FJEPA_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$FJEPA_HOME/stable-wm"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export UV_CACHE_DIR="$FJEPA_HOME/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$FJEPA_HOME/.uv-python"
export XDG_CACHE_HOME="$FJEPA_HOME/.cache"
export MPLCONFIGDIR="$FJEPA_HOME/.cache/matplotlib"
export HF_HOME="$FJEPA_HOME/.cache/huggingface"

mkdir -p "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
mkdir -p "$XDG_CACHE_HOME/fontconfig" "$MPLCONFIGDIR" "$HF_HOME"
