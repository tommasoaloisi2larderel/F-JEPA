#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FJEPA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$FJEPA_ROOT"
source ./env.sh

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  curl \
  ca-certificates \
  git \
  swig \
  zstd \
  tmux

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd Flow-JEPA
if [ ! -d .venv ]; then
  uv venv --python=3.10
fi
source .venv/bin/activate
uv pip install "stable-worldmodel[train,env,format]" huggingface_hub
uv pip install "datasets==2.19.2" "pyarrow==20.0.0"

cd ..
./download_dataset.sh tworoom

test -f "$STABLEWM_HOME/datasets/tworoom.h5"

echo "Prepared TwoRoom data under $STABLEWM_HOME"
