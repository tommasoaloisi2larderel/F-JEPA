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
  zstd

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
./download_dataset.sh pusht

mkdir -p "$STABLEWM_HOME/checkpoints/lewm-pusht"
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import os

target = Path(os.environ["STABLEWM_HOME"]) / "checkpoints" / "lewm-pusht"
snapshot_download(
    repo_id="quentinll/lewm-pusht",
    local_dir=target,
    local_dir_use_symlinks=False,
)
print(f"checkpoint_dir={target}")
print("checkpoint_files=" + ", ".join(sorted(p.name for p in target.iterdir())))
PY

test -f "$STABLEWM_HOME/datasets/pusht_expert_train.h5"
test -f "$STABLEWM_HOME/checkpoints/lewm-pusht/weights.pt"

echo "Prepared Push-T data and LeWM checkpoint under $STABLEWM_HOME"
