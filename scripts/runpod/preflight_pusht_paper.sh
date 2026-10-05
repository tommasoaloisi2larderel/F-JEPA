#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FJEPA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$FJEPA_ROOT"
source ./env.sh

cd Flow-JEPA
source .venv/bin/activate

python - <<'PY'
import h5py
import os
from pathlib import Path

import datasets
import pyarrow as pa
import stable_worldmodel as swm
import torch

stablewm_home = Path(os.environ["STABLEWM_HOME"])
dataset_path = stablewm_home / "datasets" / "pusht_expert_train.h5"
checkpoint_path = stablewm_home / "checkpoints" / "lewm-pusht" / "weights.pt"

print("FJEPA_HOME=", os.environ["FJEPA_HOME"])
print("STABLEWM_HOME=", stablewm_home)
print("datasets=", datasets.__version__)
print("pyarrow=", pa.__version__)
print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu=", torch.cuda.get_device_name(0))
    print("gpu_memory_gb=", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

if not dataset_path.is_file():
    raise FileNotFoundError(dataset_path)
if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)

with h5py.File(dataset_path, "r") as handle:
    print("h5_keys=", list(handle.keys())[:8])
    print("h5_size_bytes=", dataset_path.stat().st_size)

dataset = swm.data.load_dataset(
    "pusht_expert_train.h5",
    cache_dir=os.environ["LOCAL_DATASET_DIR"],
    num_steps=6,
    frameskip=5,
    keys_to_load=["pixels", "action", "proprio", "state"],
    keys_to_cache=["action", "proprio", "state"],
)
print("dataset_len=", len(dataset))
print("action_dim=", dataset.get_dim("action"))
sample = dataset[0]
print("sample_pixels_shape=", tuple(sample["pixels"].shape))
print("sample_action_shape=", tuple(sample["action"].shape))
print("checkpoint=", checkpoint_path)
print("preflight=OK")
PY
