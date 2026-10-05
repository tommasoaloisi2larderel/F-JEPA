#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FJEPA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$FJEPA_ROOT"
source ./env.sh

LOCAL_FAST_ROOT="${LOCAL_FAST_ROOT:-/tmp/fjepa-fast/stable-wm}"
SRC="$STABLEWM_HOME/datasets/pusht_expert_train.h5"
DST="$LOCAL_FAST_ROOT/datasets/pusht_expert_train.h5"

mkdir -p "$LOCAL_FAST_ROOT/datasets"

if [ ! -f "$SRC" ]; then
  echo "Missing source dataset: $SRC" >&2
  exit 1
fi

src_size="$(stat -c %s "$SRC")"
if [ -f "$DST" ]; then
  dst_size="$(stat -c %s "$DST")"
  if [ "$src_size" = "$dst_size" ]; then
    echo "Local Push-T dataset already staged: $DST"
    exit 0
  fi
  echo "Removing incomplete local dataset: $DST"
  rm -f "$DST"
fi

echo "Staging Push-T dataset from network volume to local disk..."
echo "Source: $SRC"
echo "Target: $DST"
cp "$SRC" "$DST.tmp"
sync "$DST.tmp"
mv "$DST.tmp" "$DST"
echo "Local Push-T dataset staged: $DST"
