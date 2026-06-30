#!/usr/bin/env bash
# Convenience wrapper for longcat_generate.py — custom-prompt generation from the
# CLI, with auto-chained long (continuous) video. Activates the env, points at the
# weights on the volume, and launches via torchrun.
#
# Examples:
#   ./runpod/generate.sh --mode t2v  --prompt "a red fox in a snowy forest" --output fox.mp4
#   ./runpod/generate.sh --mode long --segments 5 --prompt "drone over a misty valley" --output valley.mp4
#   GPUS=2 ./runpod/generate.sh --mode long --context-parallel-size 2 --prompt "..." --output out.mp4
#
# Any flag understood by longcat_generate.py is passed straight through.

set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ./env.sh
activate_env

if [ ! -d "$WEIGHTS_DIR/LongCat-Video" ]; then
  echo "ERROR: weights missing. Run ./runpod/02_download_weights.sh first." >&2
  exit 1
fi

# Make the default ./weights/LongCat-Video path resolve from inside the repo.
ln -sfn "$WEIGHTS_DIR" "$REPO_DIR/weights"
cp -f "$(pwd)/longcat_generate.py" "$REPO_DIR/longcat_generate.py"

cd "$REPO_DIR"
GPUS="${GPUS:-1}"
echo "==> Generating with $GPUS GPU(s)"
torchrun --nproc_per_node="$GPUS" longcat_generate.py "$@"
