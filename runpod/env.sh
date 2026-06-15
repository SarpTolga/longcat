#!/usr/bin/env bash
# Shared paths/vars for all LongCat-Video scripts.
# Everything lives under the persistent network volume (/workspace) so it
# survives pod deletion. Change WORKSPACE only if your volume mounts elsewhere.

export WORKSPACE="${WORKSPACE:-/workspace}"
export CONDA_DIR="$WORKSPACE/miniconda3"        # miniconda installed ON the volume -> persists
export ENV_NAME="longcat-video"                  # conda env name (lives under $CONDA_DIR)
export REPO_DIR="$WORKSPACE/LongCat-Video"       # cloned repo (persists)
export WEIGHTS_DIR="$WORKSPACE/weights"          # ~60GB of model weights (persists)
export HF_HOME="$WORKSPACE/hf_cache"             # HF cache on the volume too
export PORT="${PORT:-8080}"                       # Streamlit port (expose this in RunPod)

# Faster HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Helper: activate conda env (source this after sourcing env.sh)
activate_env() {
  # shellcheck disable=SC1091
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
}
