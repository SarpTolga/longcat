#!/usr/bin/env bash
# ONE-TIME setup. Run this the FIRST time you create a pod with a fresh volume.
# It installs miniconda + the conda env + the repo + all deps (incl. flash-attn)
# directly onto the persistent /workspace volume. Re-running is safe (idempotent):
# anything already installed is skipped, so it doubles as a "repair" script.
#
# Expected runtime on first run: ~15-35 min (flash-attn compile is the slow part).
# After this, you NEVER run it again on the same volume.

set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ./env.sh

# Keep pip's build/temp dir on the SAME filesystem as its wheel cache (both under
# the volume). Otherwise flash-attn's prebuilt-wheel install fails with
# "[Errno 18] Invalid cross-device link" when it moves the wheel into the cache.
export TMPDIR="$WORKSPACE/tmp"
mkdir -p "$TMPDIR"

echo "==> [1/6] Miniconda (on volume, persists)"
if [ ! -d "$CONDA_DIR" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  rm -f /tmp/miniconda.sh
else
  echo "    already present, skipping"
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Newer Miniconda requires accepting the Anaconda channel Terms of Service before
# it will create an env non-interactively. Harmless if already accepted / absent.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "==> [2/6] Conda env '$ENV_NAME' (python 3.10)"
if ! conda env list | grep -q "/$ENV_NAME$"; then
  conda create -y -n "$ENV_NAME" python=3.10
else
  echo "    already present, skipping"
fi
conda activate "$ENV_NAME"

echo "==> [3/6] Clone LongCat-Video repo"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --single-branch --branch main https://github.com/meituan-longcat/LongCat-Video "$REPO_DIR"
else
  echo "    already present, skipping"
fi
cd "$REPO_DIR"

echo "==> [4/6] PyTorch 2.6.0 + CUDA 12.4"
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

echo "==> [5/6] flash-attn + project requirements (SLOW: flash-attn may compile)"
pip install ninja psutil packaging
pip install flash_attn==2.7.4.post1 --no-build-isolation
pip install -r requirements.txt
conda install -y -c conda-forge librosa ffmpeg

# Avatar extras are OPTIONAL (only for audio-driven avatar mode, NOT for
# T2V/I2V/VC/long). Upstream's requirements_avatar.txt pins `libsndfile1` — a
# SYSTEM library — via pip, which has no valid PyPI package and aborts the install.
# Install the system lib via apt, drop that bogus line, and keep avatar best-effort.
apt-get update -qq && apt-get install -y -qq libsndfile1 || true
grep -viE '^[[:space:]]*libsndfile1' requirements_avatar.txt > "$TMPDIR/req_avatar.txt" \
  || cp requirements_avatar.txt "$TMPDIR/req_avatar.txt"
pip install -r "$TMPDIR/req_avatar.txt" || echo "    (avatar extras skipped — fine for video generation)"

echo "==> [6/6] HuggingFace CLI (for weight downloads)"
pip install "huggingface_hub[cli]" hf_transfer

echo ""
echo "=========================================================="
echo " Setup complete. Next: ./02_download_weights.sh"
echo "=========================================================="
