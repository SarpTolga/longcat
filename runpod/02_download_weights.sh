#!/usr/bin/env bash
# ONE-TIME (per volume) weight download. ~60GB total across 3 repos.
# Downloads to the persistent volume, so you only do this once.
# Safe to re-run: huggingface-cli resumes / skips already-downloaded files.
#
# All three are downloaded so every feature works:
#   - LongCat-Video            -> text2video, image2video, continuation, long video
#   - LongCat-Video-Avatar     -> original avatar
#   - LongCat-Video-Avatar-1.5 -> avatar v1.5 (distill + INT8, audio-driven)
#
# To save disk/time you can comment out the avatar lines you don't need.

set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ./env.sh
activate_env

mkdir -p "$WEIGHTS_DIR"

echo "==> Downloading LongCat-Video (core: t2v / i2v / continuation / long)"
huggingface-cli download meituan-longcat/LongCat-Video \
  --local-dir "$WEIGHTS_DIR/LongCat-Video"

echo "==> Downloading LongCat-Video-Avatar"
huggingface-cli download meituan-longcat/LongCat-Video-Avatar \
  --local-dir "$WEIGHTS_DIR/LongCat-Video-Avatar"

echo "==> Downloading LongCat-Video-Avatar-1.5 (recommended avatar variant)"
huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 \
  --local-dir "$WEIGHTS_DIR/LongCat-Video-Avatar-1.5"

echo ""
echo "=========================================================="
echo " Weights ready under $WEIGHTS_DIR"
echo " Next: ./start.sh   (launches the web UI)"
echo "=========================================================="
