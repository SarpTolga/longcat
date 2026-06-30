#!/usr/bin/env bash
# RESPIN ENTRYPOINT. This is the ONLY script you run on every new pod after
# the one-time setup. It activates the env and launches the Streamlit web UI.
#
# Open it from RunPod via the HTTP port you exposed (default 8080) ->
# "Connect" -> the proxied URL.

set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ./env.sh
activate_env

# Fail fast with a clear message if the one-time setup/download wasn't done,
# instead of launching a UI that errors the moment you click Generate.
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "ERROR: repo missing at $REPO_DIR. Run ./runpod/01_setup.sh first." >&2
  exit 1
fi
if [ ! -d "$WEIGHTS_DIR/LongCat-Video" ]; then
  echo "ERROR: weights missing at $WEIGHTS_DIR/LongCat-Video." >&2
  echo "       Run ./runpod/02_download_weights.sh first." >&2
  exit 1
fi

# The repo's streamlit app loads weights from ./weights — point it at the volume.
ln -sfn "$WEIGHTS_DIR" "$REPO_DIR/weights"

cd "$REPO_DIR"
echo "==> Launching Streamlit on 0.0.0.0:$PORT"
echo "    In RunPod: Connect -> HTTP Service [Port $PORT] to open the UI."
# CORS/XSRF disabled: required so Streamlit's websocket works through RunPod's
# HTTP reverse proxy (otherwise the UI loads blank or won't connect).
streamlit run ./run_streamlit.py \
  --server.address=0.0.0.0 \
  --server.port="$PORT" \
  --server.fileWatcherType none \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
