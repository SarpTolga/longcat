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

# The repo's streamlit app loads weights from ./weights — point it at the volume.
ln -sfn "$WEIGHTS_DIR" "$REPO_DIR/weights"

cd "$REPO_DIR"
echo "==> Launching Streamlit on 0.0.0.0:$PORT"
streamlit run ./run_streamlit.py \
  --server.address=0.0.0.0 \
  --server.port="$PORT" \
  --server.fileWatcherType none \
  --server.headless=true
