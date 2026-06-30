#!/usr/bin/env bash
# Launch the CONTINUOUS-clip UI (app_chain.py) — our auto-chaining browser app.
# Same idea as start.sh, but runs our custom app instead of the stock one.
# Open it from RunPod via the exposed HTTP port (default 8080).

set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ./env.sh
activate_env

if [ ! -d "$WEIGHTS_DIR/LongCat-Video" ]; then
  echo "ERROR: weights missing. Run ./runpod/02_download_weights.sh first." >&2
  exit 1
fi

# The app's default Model Dir is ./weights/LongCat-Video, resolved from the repo.
ln -sfn "$WEIGHTS_DIR" "$REPO_DIR/weights"
# Run from the repo dir so `import longcat_video...` resolves; app file stays here.
cp -f "$(pwd)/longcat_generate.py" "$REPO_DIR/longcat_generate.py"
cp -f "$(pwd)/app_chain.py" "$REPO_DIR/app_chain.py"

cd "$REPO_DIR"
echo "==> Launching continuous-clip UI on 0.0.0.0:$PORT"
echo "    In RunPod: Connect -> HTTP Service [Port $PORT]."
streamlit run ./app_chain.py \
  --server.address=0.0.0.0 \
  --server.port="$PORT" \
  --server.fileWatcherType none \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
