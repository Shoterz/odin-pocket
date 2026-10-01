#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

odin_checkpoint="${ODIN_CHECKPOINT:-runs/pocket/submission.pt}"
odin_port="${ODIN_PORT:-8788}"
odin_origin="${ODIN_PUBLIC_ORIGIN:-https://odinpocket.stocksuite.app}"

if [[ ! -f "$odin_checkpoint" ]]; then
  echo "Checkpoint not found: $odin_checkpoint. Download the release weights using the README instructions." >&2
  exit 1
fi

echo "Public demo: $odin_origin"
echo "Cloudflare Tunnel service: http://127.0.0.1:$odin_port"
echo 'CPU inference, four threads. Keep this process running to serve the demo.'
exec bash run.sh --checkpoint "$odin_checkpoint" --evidence submission/evidence \
  --device cpu --port "$odin_port" --public-origin "$odin_origin"
