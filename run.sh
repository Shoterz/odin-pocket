#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ -n "${ODIN_PYTHON:-}" ]]; then
  odin_python="$ODIN_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  odin_python=.venv/bin/python
elif [[ -x ../project/.venv/bin/python ]]; then
  odin_python=../project/.venv/bin/python
else
  echo 'Create .venv and install requirements as described in README.md.' >&2
  exit 1
fi
exec "$odin_python" -m odin.server "$@"
