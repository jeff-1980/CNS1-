#!/usr/bin/env bash
# Usage (WSL): bash run_wsl.sh probe | fetch | model
# Data -> /mnt/d/audit_data/topic_forecast (tens of MB); results -> ~/topic_forecast_out
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA=/mnt/d/audit_data/topic_forecast
OUT=$HOME/topic_forecast_out
VENV=$HOME/tfvenv
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q numpy scipy pandas scikit-learn lightgbm
fi
mkdir -p "$DATA" "$OUT"
case "${1:-}" in
  probe) "$VENV/bin/python" "$HERE/fetch_openalex.py" --out "$DATA" --probe ;;
  fetch) "$VENV/bin/python" "$HERE/fetch_openalex.py" --out "$DATA" --max-requests "${2:-9500}" ;;
  model) "$VENV/bin/python" "$HERE/model.py" --data "$DATA" --out "$OUT" ;;
  *) echo "usage: run_wsl.sh probe|fetch|model"; exit 1 ;;
esac
