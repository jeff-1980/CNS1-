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
# Non-interactive shells skip ~/.bashrc: load the key line from ~/.profile or ~/.bashrc if needed
if [ -z "${OPENALEX_API_KEY:-}" ]; then
  line="$(grep -h '^export OPENALEX_API_KEY=' "$HOME/.profile" "$HOME/.bashrc" 2>/dev/null | tail -1 || true)"
  [ -n "$line" ] && eval "$line"
fi
mkdir -p "$DATA" "$OUT"
case "${1:-}" in
  probe) "$VENV/bin/python" "$HERE/fetch_openalex.py" --out "$DATA" --probe ;;
  fetch) "$VENV/bin/python" "$HERE/fetch_openalex.py" --out "$DATA" --max-requests "${2:-9500}" ;;
  fetch_ext) "$VENV/bin/python" "$HERE/fetch_openalex.py" --out "$DATA" --ext --max-requests "${2:-9500}" ;;
  resume_ext)  # like resume, with energy/transport topics added; no model fit at the end
    for i in $(seq 1 "${2:-12}"); do
      set +e; "$VENV/bin/python" "$HERE/fetch_openalex.py" --out "$DATA" --ext --max-requests 9500; rc=$?; set -e
      [ "$rc" -eq 0 ] && break
      [ "$rc" -ne 3 ] && exit "$rc"
      echo "$(date '+%F %T') fetch budget exhausted (try $i), retry in 30 min"; sleep 1800
    done
    exit "${rc:-1}" ;;
  abstracts) "$VENV/bin/python" "$HERE/fetch_abstracts.py" "$DATA" ;;
  model) "$VENV/bin/python" "$HERE/model.py" --data "$DATA" --out "$OUT" ;;
  resume)  # keep retrying the fetch (exit 3 = budget used) every 30 min, then fit the model
    for i in $(seq 1 "${2:-10}"); do
      set +e; "$VENV/bin/python" "$HERE/fetch_openalex.py" --out "$DATA" --max-requests 9500; rc=$?; set -e
      [ "$rc" -eq 0 ] && break
      [ "$rc" -ne 3 ] && exit "$rc"
      echo "$(date '+%F %T') fetch budget exhausted (try $i), retry in 30 min"; sleep 1800
    done
    [ "${rc:-1}" -eq 0 ] || { echo "fetch still incomplete"; exit 3; }
    "$VENV/bin/python" "$HERE/model.py" --data "$DATA" --out "$OUT" ;;
  *) echo "usage: run_wsl.sh probe|fetch|fetch_ext|abstracts|model|resume|resume_ext [tries]"; exit 1 ;;
esac
