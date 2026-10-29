#!/usr/bin/env bash
# Q-20261029-1 K5 launcher (WSL, CPU). Run from the K1-K3 killtest dir (needs out/rows.json):
#   nohup bash run_k5.sh > run_k5.log 2>&1 &  echo $! > run_k5.pid
set -euo pipefail
cd "$(dirname "$0")"
PY=~/bbvenv/bin/python
SN=$(find ~/data_weld -name 'S-N.json' | head -1)
PAR=$(find ~/data_weld -name 'parameter.json' | head -1)
echo "data: $SN | $PAR"
sha256sum -c SHA256SUMS
sha256sum -c SHA256SUMS_K5
test -f out/rows.json
echo "cpus $(nproc)"
echo "start K5 $(date -Is)"
$PY k5_simtruth.py "$SN" "$PAR" out > out/k5_stdout.txt 2>&1
echo "ALL DONE $(date -Is)"
