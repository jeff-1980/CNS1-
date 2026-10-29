#!/usr/bin/env bash
# Q-20261029-1 kill test launcher (WSL, CPU only). Run detached:
#   nohup bash run.sh > run.log 2>&1 &  echo $! > run.pid
set -euo pipefail
cd "$(dirname "$0")"
PY=~/bbvenv/bin/python
SN=$(find ~/data_weld -name 'S-N.json' | head -1)
PAR=$(find ~/data_weld -name 'parameter.json' | head -1)
echo "data: $SN | $PAR"
sha256sum runout_killtest.py runout_k4_xgb.py PREREG_Q20261029_runout.md
sha256sum -c SHA256SUMS
$PY -c "import numpy, scipy, sklearn, xgboost; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'xgboost', xgboost.__version__)"
mkdir -p out
echo "start K1-K3 $(date -Is)"
$PY -I runout_killtest.py "$SN" "$PAR" out > out/k1k3_stdout.txt 2>&1
echo "start K4 $(date -Is)"
$PY -I runout_k4_xgb.py "$SN" "$PAR" out > out/k4_stdout.txt 2>&1 || echo "K4 failed (exploratory)"
echo "ALL DONE $(date -Is)"
