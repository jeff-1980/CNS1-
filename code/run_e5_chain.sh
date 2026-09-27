#!/bin/bash
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; L=../logs
( while true; do nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,utilization.gpu,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader >> $L/e5_gpu.csv; sleep 60; done ) &
MON=$!
echo "### RESUME stage1 wdcnn,drsn" >> $L/e5.log; ${PYTHON:-python} e5_size_id.py --models wdcnn,drsn >> $L/e5.log 2>&1
echo "### RESUME wp5b" >> $L/wp5b.log; ${PYTHON:-python} wp5b_selective.py >> $L/wp5b.log 2>&1; ${PYTHON_MAMBA:-python} wp5b_selective.py --mamba >> $L/wp5b.log 2>&1; echo '### WP5b CHAIN DONE' >> $L/wp5b.log
echo "### stage3 lstm" >> $L/e5.log; ${PYTHON:-python} e5_size_id.py --models wdcnn,drsn,lstm >> $L/e5.log 2>&1
kill $MON; echo "### E5 CHAIN DONE" >> $L/e5.log
