#!/bin/bash
# WP2 运行链（预注册 ba370e92…）。GPU 串行；已完成实例自动跳过，可断点续跑。
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=../results; C=../results/ckpt_wp2; L=../logs
( while true; do nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,utilization.gpu,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader >> $L/wp2_gpu.csv; sleep 60; done ) &
MON=$!
echo "### stage R primary"; ${PYTHON:-python} pu_wp2.py --block R --models wdcnn,drsn,lstm --out $R/wp2_R.json --ckpt-dir $C
echo "### stage A primary"; ${PYTHON:-python} pu_wp2.py --block A --models wdcnn,drsn,lstm --out $R/wp2_A.json --ckpt-dir $C
echo "### stage R single";  ${PYTHON:-python} pu_wp2.py --block R --models cnn1d,transformer --arms R-L1,R-L12a --out $R/wp2_R_single.json --ckpt-dir $C
kill $MON
echo "### WP2 CHAIN DONE"
