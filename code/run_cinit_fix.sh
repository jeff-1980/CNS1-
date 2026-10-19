#!/bin/bash
# 等泄漏检查推理跑完再开训练，避免 GPU 争用；GPU 监控同时记录。
cd "$(dirname "$0")"; L=../logs; mkdir -p $L; P=${PYTHON:-python3}
until grep -q "ALL DONE\|Traceback" $L/rescore_leakfree.log; do sleep 30; done
( while true; do nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,utilization.gpu,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader >> $L/cinit_fix_gpu.csv; sleep 60; done ) &
MON=$!
$P cinit_fix.py --models wdcnn,drsn,cnn1d,lstm > $L/cinit_fix.log 2>&1
kill $MON
