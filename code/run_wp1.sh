#!/bin/bash
# WP1 运行链（预注册 257f20a3…）。GPU 串行。
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=../results; C=../results/ckpt_wp1; L=../logs
( while true; do nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,utilization.gpu,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader >> $L/wp1_gpu.csv; sleep 60; done ) &
MON=$!
echo "### stage1 JNU rpm id" 
for rpm in 600 800 1000; do ${PYTHON:-python} wp1_harness.py --exp jnu --rpm $rpm --models wdcnn --seeds 0 --out $R/wp1_jnu_rpmid_$rpm.json --ckpt-dir $C/rpmid || exit 1; done
${PYTHON:-python} select_rpm.py || exit 1
RPM=$(cat $R/wp1_jnu_rpm.txt)
echo "### stage2 python env"
${PYTHON:-python} wp1_harness.py --exp cwru_clean --models lstm,transformer --out $R/wp1_cwru_clean.json --ckpt-dir $C
${PYTHON:-python} wp1_harness.py --exp cwru_awgn --models cnn1d,wdcnn,drsn,lstm,transformer --out $R/wp1_cwru_awgn.json --ckpt-dir $C
${PYTHON:-python} wp1_harness.py --exp jnu --rpm $RPM --models cnn1d,wdcnn,drsn,lstm,transformer --out $R/wp1_jnu.json --ckpt-dir $C
echo "### stage3 mamba env"
${PYTHON_MAMBA:-python} wp1_harness.py --exp cwru_clean --models vibrmamba --out $R/wp1_cwru_clean_M.json --ckpt-dir $C
${PYTHON_MAMBA:-python} wp1_harness.py --exp cwru_m2 --models mamba2 --out $R/wp1_cwru_m2.json --ckpt-dir $C
${PYTHON_MAMBA:-python} wp1_harness.py --exp cwru_awgn --models vibrmamba,mamba2 --out $R/wp1_cwru_awgn_M.json --ckpt-dir $C
${PYTHON_MAMBA:-python} wp1_harness.py --exp jnu --rpm $RPM --models vibrmamba,mamba2 --out $R/wp1_jnu_M.json --ckpt-dir $C
kill $MON
echo "### WP1 CHAIN DONE"
