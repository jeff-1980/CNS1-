#!/bin/bash
# usage: run19.sh RECORD_DIR TAG [audit_run19 args...]
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/19_kavianpour__Bearing-Fault-Diagnosis-DTCNN
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC TF_CPP_MIN_LOG_LEVEL=2
W=~/audit_ext/work19/$TAG; mkdir -p ~/audit_ext/logs19 $W; cd $W
LOG=~/audit_ext/logs19/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
~/audit_ext/envs/19/bin/python $SRC/audit_run19.py --data-root /home/jeffwork/论文CNS/data/cwru_12k_de "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep '^{' $LOG | cut -c1-400
grep -E 'Error|Traceback' -A3 $LOG | tail -10
