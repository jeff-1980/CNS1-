#!/bin/bash
# usage: run49.sh RECORD_DIR TAG [audit_run49 args...]
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/49_alidi24__deep-learning-fault-diagnosis
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC MPLBACKEND=Agg PYTHONUNBUFFERED=1
export HF_HOME=/home/jeffwork/.cache/huggingface
mkdir -p ~/audit_ext/logs49 ~/audit_ext/work49
LOG=~/audit_ext/logs49/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
cd ~/audit_ext/work49 && ~/audit_ext/envs/49/bin/python $SRC/audit_run49.py "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep '^{' $LOG | cut -c1-300
grep -E 'Error|Traceback' -A3 $LOG | tail -10
