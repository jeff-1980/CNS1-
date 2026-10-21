#!/bin/bash
# usage: run14.sh RECORD_DIR TAG [audit_run14 args...]
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/14_asdvfghg__QCNN_for_bearing_diagnosis
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC WANDB_MODE=offline WANDB_SILENT=true
W=~/audit_ext/work14/$TAG; mkdir -p ~/audit_ext/logs14 $W; cd $W; ln -sfn $SRC/data data
LOG=~/audit_ext/logs14/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
~/audit_ext/envs/14/bin/python $SRC/audit_run14.py "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep '^{' $LOG | cut -c1-400
grep -E 'Error|Traceback' -A3 $LOG | tail -10
