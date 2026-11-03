#!/bin/bash
# usage: run13.sh RECORD_DIR TAG [audit_run13 args...]
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/13_asdvfghg__BearingPGA-Net
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC WANDB_MODE=offline WANDB_SILENT=true
W=~/audit_ext/work13/$TAG; mkdir -p ~/audit_ext/logs13 $W; cd $W; ln -sfn $SRC/data data
LOG=~/audit_ext/logs13/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
~/audit_ext/envs/13/bin/python $SRC/audit_run13.py "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep '^{' $LOG | cut -c1-400
grep -E 'Error|Traceback' -A3 $LOG | tail -10
