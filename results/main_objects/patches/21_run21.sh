#!/bin/bash
# usage: run21.sh RECORD_DIR TAG [audit_run21 args...]
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/21_asdvfghg__ClassBD
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC MPLBACKEND=Agg WANDB_MODE=disabled
W=~/audit_ext/work21/ClassBD; mkdir -p ~/audit_ext/logs21; cd $W
[ "$(ls data/Paderborn/N09_M07_F10 | wc -l)" = 380 ] || { echo "data incomplete"; exit 3; }
LOG=~/audit_ext/logs21/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
~/audit_ext/envs/21/bin/python $SRC/audit_run21.py "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep '^{' $LOG | cut -c1-400
grep -E 'Error|Traceback' -A3 $LOG | tail -10
