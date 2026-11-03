#!/bin/bash
# usage: run11.sh RECORD_DIR TAG [audit_run11 args...]
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/11_Vitoyoung__HETPM_intelligent_fault_diagnosis/HETPM_open
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC AUDIT_DATA_DIR=$HOME/JNU-Bearing-Dataset MPLBACKEND=Agg
mkdir -p ~/audit_ext/logs11 ~/audit_ext/work11/$TAG; cd ~/audit_ext/work11/$TAG
LOG=~/audit_ext/logs11/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
~/audit_ext/envs/11/bin/python $SRC/audit_run11.py "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep '^{' $LOG | cut -c1-400
grep -E 'Error|Traceback' -A2 $LOG | tail -8
