#!/bin/bash
# usage: run30.sh RECORD_DIR TAG [audit_run30 args...]
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/30_zzzbbl__-HIT-CT-BiGRU-Fault-Diagnosis
export PYTHONPATH=~/audit_cns/code:$SRC/src AUDIT_RECORD_DIR=$REC MPLBACKEND=Agg
mkdir -p ~/audit_ext/logs30; cd $SRC
LOG=~/audit_ext/logs30/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
~/audit_ext/envs/30/bin/python $SRC/audit_run30.py --out ~/audit_ext/work30/$TAG "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep -o '{"model[^}]*}[^}]*}' $LOG | cut -c1-300
grep -E 'Error|Traceback' -A3 $LOG | tail -10
