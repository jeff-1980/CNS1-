#!/bin/bash
# usage: run39.sh RECORD_DIR TAG [audit_run39 args...]
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/39_CCSLab425__FMAA
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC MPLBACKEND=Agg
mkdir -p ~/audit_ext/logs39; cd $SRC
LOG=~/audit_ext/logs39/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
~/audit_ext/envs/39/bin/python $SRC/audit_run39.py "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep -o '{"seed[^}]*}[^}]*}' $LOG | cut -c1-300
grep -E 'Error|Traceback' -A3 $LOG | tail -10
