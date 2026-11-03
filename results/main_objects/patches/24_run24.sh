#!/bin/bash
# usage: run24.sh RECORD_DIR TAG [audit_run24 args...]   (needs ~/audit_ext/work24/prep with the joblib files)
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/24_Erikdai__Fault-Diagnosis-of-Aero-engine-Inter-shaft-Bearings
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MPLBACKEND=Agg
W=~/audit_ext/work24/$TAG; mkdir -p ~/audit_ext/logs24 $W; cd $W
for f in ~/audit_ext/work24/prep/*_1024_10c; do ln -sfn $f .; done
LOG=~/audit_ext/logs24/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
~/audit_ext/envs/24/bin/python $SRC/audit_run24.py "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep '^{' $LOG | cut -c1-400
grep -E 'Error|Traceback' -A3 $LOG | tail -10
