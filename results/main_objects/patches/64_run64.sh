#!/bin/bash
# usage: run64.sh RECORD_DIR TAG [audit_run64 args...]
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/64_yazyeah__MHFL-MCA-Codes-Datasets-and-Results
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC MPLBACKEND=Agg PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export LD_LIBRARY_PATH=$HOME/audit_ext/envs/64/lib:$LD_LIBRARY_PATH TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p ~/audit_ext/logs64 ~/audit_ext/work64
LOG=~/audit_ext/logs64/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
~/audit_ext/envs/64/bin/python $SRC/audit_run64.py "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep '^{' $LOG | cut -c1-300
grep -E 'Error|Traceback' -A3 $LOG | tail -10
