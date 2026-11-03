#!/bin/bash
# usage: run58.sh RECORD_DIR TAG [audit_run58 args...]
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/58_qinenergy__syn2real
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export LD_LIBRARY_PATH=$(cat ~/audit_ext/envs/58/audit_ld_library_path.txt 2>/dev/null):$LD_LIBRARY_PATH
mkdir -p ~/audit_ext/logs58
LOG=~/audit_ext/logs58/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
cd $SRC/data/preprocessed && ~/audit_ext/envs/58/bin/python $SRC/audit_run58.py "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep '^{' $LOG | cut -c1-300
grep -E 'Error|Traceback' -A3 $LOG | tail -10
