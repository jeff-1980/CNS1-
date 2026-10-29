#!/bin/bash
# usage: run54.sh RECORD_DIR TAG [audit_run54 args...]
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/54_RanGuangsheng__TGST
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC MPLBACKEND=Agg PYTHONUNBUFFERED=1
export HF_HOME=/home/jeffwork/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p ~/audit_ext/logs54
LOG=~/audit_ext/logs54/$TAG.log
W=/home/jeffwork/audit_ext/work54home
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
cd $W && HOME=$W ~/audit_ext/envs/54/bin/python $SRC/audit_run54.py "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep '^{' $LOG | cut -c1-300
grep -E 'Error|Traceback' -A3 $LOG | tail -10
