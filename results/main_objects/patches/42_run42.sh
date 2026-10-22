#!/bin/bash
# usage: run42.sh RECORD_DIR TAG [audit_run42 args...]
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/42_caie8201__Multi-Scale-CNN-Transformer-Hybrid-Network
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC MPLBACKEND=Agg
mkdir -p ~/audit_ext/logs42 ~/audit_ext/work42; cd ~/audit_ext/work42
ln -sf ~/audit_ext/data/pu42/PU3200data14.mat PU3200data14.mat
LOG=~/audit_ext/logs42/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
~/audit_ext/envs/42/bin/python $SRC/audit_run42.py "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep -o '{"group[^}]*}[^}]*}' $LOG | cut -c1-300
grep -E 'Error|Traceback' -A3 $LOG | tail -10
