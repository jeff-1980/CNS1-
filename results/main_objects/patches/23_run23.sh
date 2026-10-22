#!/bin/bash
# usage: run23.sh RECORD_DIR TAG [audit_run23 args...]   (needs ~/audit_ext/work23/prep with the dataset.py outputs)
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/23_DABINB__CSST-Net/CSST_Net
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC MPLBACKEND=Agg
W=~/audit_ext/work23/$TAG; mkdir -p ~/audit_ext/logs23 $W/predata $W/Loss_Acc $W/results $W/weights; cd $W
ln -sfn ~/audit_ext/work23/prep/'predata\data_6.npy' predata/data_6.npy
ln -sfn ~/audit_ext/work23/prep/'predata\label_6.npy' predata/label_6.npy
LOG=~/audit_ext/logs23/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
~/audit_ext/envs/23/bin/python $SRC/audit_run23.py "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep -o '{"repeat[^}]*}[^}]*}' $LOG | cut -c1-300
grep -E 'Error|Traceback' -A3 $LOG | tail -10
