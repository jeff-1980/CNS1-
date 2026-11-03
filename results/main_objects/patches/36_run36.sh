#!/bin/bash
# usage: run36.sh RECORD_DIR TAG [audit_run36 args...]
REC=$1; TAG=$2; shift 2
SRC=~/audit_ext/main_objects/36_Luojunqing666__Eviformer-An-uncertainty-fault-diagnosis-framework-guided-by-evidential-deep-learning
export PYTHONPATH=~/audit_cns/code AUDIT_RECORD_DIR=$REC
W=~/audit_ext/work36/$TAG; D=~/audit_ext/work36/cwru_root; mkdir -p ~/audit_ext/logs36 $W $D
ln -sfn /home/jeffwork/论文CNS/data/cwru_12k_de "$D/12k Drive End Bearing Fault Data"
cd $W
LOG=~/audit_ext/logs36/$TAG.log
( while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem:/{print "RAM",$3}'; done > $LOG.mon ) &
MON=$!
t0=$(date +%s)
~/audit_ext/envs/36/bin/python $SRC/audit_run36.py --data_dir $D --save_dir $W/ckpt "$@" > $LOG 2>&1
rc=$?; kill $MON
echo "exit=$rc seconds=$(( $(date +%s) - t0 )) gpu_max=$(grep -v RAM $LOG.mon | sort -n | tail -1) ram_max=$(grep RAM $LOG.mon | awk '{print $2}' | sort -n | tail -1)"
grep -o '{"group[^}]*}[^}]*}' $LOG | cut -c1-300
grep -E 'Error|Traceback' -A3 $LOG | tail -10
