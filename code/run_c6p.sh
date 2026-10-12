#!/bin/bash
# C6-P 全量：D1 → D2（每领域 45 个实例），日志不含主导类别
P=~/.claude-science/conda/envs/python/bin/python3; L=~/论文CNS/logs
cd ~/论文CNS/release/git_stage/code
echo "### D1 start $(date -Is)" >> $L/c6p_d1.log
$P c6p_train.py --domain d1 >> $L/c6p_d1.log 2>&1 || { echo "### D1 FAILED $(date -Is)" >> $L/c6p_d1.log; exit 1; }
echo "### D2 start $(date -Is)" >> $L/c6p_d2.log
$P c6p_train.py --domain d2 >> $L/c6p_d2.log 2>&1 || { echo "### D2 FAILED $(date -Is)" >> $L/c6p_d2.log; exit 1; }
echo "### C6P CHAIN DONE $(date -Is)" >> $L/c6p_d2.log
