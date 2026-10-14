#!/bin/bash
# T2：§4 校准（只看干净准确率）→ 主实验 180 个实例。日志不含主导类别。
P=~/.claude-science/conda/envs/python/bin/python3; L=~/论文CNS/logs
cd ~/论文CNS/release/git_stage/code
echo "### CALIB start $(date -Is)" >> $L/t2_calib.log
$P t2.py --stage calib >> $L/t2_calib.log 2>&1 || { echo "### CALIB FAILED $(date -Is)" >> $L/t2_calib.log; exit 1; }
echo "### CALIB done $(date -Is)" >> $L/t2_calib.log
echo "### MAIN start $(date -Is)" >> $L/t2_main.log
$P t2.py --stage main >> $L/t2_main.log 2>&1 || { echo "### MAIN FAILED $(date -Is)" >> $L/t2_main.log; exit 1; }
echo "### T2 CHAIN DONE $(date -Is)" >> $L/t2_main.log
