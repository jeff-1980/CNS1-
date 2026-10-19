#!/bin/bash
# T3：§4 校准（只看干净准确率，含 §4.1 新领域可行性）→ 主实验 150 个实例 → 判定。日志不含主导类别。
P=~/.claude-science/conda/envs/python/bin/python3; L=~/论文CNS/logs
cd ~/论文CNS/release/git_stage/code
echo "### CALIB start $(date -Is)" >> $L/t3_calib.log
$P t3.py --stage calib >> $L/t3_calib.log 2>&1 || { echo "### CALIB FAILED $(date -Is)" >> $L/t3_calib.log; exit 1; }
echo "### CALIB done $(date -Is)" >> $L/t3_calib.log
$P -c "import json;import sys;sys.exit(0 if json.load(open('../results/t3_calibration.json'))['selected'] else 3)" || { echo "### NO FEASIBLE N, main skipped $(date -Is)" >> $L/t3_calib.log; $P analyze_t3.py >> $L/t3_analysis.log 2>&1; exit 0; }
echo "### MAIN start $(date -Is)" >> $L/t3_main.log
$P t3.py --stage main >> $L/t3_main.log 2>&1 || { echo "### MAIN FAILED $(date -Is)" >> $L/t3_main.log; exit 1; }
echo "### MAIN done $(date -Is)" >> $L/t3_main.log
$P analyze_t3.py >> $L/t3_analysis.log 2>&1
echo "### T3 CHAIN DONE $(date -Is)" >> $L/t3_main.log
