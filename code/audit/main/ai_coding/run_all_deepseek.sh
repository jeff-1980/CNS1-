#!/bin/bash
# RULES_v1 §1: deepseek-v4-pro, runs 1-3 over the 79 papers with full text (003 has none).
set -u
export DEEPSEEK_API_KEY=$(powershell.exe -NoProfile -Command "[Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY','User')" | tr -d '\r\n')
cd ~/audit_cns
P=$(seq 1 80 | grep -vx 3)
for run in 1 2 3; do
  python3 -u code/audit/main/ai_coding/run_deepseek.py deepseek-v4-pro $run 8 $P
done
