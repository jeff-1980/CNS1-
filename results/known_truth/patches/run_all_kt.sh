#!/bin/bash
# Known-truth test (results/known_truth/PLAN.md, 678548b): objects 58, 54, 76, 37, 62, 53 one after another,
# training settings and seeds as in the main analysis. Records -> ~/audit_ext/known_truth/records/<obj>/...
REC=~/audit_ext/known_truth/records; mkdir -p $REC
LOG=~/audit_ext/known_truth/run_all.log
cd ~/audit_ext
for o in 58 54 76 37 62 53; do
  echo "$(date -u +%FT%TZ) start $o" >> $LOG
  ./kt$o.sh $REC main >> $LOG 2>&1
  echo "$(date -u +%FT%TZ) end $o" >> $LOG
done
echo "$(date -u +%FT%TZ) all done" >> $LOG
