#!/usr/bin/env bash
# P3：UDTL train_base.py（Basis 与 AdaBN），原样运行。作者按 target_val（目标域带标签数据）选 checkpoint。
# 迁移任务：load 0→1 与 0→3；每配置 5 次重复（作者代码无种子）。一次一个进程。
set -uo pipefail
: "${AUDIT_PY:?}" "${UDTL_DIR:?}" "${CWRU_UDTL_DIR:?}" "${OUT_DIR:?}"
TASKS=${TASKS:-"[0],[1] [0],[3]"}
REPS=${REPS:-"0 1 2 3 4"}
cd "$UDTL_DIR"
for t in $TASKS; do
  tn=$(echo "$t" | tr -d '[],')
  for method in basis adabn; do
    for r in $REPS; do
      d="$OUT_DIR/p3/t${tn}_${method}/rep${r}"
      if ls "$d"/*/training.log >/dev/null 2>&1 && grep -q "Epoch: 299 target_val-Loss" "$d"/*/training.log; then echo "skip $tn $method rep$r"; continue; fi
      rm -rf "$d"; mkdir -p "$d"
      echo "=== task $t $method rep$r $(date -Is) ==="
      if [ "$method" = basis ]; then extra=(--adabn ""); else extra=(); fi
      "$AUDIT_PY" train_base.py --model_name CNN_1d --data_name CWRU --data_dir "$CWRU_UDTL_DIR" \
        --transfer_task "$t" --normlizetype mean-std --checkpoint_dir "$d" "${extra[@]}" > "$d/stdout.txt" 2>&1 \
        || echo "FAILED task $t $method rep$r (see $d/stdout.txt)"
    done
  done
done
echo "=== P3 DONE $(date -Is) ==="
