#!/usr/bin/env bash
# P4：TFN main.py（Backbone_CNN / Random_CNN / TFN_STTF，mid_channel 32，50 epoch），原样运行。
# 作者固定种子 999；5 次重复用于测 GPU 非确定性。一次一个进程。
set -uo pipefail
: "${AUDIT_PY:?}" "${TFN_DIR:?}" "${CWRU48_DIR:?}" "${OUT_DIR:?}"
MODELS=${MODELS:-"Backbone_CNN Random_CNN TFN_STTF"}
REPS=${REPS:-"0 1 2 3 4"}
for m in $MODELS; do
  for r in $REPS; do
    d="$OUT_DIR/p4/${m}/rep${r}"
    if ls "$d"/*/Records.mat >/dev/null 2>&1; then echo "skip $m rep$r"; continue; fi
    rm -rf "$d"; mkdir -p "$d"
    echo "=== $m rep$r $(date -Is) ==="
    "$AUDIT_PY" "$TFN_DIR/main.py" --data_dir "$CWRU48_DIR" --model_name "$m" --checkpoint_dir "$d" > "$d/stdout.txt" 2>&1 \
      || echo "FAILED $m rep$r (see $d/stdout.txt)"
  done
done
echo "=== P4 DONE $(date -Is) ==="
