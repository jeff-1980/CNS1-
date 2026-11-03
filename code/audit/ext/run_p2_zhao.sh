#!/usr/bin/env bash
# P2：Zhao 2020 DL-based-Intelligent-Diagnosis-Benchmark，原样运行作者 train.py。
# 预注册 OSF yszc4 §3.2：每配置 5 次重复；作者代码无种子，5 次即 5 个独立重复。一次只跑一个进程。
# 用法：bash run_p2_zhao.sh  （需 AUDIT_PY、ZHAO_DIR、CWRU_ZHAO_DIR、OUT_DIR 环境变量）
set -uo pipefail
: "${AUDIT_PY:?}" "${ZHAO_DIR:?}" "${CWRU_ZHAO_DIR:?}" "${OUT_DIR:?}"
MODELS=${MODELS:-"cnn_1d lenet_1d alexnet_1d resnet18_1d BiLSTM1d MLP"}
REPS=${REPS:-"0 1 2 3 4"}
cd "$ZHAO_DIR"
for m in $MODELS; do
  for r in $REPS; do
    d="$OUT_DIR/p2/${m}/rep${r}"
    if ls "$d"/*/training.log >/dev/null 2>&1 && grep -q "Epoch: 99 val-Loss" "$d"/*/training.log; then echo "skip $m rep$r"; continue; fi
    rm -rf "$d"; mkdir -p "$d"
    echo "=== $m rep$r $(date -Is) ==="
    "$AUDIT_PY" train.py --model_name "$m" --data_name CWRU --data_dir "$CWRU_ZHAO_DIR" \
      --normlizetype mean-std --processing_type R_A --checkpoint_dir "$d" > "$d/stdout.txt" 2>&1 \
      || echo "FAILED $m rep$r (see $d/stdout.txt)"
  done
done
echo "=== P2 DONE $(date -Is) ==="
