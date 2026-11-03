#!/usr/bin/env bash
# P5：pdm-bench DL 流水线，cwru_cross_load（训练 1797/1772/1750 rpm，测试 1730 rpm），作者 "long" 档
# （50 epoch、batch 64、重叠 0.75/0.5，与作者 follow-up 的 cwru_cross_load 设置相同），4 个模型 × 5 个种子。
# 原样运行，只经 audit.ext.pdm_hook 加逐 epoch 的 test 记录。一次一个进程。在 code/ 下运行。
set -uo pipefail
: "${AUDIT_PY:?}" "${PDM_DIR:?}" "${PDM_WORKSPACE:?}" "${OUT_DIR:?}"
MODELS=${MODELS:-"cnn1d fft stft mlp"}
SEEDS=${SEEDS:-"41 42 43 44 45"}
EPOCHS=${EPOCHS:-50}
export WORKSPACE="$PDM_WORKSPACE" PYTHONPATH="$PDM_DIR/src:${PYTHONPATH:-}"
for m in $MODELS; do
  for s in $SEEDS; do
    d="$OUT_DIR/p5/${m}/seed${s}"
    if [ -f "$d/epochs.json" ]; then echo "skip $m seed$s"; continue; fi
    rm -rf "$d"; mkdir -p "$d"
    echo "=== $m seed$s $(date -Is) ==="
    PDM_EPOCH_LOG="$d/epochs.json" "$AUDIT_PY" -m audit.ext.pdm_hook --config-path "$PDM_DIR/config" "task=cwru_cross_load/$m" \
      "run.output_dir=$d/artifacts" "+train.random_state=$s" \
      train.epochs=$EPOCHS train.batch_size=64 train.num_workers=0 windowing.train_overlap=0.75 windowing.test_overlap=0.5 \
      > "$d/stdout.txt" 2>&1 || echo "FAILED $m seed$s (see $d/stdout.txt)"
  done
done
echo "=== P5 DONE $(date -Is) ==="
