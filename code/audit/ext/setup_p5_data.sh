#!/usr/bin/env bash
# P5 数据：按 pdm-bench 期望的 Kaggle 目录（Datasets/CWRU/{Experiment}/{rpm}/{stem}.mat）建软链接，指向本机已有的 CWRU 12k DE 原文件。
# 只覆盖本机已有的 10 类文件（3 个尺寸 × 内圈/滚动体/外圈@6 点 + 正常），不含 0.028 与外圈 @3/@12（偏离记录）。
# 99.mat 原样链接：它同时含 X098 与 X099，pdm-bench 的加载器会把两段各作为一条 1750 rpm 的正常记录（只影响训练集，审计里记录）。
set -euo pipefail
: "${CWRU12_SRC:?directory containing 97.mat … 237.mat}" "${PDM_WORKSPACE:?}"
root="$PDM_WORKSPACE/datasets/javadseraj-cwru-bearing-fault-data-set/Datasets/CWRU"
rpms=(1797 1772 1750 1730)
declare -A F=(
  [NormalBaseline/Normal]="97 98 99 100"
  [12DriveEndFault/0.007-InnerRace]="105 106 107 108"
  [12DriveEndFault/0.007-Ball]="118 119 120 121"
  [12DriveEndFault/0.007-OuterRace6]="130 131 132 133"
  [12DriveEndFault/0.014-InnerRace]="169 170 171 172"
  [12DriveEndFault/0.014-Ball]="185 186 187 188"
  [12DriveEndFault/0.014-OuterRace6]="197 198 199 200"
  [12DriveEndFault/0.021-InnerRace]="209 210 211 212"
  [12DriveEndFault/0.021-Ball]="222 223 224 225"
  [12DriveEndFault/0.021-OuterRace6]="234 235 236 237"
)
n=0
for key in "${!F[@]}"; do
  exp=${key%%/*}; stem=${key#*/}; i=0
  for num in ${F[$key]}; do
    src=$(find -L "$CWRU12_SRC" -name "$num.mat" -print -quit)
    [ -n "$src" ] || { echo "missing $num.mat under $CWRU12_SRC" >&2; exit 1; }
    d="$root/$exp/${rpms[$i]}"; mkdir -p "$d"
    ln -sfn "$(readlink -f "$src")" "$d/$stem.mat"; n=$((n+1)); i=$((i+1))
  done
done
echo "linked $n files under $root"
