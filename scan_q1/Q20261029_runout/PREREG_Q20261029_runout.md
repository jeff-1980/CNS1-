# 预注册：Q-20261029-1 杀死测试（焊接 S–N 越出点处理偏差）

冻结时间：2026-10-29（UTC），在真实数据上运行任何 K1–K4 代码之前。jeff 2026-10-29 批准该候选（"批准 Q-20261029-1"）。
候选与判据来源：/mnt/project-files/scan_daily_q1/reports/2026-10-29.md；入池前基线 baseline/bl_20261029_runout*.{py,md}（已在真实数据上跑过，属已知信息：DROP 相对 Tobit 在 2×10⁶ 处中位 −6.0%，10⁷ 处 −10.2%）。

## 数据
Deng et al. 2025 Sci Data 焊接接头疲劳库（figshare 10.6084/m9.figshare.29254265，CC BY 4.0），jeff 电脑 WSL ~/data_weld 的 S-N.json（dataset_id, life_n, stress_range, runout）与 parameter.json（dataset_id, doi, 其余参数）。只读。

## 合格系列（与基线相同）
按 dataset_id 分系列；丢弃 life_n ≤ 0、stress_range ≤ 0、runout ∉ {0,1} 的点。合格 = 失效点 ≥ 3、越出点 ≥ 1、失效点应力不全相同，且 Basquin 删失 MLE（Tobit）收敛。

## 拟合（log10 N = a + b·log10 S，正态残差）
- DROP：只用失效点 OLS。FAIL：越出点当失效 OLS。TOBIT：越出点在停机次数处右删失的 MLE（Nelder–Mead，从 DROP 起步）。
- FL（K1 主参照，带拐点）：固定疲劳极限删失 MLE，log10 N = a + b·log10(S − S0)；S0 = f × 最小失效应力，f ∈ {0, 0.05, …, 0.95} 剖面取最大似然；S ≤ S0 的越出点似然记 1。f 取到 0.95 的系列记为"边界解"。
- RFLM（K1 次参照）：Pascual & Meeker (1999) 随机疲劳极限模型，ln V ~ N(μγ, σγ)，48 点 Gauss–Legendre 积分，L-BFGS-B 有界（b1 ∈ [−30, −0.3]，σ ∈ [0.01, 5]，μγ ∈ [ln 0.05·Smin, ln 1.5·Smax]，σγ ∈ [0.01, 1]），3 个起点；强度取中位失效概率 0.5 处。
- 强度读在 N = 2×10⁶（判定）与 10⁷（报告）。

## 判据（任一不过 → KILL）
- **K1 稳健性**：DROP 相对 FL 在 2×10⁶ 处的 |S_DROP/S_FL − 1| 跨系列中位 **< 3% → 杀死**。RFLM、去掉 FL 边界解后的结果只报告，不改判定。
- **K2 安慰剂**：限 nf − nr ≥ 3 的系列。每个系列随机删去 nr 个失效点（越出点保留），重拟 Tobit，求相对全量 Tobit 在 2×10⁶ 处的偏差；重复 50 次（种子 20261029），每次得一个跨系列的 |偏差| 中位数。DROP（相对 Tobit，同一子集）的 |偏差| 中位数 **≤ 50 个安慰剂中位数的 95% 分位 → 杀死**。另报 10⁷ 处同样统计和带符号中位。
- **K3 设计后果**：IIW 特征强度（2×10⁶ 处，95% 存活、75% 置信单侧容限，k = t′₀.₇₅(n−1, 1.645√n)/√n；DROP 用 n = nf 和 OLS 残差 sd，TOBIT 用 n = nf + nr 和 MLE σ；自由斜率）。FAT 等级 = 不超过特征强度的最大 IIW 等级（…, 36, 40, 45, 50, 56, 63, 71, 80, 90, 100, 112, 125, …，两端按约 12% 几何延伸）。DROP 与 TOBIT 等级不同的系列 **< 15% → 杀死**。另报：均值曲线等级变化、固定斜率 m = 3 的等级变化、降级/升级比例、≥2 档比例。
- **K4（探索，不判死）**：按 DOI 分组 5 折交叉验证，XGBoost（log10 S 单调递减约束；depth 6，eta 0.05，500 轮，subsample/colsample 0.8）三种：DROP 回归（只用失效点）、FAIL 回归、AFT（survival:aft，normal，scale 1.0，越出点右删失）。特征 = log10 S + parameter.json 的数值/类别字段（自动识别；名字像强度、FAT、斜率、寿命、年份、DOI 等的字段一律剔除以防泄漏，脚本输出所用字段）。对留出 DOI 里的合格系列，在 [0.3·Smin, 3·Smax] 60 点网格上读模型 2×10⁶ 强度（无交点则夹到网格端点并计数），与该系列 TOBIT、FL 参照比较。文献编码（50 篇疲劳 ML 论文）需人工第二编码者，本测试不做。

## 判定
K1、K2、K3 都过 → PASS（建议进入正式研究，下一步由 jeff 决定）；否则 KILL。K4 只描述。

## 预期偏离处理
- 只改不影响判定的 bug（崩溃、读文件），在结果里逐条记录；任何改动判定统计量定义的修改都算偏离，判定仍按本文件原定义计算并同时报告。
- 运行环境：jeff 电脑 WSL，~/bbvenv/bin/python，CPU；长步骤后台运行并记 PID 与日志。

## 代码
- runout_killtest.py（K1–K3）、runout_k4_xgb.py（K4）、run.sh（启动器）；SHA-256 见同目录 SHA256SUMS（含本文件），冻结 SHA 另记在 PREREG_Q20261029_runout.sha256.txt。
- 合成数据自检（冻结前）：无疲劳极限的 Basquin 数据上，TOBIT、FL、RFLM 在 2×10⁶ 处相对真值中位偏差 ≤ 3.4%；有随机疲劳极限的数据上 ≤ 1%。
