# CNS1# 稿件“结论边界”段落草稿（2026-10-15，待 Jeff 审）

依据：C6-P（`VERDICT_C6P.md`）、T1（`POSTHOC_T1_margin.md`，探索性）、T2（`VERDICT_T2.md`，INDETERMINATE）、T3（`VERDICT_T3.md`，DATA，2/3 架构）。分支 `claude/project-thread-lq0bhi`，提交 ab67b8d。只用已有结果，不新增主张。

## English draft (for the Discussion / Limitations section)

**Scope of the coverage–attractor result.** The finding that training coverage determines the collapse class under strong noise was established on CWRU, where models trained from different seeds map pure noise to the same class (pairwise seed agreement 0.97–1.00 across three architectures; replicated in three independent sets of seeds). We tested whether this determinacy, which the coverage effect presupposes, transfers to other domains in a series of preregistered experiments. On ECG (PTB-XL) and natural images (CIFAR-100) the noise attractor varied across seeds, so the coverage effect could neither be confirmed nor refuted. When learning strength was matched, a non-bearing time-series dataset (StarLightCurves) trained to the same or higher clean accuracy (0.96–0.97) still showed low seed agreement (0.36–0.64); the domain difference exceeded the preregistered equivalence margin in two of three architectures. Weakening CWRU models with label noise removed their determinacy (agreement 0.28–0.49), so both a well-learned model and CWRU-type data were required. On CWRU the attractor followed the high-frequency part of the noise (high-pass noise reproduced the white-noise attractor in 10/10 models, low-pass in 0/10), which is consistent with a discriminative band that broadband noise excites. We therefore restrict the coverage–attractor claim to test-rig vibration data of this kind; whether it holds for other bearing or machinery datasets, and whether it is specific to bearings or to test-rig data in general, remains untested. The CWRU cells of these experiments were trained with the pre-correction loader (Section 3.8), in which one Normal recording in the training pool duplicated a test recording; re-scoring the existing models on unseen Normal windows left the clean accuracy of all well-trained CWRU models unchanged (Normal recall 1.000), so the strength matching holds, but seed agreement was not re-measured on the corrected training pool.

## 中文对照（要点）

- 结论只在 CWRU 上成立；它依赖“不同种子把纯噪声映射到同一类”（一致性 0.97–1.00，三组独立种子复现）
- 心电、自然图像上吸引子随种子变化，覆盖效应既不能证实也不能否定（C6-P）
- 学习强度对齐后，非轴承时间序列（StarLightCurves，干净准确率 0.96–0.97）一致性仍只有 0.36–0.64，2/3 架构超过预设等效界（T3）
- CWRU 用标签噪声弱化后确定性消失（0.28–0.49），所以“学好”和“CWRU 类数据”缺一不可（T2、T3）
- CWRU 吸引子跟随噪声高频部分（高通 10/10、低通 0/10）
- CWRU 格用的是修正前的加载器（训练池有一个 Normal 录音与测试录音相同）；用没见过的 Normal 窗口重算，学好的 CWRU 模型干净准确率不变（Normal 召回 1.000），强度对齐成立；种子一致性未在修正后的训练池上重测（提交 7e3261b）
- 结论限于这类试验台振动数据；对其他轴承 / 机械数据集、以及“轴承特有还是试验台数据特有”未检验

## 写入前需核对

- 投稿去向未定（MSSP 已桌拒），段落按通用 Discussion 写，未按期刊格式
- “三组独立种子复现”指 T1（E5 种子）、T2（0–9）、T3（10–19）
- T1 为探索性分析，若正文只引预注册结果，可删去 T1 相关表述，用 T2 / T3 的高通 / 低通数字
