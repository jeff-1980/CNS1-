# 第 3 层"化验" v0（2026-11-03）

`assay.py` 两件事：

1. **候选方向台账回放**（`ledger.jsonl`，38 条：3 个存活 + 35 个杀死/丢弃，字段与论文卡片同一七元组，外加死因码）。
   只用入池前就知道的字段打分（硬件、数据可得、杠杆、机械相关度、撞车、拥挤度），结果在 `replay_ledger.csv`。
   存活方向排第 1、2、4 名（AeroUp、审计、Backblaze），存活 vs 其余 AUC 0.986。
   **注意**：撞车/拥挤两个字段是我按记忆填的，权重也是从同一批教训里定的，所以这个数字偏乐观，只能说明规则没有自相矛盾，不是独立验证。
   更有用的是反例：焊接小样本效应（第 3）和越出点偏差（第 5）在入池前看起来都很好，死在基线和真值模拟上。入池前的筛子抓不住 LOST_BASELINE / MISSPEC 两类死因，杀死测试不能省。

2. **雷达主题对化验**（`assayed_top50.csv`）：涌现概率 × 伙伴主题领域过滤（商科、经济、心理、社科等 = 0；医学 0.3）× 可行性（该主题 NC/SA 卡片里纯计算占比）× 天花板（该主题卡片里出现过子刊杠杆的占比）× 真实世界数据占比 × 拥挤度（伙伴主题年发文量）× 机械核心占比。
   主题级先验来自 2322 张卡片（按每篇前 3 个主题归并，加平滑）。

## 结论（v0）
- 化验后前 20 仍是"某个工程对象 + 通用 AI 方法"（电力巡检机器人、火灾安全、自主系统安全、电网可靠性与维护、制冷空调 + XAI、交通安全 + NLP、岩土 + 模型降阶）。它们是方向，不是题目。
- 雷达在主题粒度上只能指出"哪些工程运行系统正被 AI 方法接触"；金子的位置还要靠数据地图（这些系统有没有公开的多年事件/运行记录）才能定。下一步就是数据地图 v1 按这些对象去找数据。

## 规则对齐（2026-11-03，按 jeff 批准的流程补充提案 §A/§B）
- **评估类型声明**：`assay_summary.json` 现在标明：台账回放 = `relative_only`（字段和权重出自同一批教训，不是独立验证）；雷达 K1/K3 = `real_gt`（OpenAlex 真实计数，标签是 T+3 真实涌现）；化验后的主题对 = 无真值、前瞻性，只有杀死测试能给它们打分。
- **失败记录原因码**：`ledger.jsonl` 每条加 `reason_code`（H1/H3/H4/baseline/K-sim/crowded/data），与两个扫描池的 `failed_ideas.jsonl` 同一套码；`assay.py` 会自动读入 `scan_daily/failed_ideas.jsonl` 里带完整七元组字段的条目，一起回放。
- **三档判定与修补预算**：雷达产出的候选进入扫描池后照池子规则：先小子集、通过/杀死/可修三档、最多 2 次修补、超 4 GPU 小时需 jeff 决定。本层不跑 GPU，不另设规则。
- LLM 审核步骤（卡片拆解、将来的命题生成）用 Claude Sonnet 子代理，不用 DeepSeek 或 Haiku。

## 扩展版（13 个子领域，2026-11-03）
`python3 assay/assay.py ext` 读 `results/ext/forecast_T2025_top2000.csv`，输出 `assayed_top50_ext.csv`、`assay_summary_ext.json`。
化验后前 20 里"电动汽车与基础设施"占了近一半（配对抗鲁棒性、图神经网络、能效、气候、工业效率等），其余是制冷空调 + 可解释 AI、机器人操作 + 火灾安全、自动驾驶安全 + 基础设施韧性、电力线巡检机器人、地热 + 可解释 AI、故障诊断 + 电动汽车。
注意：能源/交通起点主题在卡片先验里"机械核心占比"取默认值 0.31，会比机械起点主题略吃亏；这是已知偏差，不是结论。

## Radar target decision (2026-11-04, jeff)
Primary radar = v1 `ns` target (results/v1/ns: pair first co-appears in a Nature/Science-family paper), run with `python3 assay/assay.py v1`
(mech-core source topics, generic-AI partners removed). The old co-occurrence target (results/ext, `assay.py ext`) is kept only as a
control; both feed the scan with seed_source labels radar_v1 / radar_v0 so hit rates can be compared. Sci Data feature not adopted (no gain).
