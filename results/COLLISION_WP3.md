# WP3-lite 碰撞检查（规划 v12 §3 路线 A 第二优先），2026-09-25

**待查主张（A3）**：checkpoint 选模协议 / 训练时点改变强噪声下的崩溃——CWRU 验证集第 1–2 轮即饱和，"严格提升即保存"的规则因此选中 epoch 1–2 的模型；完整训练后，6 个会崩溃的架构中有 5 个崩溃 cell 增多（cnn1d 相反）。

## 检索

OpenAlex，仅标题检索 12 组（2015 起，按被引排序）：robust overfitting；early stopping × robustness / noise；training duration × OOD；checkpoint selection × robustness；model selection × distribution shift；lost domain generalization；accuracy on the line；corruption robustness × training dynamics；fault diagnosis × early stopping / checkpoint / validation strategy。原始记录 `tmp/wp3/collision_wp3.json`。

## 结果：**部分命中**——一般现象已知，领域内的具体实例与机制未见

| 已有工作 | 与 A3 的关系 |
|---|---|
| Rice, Wong, Kolter. *Overfitting in adversarially robust deep learning*, ICML 2020（arXiv 2002.11569） | **一般现象等价**：训练越久，鲁棒性能越差，早停的 checkpoint 更鲁棒（对抗扰动场景）。A3 的"完整训练后崩溃增多"是其在 AWGN、振动诊断上的对应 |
| Gulrajani, Lopez-Paz. *In Search of Lost Domain Generalization*, ICLR 2021（arXiv 2007.01434） | **一般现象等价**：分布偏移下，模型选择准则本身显著改变结论，必须作为协议的一部分报告 |
| Andreassen et al. *The Evolution of OOD Robustness Throughout Fine-Tuning*（arXiv 2106.15831）；Miller et al. *Accuracy on the Line*（arXiv 2107.04649） | 训练过程中 OOD 鲁棒性的演化 / 与 ID 准确率的关系；背景 |
| Hendrycks, Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions*, ICLR 2019 | 腐蚀鲁棒性（含高斯噪声）的基准框架；背景 |
| 故障诊断领域 | 标题检索 **0–2 条**，均不相关 |

**未见的部分**：(1) 振动诊断中，验证集早饱和 + 严格提升规则 → 系统性选中 epoch 1–2 模型，并由此改变已发表的崩溃 cell 统计（E4 的 150/150 逐位复现）；(2) 架构间方向相反（cnn1d 与其余 5 个）；(3) 选模协议对**崩溃靶标**（而非准确率）的影响。

## 判定（按规划 v12 与三门 G2 的规则）

- A3 的一般命题"训练时点 / 选模协议改变噪声鲁棒性"**已被 Rice 2020、Gulrajani 2021 覆盖** → **降为对照，不作头牌**；稿件中作为"已知现象在振动诊断中的实例与协议警示"引用上述工作
- 按 v12 规则"只有没有等价工作，才补最小训练轨迹" → **不补训练轨迹**（WP3-lite 到此结束）
- 保留在稿件中的是领域内的具体发现（验证集饱和机制、架构方向相反、E4 复现），均已有数据，无需新实验
- 局限：只用了 OpenAlex 标题检索；未检 arXiv 全文 / CNKI；Rice 2020 与 Gulrajani 2021 的结论按摘要引用，写稿时需读全文核对表述

## 2026-09-25 更正：读全文后的修订（原判定依据摘要）

- **Rice, Wong, Kolter 2020（arXiv 2002.11569）**：研究对象是**对抗训练**下对**最坏情况扰动**的鲁棒性；早期 checkpoint 更鲁棒（"robust overfitting"）。原文明确指出，在**标准训练**中测试损失通常随训练逐步改善，最佳 checkpoint 往往就在训练末尾。原文未涉及高斯 / 随机噪声（全文 "Gaussian" 0 次）。→ 它**不覆盖** A3：A3 是标准训练、随机 AWGN、完整训练后崩溃增多（5/7 架构）——是与 robust overfitting **类比**的现象，不是其实例
- **Gulrajani & Lopez-Paz 2021（arXiv 2007.01434）**：model selection（超参数、**训练 checkpoint**、架构变体）是学习问题的一部分；没有给定选模方法的领域泛化算法"应视为不完整"。→ 支持"选模协议须作为协议的一部分报告"这一**警示**，不涉及噪声崩溃
- **修订后的判定**：一般命题"选模协议改变分布偏移下的结论"已被 Gulrajani 2021 覆盖；"标准训练 + 随机噪声下完整训练增加崩溃"**未见等价工作**（本深度：OpenAlex 标题 12 组 + arXiv 59 篇筛查）。按 v12 规则本应补最小训练轨迹——**已有**：E1（cnn1d 7 个轨迹点）与 E4 / WP1（7 个架构 × 8 个 epoch），无需新实验
- 对稿件：Jeff 已裁定 checkpoint 不进头牌（2026-09-25），此更正**不改变**该裁定（效应只在 cnn1d 上反向、其余架构增量小，且依赖验证集早饱和）；Results 4.3 的引用措辞已按全文修正，不再写"一般现象已知"
