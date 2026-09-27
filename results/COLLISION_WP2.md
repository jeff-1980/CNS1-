# WP2 碰撞检查（规划 L9 / 三门 G2），2026-09-25

**待查主张**：振动故障分类器在强噪声下崩向哪一类，由故障训练数据的**损伤构成**（损伤尺寸 / 程度的覆盖）决定。

## 检索

- 领域规模：OpenAlex 全文检索 "bearing fault diagnosis deep learning noise"（2015 起）约 1.69 万篇
- 第一轮：8 组相关度检索（严重程度 × 训练构成 × 噪声误分类；强噪声下预测塌缩到单一类；训练集构成与噪声偏置；误判为内圈；图像领域的噪声 / 腐蚀下的类偏置；早停与噪声鲁棒性），每组看前 8 条
- 第二轮：10 组**仅标题**检索（fault severity noise；damage size；unseen / cross severity；collapse；prediction bias；misclassification noise；class bias gaussian noise 等）
- 原始记录：`tmp/wp2/collision_openalex.json`、`tmp/wp2/collision_openalex_title.json`

## 结果

- **未发现等价主张。** 相关度检索前列均为综述与通用方法论文；标题检索命中 0–3 条
- 最接近的两篇均为"**跨损伤程度泛化**"（训练集中没有的严重程度），研究的是准确率 / 开放集识别，**不涉及强噪声下的崩溃靶标**：
  - Generalized Cross-Severity Fault Diagnosis of Bearings via a Hierarchical Cross-Category Inference Framework，IEEE TII 2021，10.1109/tii.2021.3116145
  - Few-Shot Bearing Fault Diagnosis under Joint Fault-Severity and Load Shift: A Leak-Free Cross-Domain Benchmark，CMES 2026，10.32604/cmes.2026.084403
- 通用 ML 方向的竞争解释仍是 2026-08 碰撞检查中列出的 Neural Collapse（PNAS 2020）/ Minority Collapse（PNAS 2021）；"塌缩方向可由训练分布的某个可计算量预测"（C5）当时判为空位，本轮未见反例

## 判定与局限

- **G2 维持"暂通过"**：本深度下没有命中，WP2 可以进入预注册
- 局限：只用了 OpenAlex（相关度 + 标题检索），没有 arXiv 全文、Google Scholar 和中文数据库；标题检索对措辞敏感。**投稿前（门 B′）需要再做一轮**，并加入 arXiv 与 CNKI
