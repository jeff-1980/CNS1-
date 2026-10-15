# 执行偏离记录：选模协议审计（预注册 results/PREREG_AUDIT_pilot.md，OSF yszc4）

按 §6，冻结后的执行偏离逐条记录在这里。

| 日期 | 对象 | 偏离 | 理由 | 对结果的影响 |
|---|---|---|---|---|
| 2026-10-15 | P2 Zhao benchmark | `models/BiLSTM1d.py` 一行 `.view` 改为 `.reshape`（`code/audit/ext/patches/zhao_bilstm1d_contiguous.patch`），其余源码未动 | 原代码在 torch 2.13 + cuDNN 上 LSTM 输出非连续，`.view` 直接报错；作者环境为 torch 1.1 | 数值相同（reshape 对非连续张量做拷贝，元素与顺序不变）；“原代码在当前 PyTorch 上无法运行”同时作为审计发现记录 |
| 2026-10-15 | P2–P4 | 依赖版本用 pandas 2.3 / numpy 2.4 / torch 2.13，而非作者的 torch 1.x / pandas 0.24 | pandas 1.x 需 numpy 1.x，与 torch 环境冲突；作者代码只用到 DataFrame 与 read_csv | 预期无影响；复现等级按 BEHAVIORAL 判 |
| 2026-10-15 | P2 | 只跑原始时域输入（CWRU，R_A），未跑 FFT/CWT/STFT 等变体 | 试点阶段控制规模 | 头条主张限定为原始输入下的模型排序 |
