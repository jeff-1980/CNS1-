# 执行偏离记录：选模协议审计（预注册 results/PREREG_AUDIT_pilot.md，OSF yszc4）

按 §6，冻结后的执行偏离逐条记录在这里。

| 日期 | 对象 | 偏离 | 理由 | 对结果的影响 |
|---|---|---|---|---|
| 2026-10-15 | P2 Zhao benchmark | `models/BiLSTM1d.py` 一行 `.view` 改为 `.reshape`（`code/audit/ext/patches/zhao_bilstm1d_contiguous.patch`），其余源码未动 | 原代码在 torch 2.13 + cuDNN 上 LSTM 输出非连续，`.view` 直接报错；作者环境为 torch 1.1 | 数值相同（reshape 对非连续张量做拷贝，元素与顺序不变）；“原代码在当前 PyTorch 上无法运行”同时作为审计发现记录 |
| 2026-10-15 | P2–P4 | 依赖版本用 pandas 2.3 / numpy 2.4 / torch 2.13，而非作者的 torch 1.x / pandas 0.24 | pandas 1.x 需 numpy 1.x，与 torch 环境冲突；作者代码只用到 DataFrame 与 read_csv | 预期无影响；复现等级按 BEHAVIORAL 判 |
| 2026-10-15 | P2 | 只跑原始时域输入（CWRU，R_A），未跑 FFT/CWT/STFT 等变体 | 试点阶段控制规模 | 头条主张限定为原始输入下的模型排序 |
| 2026-10-15 | P2–P4 汇总 | §4 的 H1 按“某组过半运行在前 5% epoch 内达到 ≥ 0.999”计（`code/audit/ext/partII_summary.py`），同时报告“任一运行”口径；§3.3 的“排名改变”按“作者协议下严格有序的组对在替代协议下严格反序”计，并列变有序不算；配对 CI 按 rep 序号配对 | 预注册未写到运行级与并列的处理；P2/P3 作者代码无种子，P4 固定种子 999，没有可对应的种子 | 两种 H1 口径都列出；判定用较严的口径 |
| 2026-10-15 | P3 UDTL | 运行脚本的“已完成跳过”检查查错了日志名（UDTL 写 `train.log`，不是 `training.log`），已改 | 仅影响重跑时是否跳过 | 无：20 次都是完整新跑 |
| 2026-10-15 | P5 pdm-bench | 不改源码，经 `code/audit/ext/pdm_hook.py` 在每轮结束时用作者的 evaluate 与损失函数在 test 上多评估一次（RNG 与模型模式隔离），得到逐 epoch 的 test 曲线；作者口径 = 最后一轮，另报 best（按 test 挑） | 作者代码只在最后一轮评估 test、无验证集，没有逐 epoch 记录就算不出替代协议 | 训练轨迹不变（评估不更新参数，RNG 已隔离）；最后一轮记录与作者 TEST 结果逐次核对 |
| 2026-10-15 | P5 pdm-bench | 只跑 cwru_cross_load，4 个 DL 模型（cnn1d / fft / stft / mlp），作者 "long" 档，种子 41–45；数据用本机 CWRU 12k DE 的 10 类文件按 Kaggle 目录建软链接，不含 0.028 与外圈 @3/@12 | 试点规模；本机没有 Kaggle 版全集 | 测试集类别构成与作者完整数据不同，结果只用于协议比较，不与原文数值对照 |
| 2026-10-16 | P5 pdm-bench | 运行时多传 `--config-path $PDM_DIR/config`（commit dda8f25）；CWRU 软链接源为 `~/audit_ext/data/cwru_udtl`（四个负载齐全），最终指向 `~/论文CNS/data/cwru_12k_de` 原文件；xgboost 以 --no-deps 安装 | 由审计钩子导入后调用 Hydra main 时，相对 config_path 被当成包路径；`cwru_zhao` 只有 load 0；DL 流水线不用 xgboost | 无：作者代码与配置未改，20 次最后一轮记录均与作者 TEST 一致 |
