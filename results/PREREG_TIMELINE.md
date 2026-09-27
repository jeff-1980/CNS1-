# 预注册时间记录（审稿意见 8）

> 下表是本机文件系统的修改时间（mtime）。预注册文件的 SHA256 已登记在 `registry/`，分析脚本运行时会校验；**这只能证明文件在登记后未被修改，不能向第三方证明它写于实验之前**。mtime 可以被本地改动，也不是可信的第三方时间戳。结果文件的 mtime 是**最后一次写入**的时间，不是开始时间；训练开始时间可以从运行日志与耗时反推。
>
> 投稿时如实写明这一点；今后的预注册应在训练前上传到带时间戳的公共存档（如 OSF Registries 或 Zenodo）。

| 实验 | 性质 | 预注册文件（mtime） | 结果文件（最后写入） | 规模 |
|---|---|---|---|---|
| 2×2 coverage × ratio | training | `PREREG_2x2_coverage_ratio.md`（2026-09-23 20:03） | `arm2x2.json`（2026-09-23 20:58） | 60 instances, ~54 min |
| E1/E2/E3 (cnn1d factorial, size vs file count, lstm/transformer) | training | `PREREG_ext_cnn1d_size_arch.md`（2026-09-23 21:09） | `arm2x2_lt.json`（2026-09-24 02:49） | 3 chained runs |
| E4 reference protocol | training | `PREREG_E4_paperB_checkpoint.md`（2026-09-24 06:52） | `e4_paperB_harness.json`（2026-09-24 07:21） | 15 instances |
| WP1 reference-protocol baseline | training | `PREREG_WP1_protocol_baseline.md`（2026-09-24 09:38） | `wp1_cwru_clean.json`（2026-09-24 12:05） | 92 instances |
| WP2 PU damage extent | training | `PREREG_WP2_pu_extent.md`（2026-09-24 23:46） | `wp2_R.json`（2026-09-25 00:56） | 110 instances |
| WP5 decision consequence | inference on existing weights (frozen before analysis) | `PREREG_WP5_decision_consequence.md`（2026-09-25 12:08） | `wp5_main.json`（2026-09-25 17:44） | 350 weights |
| B line development | inference / rule selection | `PREREG_B_hierarchical_attractor.md`（2026-09-25 12:21） | `b_dev.json`（2026-09-25 13:56） | existing weights |
| B line confirmation | training (rule frozen before) | `B_RULE_FREEZE.md`（2026-09-25 13:57） | `b_confirm.json`（2026-09-25 17:37） | new instances |
| E5 size identification | training | `PREREG_E5_size_identification.md`（2026-09-25 22:21） | `e5_size_id.json`（2026-09-26 05:01） | 189 instances |
| C-INIT fixed-initialization confirmation | training | `PREREG_CINIT_confirmation.md`（2026-09-26 09:02） | `cinit_e5.json`（2026-09-26 17:25） | 269 instances |

**事后分析（未预注册，稿件中逐项标注）**：架构级 bootstrap（`arm2x2_posthoc_archcluster.json`）；WP2 可学性诊断（`wp2_posthoc_learnability.json`）；PU 已见 / 未见轴承重推理（`pu_seen_reanalysis.json`）；WP5b 选择性分类分析（v1 `wp5b_*.json`，v2 `wp5b2_*.json`）；WP5-F1 配对敏感性（`wp5_f1_paired.json`）；历史初始化重建（`init_reconstruction*.json`）；崩溃定义敏感性（`collapse_defs_sensitivity.json`）。
