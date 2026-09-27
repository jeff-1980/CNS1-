# B 线执行偏离日志（预注册 SHA256 a8f62cc0…）

1. **开发集范围更正**（2026-09-25，写于任何 B 线计算之前）。预注册 §4 把"ext 权重"列入开发集，但 WP5 并未评估 ext（E1 / E2）权重，这是预注册文字的笔误。开发集按实际执行：arm2x2 + arm2x2_lt（100）、CWRU 16 文件 E4 / WP1 NA- 非 Mamba 架构的 best 与 final、PU WP2 全部臂、JNU 600 rpm WP1 NA- 非 Mamba 架构的 best 与 final。**Mamba 类（vibrmamba、mamba2）不进开发集**（需另一环境；开发集只用于选规则，确认集本来就不含 Mamba）。开发集结局由 `b_dev.py` 按预注册 §2 的同一噪声协议自行计算，并与 WP5 的 −8 dB `top_mode` 对照报告。
