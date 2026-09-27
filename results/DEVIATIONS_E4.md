# E4 执行偏离日志（预注册 SHA256 a3ab8c93… 之后、任何结果之前）

1. **`num_workers` 4 → 0**（2026-09-24）。首次启动时，DataLoader 在创建 worker 阶段报错 `PermissionError: [Errno 1] Operation not permitted`（沙盒不允许 multiprocessing socket）。
   视为数值等价的依据：`RandomSampler` 的 shuffle 顺序由主进程的 torch 全局生成器产生，与 worker 数无关；NA- 训练（`noise_type=None`）时 `BearingDataset.__getitem__` 不消耗任何随机数。报错发生在第一个 batch 之前，没有产生任何结果。
