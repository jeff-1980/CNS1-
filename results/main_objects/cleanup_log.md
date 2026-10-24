# 正式阶段存储清理记录

按 jeff 2026-10-24 授权执行（10:46 UTC 消息）：删除 `~/audit_ext` 下已完成对象的中间文件与 checkpoint；kaist64、seu51、hit23、sqv38、awt 的原始数据移到 `D:\audit_data`（复制后逐文件 sha256 校验一致再删 WSL 副本）；保留 PU、CWRU、`~/data_pu`、对象 60 与 71 用到的数据及各对象环境；不打断对象 60。压缩 vhdx 需 GPU 空闲后另行确认（未做）。

## 2026-10-24 执行

- 数据迁移（原路径留符号链接 `~/audit_ext/data/<名> → /mnt/d/audit_data/<名>`，各目录内存 `SHA256SUMS_audit.txt`）：

| 数据 | 大小 | 文件数 | 对象 | 校验 |
|---|---|---|---|---|
| kaist64 | 2.1 GB | 31 | 64 | sha256 全部一致 |
| seu51 | 3.4 GB | 2 | 51 | sha256 全部一致 |
| hit23 | 2.7 GB | 5 | 23 | sha256 全部一致 |
| sqv38 | 2.3 GB | 164 | 38 | sha256 全部一致 |
| awt | 1.2 GB | 13 | 15 | sha256 全部一致 |

- 删除的工作目录（已完成对象的中间文件：预处理缓存、临时输出、checkpoint）：work04 372M、work08 25M、work11 4.8M、work13 36M、work14 130M、work15 18M、work18 210M、work19、work20 21M、work21 102M、work23 4.6G（`prep/` 预处理缓存 4.5G）、work24 98M、work25、work30 114M、work31、work34 93M、work36 339M、work38 73M、work42、work49 4.6M、work51 8.6M、work53 78M、work54home 2.8M、work62 44M、work64 51M、work73（符号链接树，只删链接，`~/data_pu` 未动）。保留 work71（对象 71 待跑）。
- 删除仓库内运行产生的未跟踪 checkpoint / 缓存（只删 `git ls-files --others` 中扩展名为 pth/pt/pkl 等或位于 tempdata 的文件；仓库自带文件、审计 driver、日志均未动；对象 60、71 的仓库不处理）：73 共 34 个文件 538 MB（`tempdata/5way.pkl` 数据缓存 + 各结果目录的 `.pkl` 模型）、74 共 153 个文件 519 MB（`tempdata*/5way.pkl` + `results/` 下 `.pkl` 模型）、39 共 16 个 `.pth` 33 MB。
- 未处理：对象 30 仓库内 `data/hit_full`（0.79 GB，该对象的数据，不在清单内）、各对象环境 `envs/`、`main_records`、PU / CWRU 数据。
- 结果：WSL 内部释放约 19 GB（`/` 已用 490G → 471G）；D 盘 125G → 114G 可用。**C 盘不会因此变空**：WSL 虚拟磁盘不自动收缩，需 GPU 空闲后 `wsl --shutdown` 再压缩 vhdx（待 jeff 确认）。
