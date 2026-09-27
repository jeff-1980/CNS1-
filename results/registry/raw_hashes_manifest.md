# 原始数据与代码哈希冻结（门 A 第 1 项）

> 生成：2026-09-23T19:51:06；脚本 `code/hash_freeze.py`；逐文件明细 `raw_hashes.csv`

| 数据集 | 根目录 | 文件数 | 总字节 | 0 字节文件 | 聚合 SHA256（相对路径:哈希 逐行拼接） |
|---|---|---|---|---|---|
| CWRU40 | `/home/jeffwork/论文8/data/cwru_12k_de` | 40 | 140,322,880 | 0 | `825df805ff19e8a82b895a589bf02ac313080bed07d4e5a214c7dd38e7a27136` |
| JNU | `/home/jeffwork/JNU-Bearing-Dataset` | 12 | 78,585,213 | 0 | `a892bc7fa9159ea09fe577ceb0a4fa53d8969e894a7d52b11bdf9132186540e3` |
| PU | `/home/jeffwork/data_pu` | 2240 | 19,560,863,456 | 0 | `4d3146b9f68344904ff8ac8d5ba21dafd958a664dd216c4d59208da2d03c1548` |

CWRU16 子集 = 16 个符号链接，目标均在 CWRU40 内（映射见 CSV `kind=subset_link`）。

代码文件 21 个（`kind=code`）：`prior_screen.py`、`cross_eval.py`、`pbx/data/*.py`、`pbx/models/*.py` 等。

## 核对记录

- **PU 计数**：递归 2240 个 .mat（README 的"800 个平铺在根目录"只是 paperB 所用 10 个轴承 × 4 工况 × 20 次；其余 18 个子目录各 80 个）。
- **PU 内容重复**：80 组 / 160 个文件逐字节相同，全部是 `KI05/` 子目录中误放的 **KI01** 副本（例：`KI05/N09_M07_F10_KI01_1.mat` ≡ 根目录 `N09_M07_F10_KI01_1.mat`）。
  本项目 PU 扫掠（`pu_fine.log`）实际用 `{0: K001,K002,K003; 1: KA04,KA15,KA16; 2: KI04,KI14,KI17}`，paperB 用根目录 → **两侧都不读 `KI05/`，已有结果不受影响**。
- `pu_splits.py` 名单中的 **KA30、KI16 在磁盘上不存在**；因 `codes_per_class=3` 只取前三个，未触发。
- 所有数据文件 0 字节计数 = 0（本项目使用的 `论文8/data/cwru_12k_de` 为好副本；库内坏副本不在冻结范围）。
