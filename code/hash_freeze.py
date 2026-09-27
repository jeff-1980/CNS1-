"""门 A 第 1 项：冻结原始数据与代码哈希。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import hashlib, os, csv, glob, datetime, pathlib
from concurrent.futures import ThreadPoolExecutor
ROOTS = {
    "CWRU40": (_CWRU40, "**/*.mat"),
    "JNU":    (_JNU_RAW, "**/*.csv"),
    "PU":     (_PU, "**/*.mat"),   # 递归：根目录 800 = paperB 10 轴承；子目录 = 本项目 pu_splits 用的其余轴承
}
CODE = ["code/prior_screen.py", "code/cross_eval.py", "code/pu_splits.py", "code/jnu_balance.py",
        "code/analyze_arms.py", "code/analyze_bbp.py", "code/analyze_variance_share.py"] + \
       sorted(glob.glob("code/pbx/data/*.py")) + sorted(glob.glob("code/pbx/models/*.py"))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
rows = []
for ds, (root, pat) in ROOTS.items():
    files = sorted(glob.glob(os.path.join(root, pat), recursive=True))
    with ThreadPoolExecutor(8) as ex: hs = list(ex.map(sha, files))
    for f, h in zip(files, hs):
        st = os.stat(f)
        rows.append(dict(kind="data", dataset=ds, path=f, bytes=st.st_size,
                         mtime=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"), sha256=h))
# 16 文件子集：只记录符号链接 → 目标映射（目标哈希已在 CWRU40 中）
for l in sorted(glob.glob("tmp/cwru_16/*.mat")):
    tgt = os.path.realpath(l)
    rows.append(dict(kind="subset_link", dataset="CWRU16", path=l, bytes=os.stat(tgt).st_size, mtime="", sha256="→ " + tgt))
for c in CODE:
    st = os.stat(c)
    rows.append(dict(kind="code", dataset="-", path=c, bytes=st.st_size,
                     mtime=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"), sha256=sha(c)))
out = "results/registry/raw_hashes.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# 汇总
agg = {}
for ds in ("CWRU40", "JNU", "PU"):
    rs = [r for r in rows if r["dataset"] == ds]
    h = hashlib.sha256("".join(f"{os.path.relpath(r['path'], ROOTS[ds][0])}:{r['sha256']}\n" for r in rs).encode()).hexdigest()
    agg[ds] = (len(rs), sum(r["bytes"] for r in rs), h, sum(r["bytes"] == 0 for r in rs))
sub = [r for r in rows if r["dataset"] == "CWRU16"]
code = [r for r in rows if r["kind"] == "code"]
md = [f"# 原始数据与代码哈希冻结（门 A 第 1 项）\n", f"> 生成：{datetime.datetime.now().isoformat(timespec='seconds')}；脚本 `code/hash_freeze.py`；逐文件明细 `raw_hashes.csv`\n",
      "| 数据集 | 根目录 | 文件数 | 总字节 | 0 字节文件 | 聚合 SHA256（相对路径:哈希 逐行拼接） |", "|---|---|---|---|---|---|"]
for ds, (n, b, h, z) in agg.items():
    md.append(f"| {ds} | `{ROOTS[ds][0]}` | {n} | {b:,} | {z} | `{h}` |")
md += ["", f"CWRU16 子集 = {len(sub)} 个符号链接，目标均在 CWRU40 内（映射见 CSV `kind=subset_link`）。", "",
       f"代码文件 {len(code)} 个（`kind=code`）：`prior_screen.py`、`cross_eval.py`、`pbx/data/*.py`、`pbx/models/*.py` 等。"]
open("results/registry/raw_hashes_manifest.md", "w").write("\n".join(md) + "\n")
for ds, v in agg.items(): print(ds, v[0], v[1], v[3], v[2][:16])
print("CWRU16 links", len(sub), "| code", len(code), "| rows", len(rows))
