"""审稿意见 6：崩溃定义与聚合顺序的敏感性（事后分析，不改变任何预注册判定）。
定义：主导类别 = 5 次噪声抽样各自最大类的众数；集中度 = 5 次最大类占比的均值；崩溃事件 = 集中度 ≥ 0.90（主），≥ 0.60（次）。
另一聚合（先平均类分布、再取最大）：若 5 次抽样的最大类全部相同，两种聚合在数学上完全相等；只有不一致的实例可能不同，且此时"先平均"的值 ≤ 集中度均值。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, collections, numpy as np, pathlib
R = pathlib.Path((_ROOT + "/results"))
SRC = {"CWRU 2×2 (cnn1d/wdcnn/drsn)": "arm2x2.json", "CWRU 2×2 (lstm/transformer)": "arm2x2_lt.json",
       "PU WP2 real": "wp2_R.json", "PU WP2 artificial": "wp2_A.json", "PU WP2 real (cnn1d/transformer)": "wp2_R_single.json", "CWRU E5": "e5_size_id.json"}
out = {}
for lab, f in SRC.items():
    if not (R / f).exists(): continue
    d = json.load(open(R / f)); rows = collections.Counter(); per = collections.defaultdict(collections.Counter)
    for r in d:
        o = r["observed"]["-8.0"]; tops = o["tops"]; agree = max(collections.Counter(tops).values())
        c = o["conc_mean"]; rows["n"] += 1; rows[f"agree_{agree}of5"] += 1
        rows["event_090"] += c >= 0.90; rows["event_060"] += c >= 0.60; rows["dominant_but_no_event_060"] += c < 0.60
        per[r["model"]]["n"] += 1; per[r["model"]]["unanimous"] += agree == 5; per[r["model"]]["event_090"] += c >= 0.90
    out[lab] = dict(overall=dict(rows), per_model={m: dict(v) for m, v in per.items()})
    u = rows["agree_5of5"] / rows["n"]
    print(f"{lab:34s} n={rows['n']:3d} 5/5 一致 {rows['agree_5of5']:3d} ({u:.0%})  集中度≥0.90 {rows['event_090']:3d}  ≥0.60 {rows['event_060']:3d}  主导类别但未达 0.60 {rows['dominant_but_no_event_060']}")
json.dump(out, open(R / "collapse_defs_sensitivity.json", "w"), indent=1, ensure_ascii=False)
