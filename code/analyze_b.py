"""B 线判定：严格按 PREREG_B（a8f62cc0…）§6 与 B_RULE_FREEZE（ddc426eb…）。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, hashlib, sys, collections, pathlib
from math import comb
import numpy as np
R = pathlib.Path((_ROOT + "/results"))
for f, h in (("PREREG_B_hierarchical_attractor.md", "prereg_b_sha256.txt"), ("B_RULE_FREEZE.md", "b_rule_freeze_sha256.txt")):
    if hashlib.sha256((R / f).read_bytes()).hexdigest() != (R / "registry" / h).read_text().split()[0]: sys.exit(f"{f} 哈希不符，拒绝判定")
d = json.load(open(R / "b_confirm.json")); att = [r for r in d if r["attractor"]]
out = dict(n=len(d), n_attractor=len(att), n_no_attractor=len(d) - len(att))
print(f"确认实例 {len(d)}；有吸引子 {len(att)}；无吸引子 {len(d)-len(att)}")
tab = collections.defaultdict(collections.Counter)
for r in d: tab[(r["dataset"], str(r["cond"]))][(r["model"], r["macro"] if r["attractor"] else "none", r["a"])] += 1
print("\n条件 × 架构：(宏类结局, C-a 预测) 计数")
for k in sorted(tab): print(f"  {k[0]} {k[1]:12s} " + "  ".join(f"{m}:{o}/{p}×{n}" for (m, o, p), n in sorted(tab[k].items())))
fails = []
if len(d) - len(att) > 0.5 * len(d): fails.append("无吸引子实例 > 50%（§2）")
macros = collections.Counter(r["macro"] for r in att); out["macro_counts"] = dict(macros)
c1 = len(macros) == 2
if not c1: fails.append(f"条件 1：确认集宏类只有 {dict(macros)}，常数预测即 100%")
maj = macros.most_common(1)[0][0] if att else None
A_rule = float(np.mean([r["a"] == r["macro"] for r in att])) if att else float("nan"); A_maj = macros[maj] / len(att) if att else float("nan")
n10 = sum(r["a"] == r["macro"] and maj != r["macro"] for r in att); n01 = sum(r["a"] != r["macro"] and maj == r["macro"] for r in att)
p = sum(comb(n10 + n01, k) for k in range(n10, n10 + n01 + 1)) / 2 ** (n10 + n01) if n10 + n01 else 1.0
c2 = (A_rule - A_maj >= 0.15) and p < 0.05
if not c2: fails.append(f"条件 2：A_rule − A_maj = {A_rule - A_maj:+.3f}，符号检验 p = {p:.4f}（n10={n10}, n01={n01}）")
per_ds = {}
for ds in ("PU", "JNU"):
    rs = [r for r in att if r["dataset"] == ds]
    if not rs: per_ds[ds] = None; continue
    cm = collections.Counter(r["macro"] for r in rs).most_common(1)[0][1] / len(rs); ra = float(np.mean([r["a"] == r["macro"] for r in rs]))
    per_ds[ds] = dict(rule=ra, const=cm, n=len(rs), macro=dict(collections.Counter(r["macro"] for r in rs)))
ok3 = [v for v in per_ds.values() if v]; c3 = len(ok3) == 2 and all(v["rule"] >= v["const"] for v in ok3) and any(v["rule"] > v["const"] for v in ok3)
if not c3: fails.append(f"条件 3：数据集内部 {per_ds}")
per_m = {m: float(np.mean([r["a"] == r["macro"] for r in att if r["model"] == m])) for m in ("wdcnn", "drsn", "lstm") if any(r["model"] == m for r in att)}
c4 = sum(v > A_maj for v in per_m.values()) >= 2
if not c4: fails.append(f"条件 4：各架构规则准确率 {per_m} vs A_maj {A_maj:.3f}")
label = "PASS" if not fails else "KILLED"
desc = {k: float(np.mean([r[k] == r["macro"] for r in att])) if att else None for k in ("b", "c")}
out.update(A_rule=A_rule, A_maj=A_maj, majority=maj, sign_test=dict(n10=n10, n01=n01, p=p), per_dataset=per_ds, per_arch=per_m, conditions=dict(c1=c1, c2=c2, c3=c3, c4=c4),
           fails=fails, label=label, descriptive_Cb_Cc=desc, conc={f"{r['dataset']}|{r['cond']}|{r['model']}|s{r['seed']}": round(r["conc"], 3) for r in d})
print(f"\nA_rule = {A_rule:.3f}   A_maj = {A_maj:.3f}（常数 = {maj}）   符号检验 n10={n10} n01={n01} p={p:.4f}")
print(f"数据集内部：{per_ds}\n架构：{per_m}\n描述（不参与判定）C-b / C-c 准确率：{desc}")
print(f"条件 1–4：{c1} {c2} {c3} {c4}")
for f in fails: print("  ✗", f)
print(f"\n>>> B 线判定：{label}")
json.dump(out, open(R / "b_verdict.json", "w"), indent=1, ensure_ascii=False, default=float)
