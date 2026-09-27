"""二轮复核 R2：WP5b v2 汇总，按训练条件与数据集分开（事后分析，不改变 WP5 预注册判定）。
总体：CWRU40-F1（干净训练，最后一轮权重）、CWRU16-F2-best / -final（干净训练，参考协议）、CWRU16-F3（加噪训练，单列）、PU-R、PU-A。
主文"干净训练 CWRU"= F1 ∪ F2（与 Fig. 5a–d 相同）。risk 两种汇总：instance_mean = 覆盖率 > 0 的实例上 selective risk 的均值；
pooled = Σ 已接受错误 / Σ 已接受窗口。覆盖率为 0 的实例 risk 未定义，单独计数。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, collections, pathlib, numpy as np
R = pathlib.Path((_ROOT + "/results"))
d = []
for f in ("wp5b2_main.json", "wp5b2_mamba.json"):
    if (R / f).exists(): d += json.load(open(R / f))
def pop(r):
    if r["factor"] == "F1": return "CWRU40_clean_F1"
    if r["factor"] == "F2": return f"CWRU16_clean_F2_{r['cond']}"
    if r["factor"] == "F3": return f"CWRU16_noisetrained_F3_{r['cond']}"
    return "PU_" + r["group"][2:]
def summ(rs, snr):
    x = [r["by_snr"][snr]["B2"] for r in rs]; n_win = sum(r["test_counts"]) if isinstance(rs[0]["test_counts"], list) else None
    N = [sum(r["test_counts"]) for r in rs]
    acc_n = [v["coverage"] * n for v, n in zip(x, N)]; err_n = [v["acc_err_over_all"] * n for v, n in zip(x, N)]
    rd = [v["risk_accepted"] for v in x if v["risk_accepted"] is not None]
    return dict(n=len(rs), coverage=float(np.mean([v["coverage"] for v in x])), coverage_min=float(min(v["coverage"] for v in x)), coverage_max=float(max(v["coverage"] for v in x)),
                risk_instance_mean=float(np.mean(rd)) if rd else None, n_risk_defined=len(rd), risk_pooled=float(sum(err_n) / sum(acc_n)) if sum(acc_n) > 0 else None,
                accepted_errors_per_window=float(np.mean([v["acc_err_over_all"] for v in x])),
                accept_healthy=float(np.mean([v["accept_by_class"]["0"] for v in x])), accept_fault=float(np.mean([np.mean([v["accept_by_class"][k] for k in v["accept_by_class"] if k != "0"]) for v in x])),
                aurc=float(np.mean([v["aurc"] for v in x])), frac_conf_saturated=float(np.mean([v["frac_conf_saturated"] for v in x])), largest_tie_frac=float(np.mean([v["largest_tie_frac"] for v in x])),
                rc=np.mean([v["rc"] for v in x], 0).round(4).tolist())
out = dict(note=__doc__, n=len(d), cells={})
G = collections.defaultdict(list)
for r in d:
    G[(pop(r), r["model"])].append(r)
    if r["factor"] in ("F1", "F2"): G[("CWRU_clean_F1F2", r["model"])].append(r)
    if r["factor"] == "PU": G[("PU_all", r["model"])].append(r)
for (p, m), rs in sorted(G.items()):
    out["cells"][f"{p}|{m}"] = {snr: summ(rs, snr) for snr in ("10.0", "0.0", "-8.0")}
f = lambda x: "  -  " if x is None else f"{x:.3f}"
for p in ("CWRU_clean_F1F2", "CWRU40_clean_F1", "CWRU16_clean_F2_best", "CWRU16_clean_F2_final", "CWRU16_noisetrained_F3_best", "CWRU16_noisetrained_F3_final", "PU_all"):
    for k in sorted(k for k in out["cells"] if k.startswith(p + "|")):
        c = out["cells"][k]["-8.0"]; c0 = out["cells"][k]["0.0"]
        print(f"{k:42s} n={c['n']:3d} −8dB cov {c['coverage']:.4f} [{c['coverage_min']:.3f},{c['coverage_max']:.3f}] risk inst {f(c['risk_instance_mean'])} ({c['n_risk_defined']}) pooled {f(c['risk_pooled'])} sat {c['frac_conf_saturated']:.2f} tie {c['largest_tie_frac']:.2f} | 0dB cov {c0['coverage']:.3f}")
json.dump(out, open(R / "wp5b2_summary.json", "w"), indent=1)
