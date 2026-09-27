"""二轮复核 R3：WP5-F1 的配对敏感性分析（事后，不替换预注册判定）。
预注册（PREREG_WP5 §统计）规定 F1 为"10 vs 10 独立实例、按侧重抽"；但 C007 与 CALL 臂共享种子与同比例下的 Normal 窗、测试噪声，
所以另做配对：按 (seed, ratio) 作差 → 种子内对两个比例取平均 → 对 5 个种子 bootstrap 10 000 次。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, pathlib, numpy as np
R = pathlib.Path((_ROOT + "/results")); D, NB = 0.20, 10000
d = json.load(open(R / "wp5_main.json")); pre = json.load(open(R / "wp5_verdict.json"))["F1"]
v = lambda r: r["by_snr"]["-8.0"]["B2"]["HER"]
def lab(dl, lo, hi):
    if abs(dl) >= D and (lo > 0 or hi < 0): return "MATTERS"
    if lo > -D and hi < D: return "EQUIV"
    return "UNDETERMINED"
out = {}
for m in ("wdcnn", "drsn", "lstm", "cnn1d", "transformer"):
    rs = {(r["cond"], r["seed"]): v(r) for r in d if r["factor"] == "F1" and r["model"] == m}
    seeds = sorted({s for _, s in rs})
    per_seed = np.array([np.mean([rs[(f"C007_{q}", s)] - rs[(f"CALL_{q}", s)] for q in ("R11", "R31")]) for s in seeds])
    rng = np.random.default_rng(0); bs = np.array([rng.choice(per_seed, len(per_seed)).mean() for _ in range(NB)]); lo, hi = np.percentile(bs, [2.5, 97.5])
    p = pre["primary"].get(m) or pre["secondary_archs"].get(m)
    out[m] = dict(delta=float(per_seed.mean()), ci_paired=[float(lo), float(hi)], label_paired=lab(per_seed.mean(), lo, hi), per_seed=per_seed.round(4).tolist(),
                  ci_prereg_unpaired=p["ci"], label_prereg=p["label"], point_beyond_bound=bool(abs(per_seed.mean()) >= D), ci_entirely_beyond_bound=bool(lo >= D or hi <= -D))
    print(f"{m:11s} Δ={per_seed.mean():+.4f} paired CI [{lo:+.3f}, {hi:+.3f}] {out[m]['label_paired']} | prereg CI [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}] {p['label']} | point≥0.20 {out[m]['point_beyond_bound']} CI beyond {out[m]['ci_entirely_beyond_bound']}")
F2 = json.load(open(R / "wp5_verdict.json"))["F2"]
out["F2_note"] = "F2 already paired (best vs final of the same run, per-seed differences bootstrapped)"
out["F2_bound"] = {m: dict(delta=x["delta"], ci=x["ci"], ci_entirely_beyond_bound=bool(x["ci"][0] >= D or x["ci"][1] <= -D)) for m, x in {**F2["primary"], **F2["secondary_archs"]}.items() if "ci" in x}
print({m: (round(x["delta"], 3), [round(c, 3) for c in x["ci"]], x["ci_entirely_beyond_bound"]) for m, x in out["F2_bound"].items()})
json.dump(dict(note="POST HOC paired sensitivity (review round 2, R3); does not replace preregistered WP5 labels", **out), open(R / "wp5_f1_paired.json", "w"), indent=1)
