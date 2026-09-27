"""按 PREREG_WP5_decision_consequence.md 分析。运行时校验预注册哈希。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, hashlib, sys, os, collections, pathlib
import numpy as np
R = pathlib.Path((_ROOT + "/results"))
if hashlib.sha256((R / "PREREG_WP5_decision_consequence.md").read_bytes()).hexdigest() != (R / "registry/prereg_wp5_sha256.txt").read_text().split()[0]:
    sys.exit("预注册哈希不符，拒绝判定")
D, NB, SNR = 0.20, 10000, "-8.0"
d = []
for f in ("wp5_main.json", "wp5_mamba.json"):
    if (R / f).exists(): d += json.load(open(R / f))
rng = np.random.default_rng(0)
v = lambda r, b="B2", m="HER", s=SNR: r["by_snr"][s][b][m]
def lab(delta, lo, hi):
    if abs(delta) >= D and (lo > 0 or hi < 0): return "MATTERS"
    if lo > -D and hi < D: return "EQUIV"
    return "UNDETERMINED"
def factor_label(per):
    ok = [x for x in per.values() if "label" in x]
    mat = [x for x in ok if x["label"] == "MATTERS"]; signs = {np.sign(x["delta"]) for x in mat}
    if len(mat) >= 2 and len(signs) == 1: return "DESIGN_MATTERS"
    if sum(x["label"] == "EQUIV" for x in ok) >= 2: return "NO_CONSEQUENCE"
    return "INDETERMINATE"
def F1(models, b="B2", m="HER", s=SNR):
    per = {}
    for mo in models:
        a = np.array([v(r, b, m, s) for r in d if r["factor"] == "F1" and r["model"] == mo and r["cov"] == "007"])
        c = np.array([v(r, b, m, s) for r in d if r["factor"] == "F1" and r["model"] == mo and r["cov"] == "ALL"])
        if len(a) < 2 or len(c) < 2: per[mo] = dict(n=[len(a), len(c)]); continue
        bs = np.array([rng.choice(a, len(a)).mean() - rng.choice(c, len(c)).mean() for _ in range(NB)]); lo, hi = np.percentile(bs, [2.5, 97.5])
        dl = a.mean() - c.mean(); per[mo] = dict(C007=a.mean(), CALL=c.mean(), delta=dl, ci=[lo, hi], n=[len(a), len(c)], label=lab(dl, lo, hi))
    return per
def F2(models, b="B2", m="HER", s=SNR, fac="F2"):
    per = {}
    for mo in models:
        rs = [r for r in d if r["factor"] == fac and r["model"] == mo]
        pairs = [(v(x, b, m, s), v(y, b, m, s)) for x in rs if x["cond"] == "best" for y in rs if y["cond"] == "final" and y["seed"] == x["seed"]]
        if len(pairs) < 2: per[mo] = dict(n=len(pairs)); continue
        diff = np.array([p - q for p, q in pairs]); bs = np.array([rng.choice(diff, len(diff)).mean() for _ in range(NB)]); lo, hi = np.percentile(bs, [2.5, 97.5])
        per[mo] = dict(best=np.mean([p for p, _ in pairs]), final=np.mean([q for _, q in pairs]), delta=diff.mean(), ci=[lo, hi], n=len(pairs), label=lab(diff.mean(), lo, hi))
    return per
P1, P2 = ["wdcnn", "drsn", "lstm"], ["cnn1d", "wdcnn", "drsn"]
S1, S2 = ["cnn1d", "transformer"], ["lstm", "transformer", "vibrmamba", "mamba2"]
out = {"n_records": len(d), "by_src": dict(collections.Counter(r["src"] for r in d))}
print(f"records {len(d)}  {out['by_src']}")
for name, fn, pm, sm in (("F1 损伤覆盖（C007 − CALL）", F1, P1, S1), ("F2 选模协议（best − final）", F2, P2, S2)):
    per = fn(pm); fl = factor_label(per); key = name[:2]
    out[key] = dict(label=fl, primary=per, secondary_archs=fn(sm))
    print(f"\n=== {name}：主结局 B2 · −8 dB · HER_bal → {fl}")
    for mo, x in per.items():
        print(f"  {mo:11s} " + (f"{x.get('C007', x.get('best')):.3f} vs {x.get('CALL', x.get('final')):.3f}  Δ={x['delta']:+.3f} CI[{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}] → {x['label']}" if "label" in x else str(x)))
    for mo, x in out[key]["secondary_archs"].items():
        if "label" in x: print(f"  [次要架构] {mo:11s} Δ={x['delta']:+.3f} CI[{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}] {x['label']}")
    sec = {}
    for b in ("B0", "B1", "B2"):
        for m in ("HER", "FAR", "MFR", "WTR"):
            for s in ("-8.0", "-4.0", "0.0"):
                if (b, m, s) == ("B2", "HER", SNR): continue
                p = fn(pm, b, m, s); sec[f"{b}|{m}|{s}"] = dict(label=factor_label(p), deltas={k: round(x["delta"], 3) for k, x in p.items() if "delta" in x})
    out[key]["secondary_metrics"] = sec
    print("  [次要指标] " + "  ".join(f"{k}:{x['label'][:6]}" for k, x in sec.items() if k.startswith("B2") or k.startswith("B0|HER") or k.startswith("B0|WTR")))
# H5-S
prim = [r for r in d if (r["factor"] == "F1" and r["model"] in P1) or (r["factor"] == "F2" and r["model"] in P2 and r["src"] == "e4_paperB_harness.json")]
sil = [v(r, "B2", "SILENT") for r in prim]; sil_ok = [x for x in sil if not np.isnan(x)]
fr = np.mean([x >= 0.5 for x in sil_ok]) if sil_ok else float("nan")
hs = "SILENT_UNDER_CONFIDENCE" if fr >= 2 / 3 else ("CAUGHT_BY_CONFIDENCE" if fr <= 1 / 3 else "MIXED")
out["H5_S"] = dict(label=hs, frac_silent_ge_0_5=fr, n=len(sil_ok), n_no_error=len(sil) - len(sil_ok), median_silent=float(np.median(sil_ok)) if sil_ok else None)
print(f"\n=== H5-S 静默性（B2，−8 dB，主架构 {len(sil_ok)} 实例，另 {len(sil)-len(sil_ok)} 个无错误）：SILENT≥0.5 占 {fr:.2f} → {hs}")
# H5-G
fx = [r for r in d if r["factor"] in ("F1", "F2")]
g = np.mean([v(r, "B1", "REJ") >= 0.95 for r in fx]); hg = "GATE_NEUTRALIZES_AT_-8dB" if g >= 0.90 else "GATE_PARTIAL_AT_-8dB"
cells = [(r, s) for r in fx for s in r["by_snr"] if v(r, "B0", "HER", s) >= 0.20]
stop_all = np.mean([v(r, "B1", "REJ", s) >= 0.95 for r, s in cells]) if cells else float("nan")
esc = collections.defaultdict(list)
for r, s in cells:
    if v(r, "B1", "REJ", s) < 0.50: esc[(r["factor"], r["model"])].append(float(s))
out["H5_G"] = dict(label=hg, frac_rej_ge_0_95_at_m8=g, n=len(fx), n_error_cells=len(cells), frac_error_cells_gate_stops=stop_all,
                   highest_snr_gate_misses={f"{a}|{b}": max(x) for (a, b), x in esc.items()}, n_escape_cells={f"{a}|{b}": len(x) for (a, b), x in esc.items()})
print(f"=== H5-G 信号质量门：−8 dB 下 REJ≥0.95 的实例占 {g:.2f}（n={len(fx)}）→ {hg}")
print(f"    B0 HER≥0.20 的 (实例, SNR) 单元 {len(cells)} 个；其中门 REJ≥0.95 的占 {stop_all:.2f}")
print(f"    门拦不住（REJ<0.5）而 B0 HER≥0.2 的最高 SNR：{out['H5_G']['highest_snr_gate_misses']}  单元数：{out['H5_G']['n_escape_cells']}")
# 规划使用
dm = any(out[k]["label"] == "DESIGN_MATTERS" for k in ("F1", "F2"))
gate_all = cells and stop_all >= 0.999
out["WP5_in_plan"] = "✓" if dm and not gate_all else "✗"
# 描述：各组均值表
desc = collections.defaultdict(dict)
for r in d:
    k = (r["group"], r["factor"], r["model"], r.get("cov") or r["cond"] if r["factor"] == "F1" else r["cond"])
    for s, x in r["by_snr"].items():
        for b in ("B0", "B1", "B2"):
            for m in ("HER", "FAR", "MFR", "WTR", "REJ", "SILENT"):
                desc[k].setdefault(f"{s}|{b}|{m}", []).append(x[b][m])
out["descriptive"] = {"|".join(map(str, k)): {kk: float(np.nanmean(vv)) if not all(np.isnan(vv)) else None for kk, vv in x.items()} for k, x in desc.items()}
agree = [r["by_snr"][SNR]["top_mode"] == r["old_top"] for r in d if r.get("old_top") is not None]
out["top_agreement_with_earlier"] = dict(k=int(sum(agree)), n=len(agree))
print(f"\n−8 dB 靶标与此前记录一致：{sum(agree)}/{len(agree)}")
print("\n=== 描述：−8 dB 各组均值（B0 HER / B1 REJ / B2 HER / B2 SILENT / B0 MFR / B0 FAR）")
for k in sorted(out["descriptive"]):
    x = out["descriptive"][k]; f = lambda q: "  -  " if x.get(q) is None else f"{x[q]:.2f}"
    print(f"  {k:42s} {f('-8.0|B0|HER')} {f('-8.0|B1|REJ')} {f('-8.0|B2|HER')} {f('-8.0|B2|SILENT')} {f('-8.0|B0|MFR')} {f('-8.0|B0|FAR')}")
print(f"\n>>> H5-F1: {out['F1']['label']} | H5-F2: {out['F2']['label']} | H5-S: {hs} | H5-G: {hg} → 规划中 WP5 记为 {out['WP5_in_plan']}")
json.dump(out, open(R / "wp5_verdict.json", "w"), indent=1, default=float, ensure_ascii=False)
