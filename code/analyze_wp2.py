"""按 PREREG_WP2_pu_extent.md 分析。运行时校验预注册哈希。"""
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
import numpy as np
R = pathlib.Path((_ROOT + "/results"))
if hashlib.sha256((R / "PREREG_WP2_pu_extent.md").read_bytes()).hexdigest() != (R / "registry/prereg_wp2_sha256.txt").read_text().split()[0]:
    sys.exit("预注册哈希不符，拒绝判定")
CLS = {0: "Healthy", 1: "OR", 2: "IR"}; PRIMARY = ["wdcnn", "drsn", "lstm"]; D, NB = 0.20, 10000
d = []
for f in ("wp2_R.json", "wp2_A.json", "wp2_R_single.json"):
    if (R / f).exists(): d += json.load(open(R / f))
def P(rs, cls, snr="-8.0", need=False):
    v = [int(r["observed"][snr]["top_mode"] == cls) for r in rs if not need or r["observed"][snr]["conc_mean"] >= 0.90]
    return (np.mean(v) if v else np.nan), len(v)
def get(block, arm, m): return sorted([r for r in d if r["arm"] == arm and r["model"] == m], key=lambda r: r["seed"])
def block_eval(b, cls=2, snr="-8.0", need=False, models=PRIMARY):
    L1, La, Lb = f"{b}-L1", f"{b}-L12a", f"{b}-L12b"; rng = np.random.default_rng(0); per = {}
    for m in models:
        A1, Aa, Ab = get(b, L1, m), get(b, La, m), get(b, Lb, m)
        if min(len(A1), len(Aa), len(Ab)) < 5: per[m] = dict(missing=[len(A1), len(Aa), len(Ab)]); continue
        y = lambda rs: np.array([int(r["observed"][snr]["top_mode"] == cls) for r in rs if not need or r["observed"][snr]["conc_mean"] >= 0.90], float)
        y1, ya, yb = y(A1), y(Aa), y(Ab)
        if min(len(y1), len(ya), len(yb)) == 0: per[m] = dict(empty=True); continue
        delta = y1.mean() - 0.5 * (ya.mean() + yb.mean()); da, db = y1.mean() - ya.mean(), y1.mean() - yb.mean()
        bs = [rng.choice(y1, len(y1)).mean() - 0.5 * (rng.choice(ya, len(ya)).mean() + rng.choice(yb, len(yb)).mean()) for _ in range(NB)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        cw = delta >= D and lo > 0 and da > 0 and db > 0
        rev = delta <= -D and hi < 0
        ind = (np.sign(da) != np.sign(db)) and abs(da) >= D and abs(db) >= D
        eq = lo > -D and hi < D
        per[m] = dict(P_L1=y1.mean(), P_L12a=ya.mean(), P_L12b=yb.mean(), delta=delta, ci=[lo, hi], d_a=da, d_b=db, n=[len(y1), len(ya), len(yb)], cwru_dir=bool(cw), reversed=bool(rev), individual=bool(ind), equiv=bool(eq))
    ok = [v for v in per.values() if "delta" in v]
    k = lambda key: sum(v[key] for v in ok)
    label = "WP2_PASS" if k("cwru_dir") >= 2 else "REVERSED" if k("reversed") >= 2 else "INDIVIDUAL" if k("individual") >= 2 else "NO_SHIFT" if k("equiv") >= 2 else "INDETERMINATE"
    if len(ok) < 3 and models == PRIMARY: label += f"(only {len(ok)}/3 archs complete)"
    return label, per
out = {"n_records": len(d)}
for b in ("R", "A"):
    lab, per = block_eval(b); out[f"{b}_primary"] = dict(label=lab, per_arch=per)
    print(f"\n=== 块 {b} 主分析（−8 dB，Δ_IR）→ {lab}")
    for m, v in per.items():
        if "delta" in v: print(f"  {m:6s} P_IR L1={v['P_L1']:.2f} L12a={v['P_L12a']:.2f} L12b={v['P_L12b']:.2f}  Δ={v['delta']:+.2f} CI[{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]  dir={v['cwru_dir']} rev={v['reversed']} ind={v['individual']} eq={v['equiv']}")
        else: print(f"  {m:6s} {v}")
    out[f"{b}_secondary"] = {}
    for tag, kw in {"dOR_-8": dict(cls=1), "dIR_-8_conc90": dict(need=True), "dIR_-10": dict(snr="-10.0"), "dIR_-12": dict(snr="-12.0"), "dHealthy_-8": dict(cls=0)}.items():
        l2, p2 = block_eval(b, **kw); out[f"{b}_secondary"][tag] = dict(label=l2, per_arch=p2)
        print(f"  [次要] {tag:14s} {l2:32s} " + "  ".join(f"{m}:{v['delta']:+.2f}" for m, v in p2.items() if "delta" in v))
    print(f"  −8 dB 靶标分布 / 浓度 / +10 dB 准确率：")
    for arm in BLOCK_ARMS if False else [f"{b}-L1", f"{b}-L12a", f"{b}-L12b"]:
        for m in ["wdcnn", "drsn", "lstm", "cnn1d", "transformer"]:
            rs = get(b, arm, m)
            if not rs: continue
            tg = collections.Counter(CLS[r["observed"]["-8.0"]["top_mode"]] for r in rs)
            cc = [r["observed"]["-8.0"]["conc_mean"] for r in rs]; acc = np.mean([r["observed"]["10.0"]["acc_mean"] for r in rs])
            flag = "  ⚠未学会" if acc < 0.90 else ""
            out.setdefault(f"{b}_dist", {})[f"{arm}|{m}"] = dict(targets=dict(tg), conc=[min(cc), max(cc)], acc10=acc)
            print(f"    {arm:7s} {m:11s} n={len(rs)} {dict(tg)}  conc {min(cc):.2f}–{max(cc):.2f}  acc+10={acc:.3f}{flag}")
lab, per = block_eval("R", models=["cnn1d", "transformer"])
print("\n=== 单列：cnn1d / transformer（R-L1 vs R-L12a）")
for m in ("cnn1d", "transformer"):
    A1, Aa = get("R", "R-L1", m), get("R", "R-L12a", m)
    if A1 and Aa:
        p1, pa = P(A1, 2)[0], P(Aa, 2)[0]; out.setdefault("single", {})[m] = dict(P_IR_L1=p1, P_IR_L12a=pa, diff=p1 - pa)
        print(f"  {m:11s} P_IR L1={p1:.2f} L12a={pa:.2f} 差={p1-pa:+.2f}")
out["WP2_in_plan"] = "✓" if out["R_primary"]["label"] == "WP2_PASS" else "✗"
print(f"\n>>> 真实损伤块判定: {out['R_primary']['label']} → 规划中 WP2 记为 {out['WP2_in_plan']}；人工损伤块（补充）: {out['A_primary']['label']}")
json.dump(out, open(R / "wp2_verdict.json", "w"), indent=1, default=float, ensure_ascii=False)
