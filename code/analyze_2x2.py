"""按 PREREG_2x2_coverage_ratio.md 分析。运行时校验预注册哈希，不一致则拒绝输出判定。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, hashlib, collections, sys, pathlib
import numpy as np
RES = pathlib.Path((_ROOT + "/results"))
PRE = RES / "PREREG_2x2_coverage_ratio.md"
EXPECT = (RES / "registry/prereg_sha256.txt").read_text().split()[0]
GOT = hashlib.sha256(PRE.read_bytes()).hexdigest()
if GOT != EXPECT: sys.exit(f"预注册哈希不符：{GOT} != {EXPECT}，拒绝判定")
DELTA, NB, CLS = 0.20, 10000, {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}
ARMS = ["C007_R11", "C007_R31", "CALL_R11", "CALL_R31"]

def load():
    d = json.load(open(RES / "arm2x2.json"))
    assert len({(r["arm"], r["model"], r["seed"]) for r in d}) == len(d)
    return d

def Y(r, snr="-8.0", target=1, need_conc=False):
    o = r["observed"][snr]
    if need_conc and o["conc_mean"] < 0.90: return None
    return int(o["top_mode"] == target)

def effects(P):
    dC = 0.5 * ((P["C007_R11"] - P["CALL_R11"]) + (P["C007_R31"] - P["CALL_R31"]))
    dR = 0.5 * ((P["C007_R31"] - P["C007_R11"]) + (P["CALL_R31"] - P["CALL_R11"]))
    I = (P["C007_R31"] - P["CALL_R31"]) - (P["C007_R11"] - P["CALL_R11"])
    return dC, dR, I

def analyse(d, **kw):
    cells = {a: collections.defaultdict(list) for a in ARMS}
    for r in d:
        y = Y(r, **kw)
        if y is not None: cells[r["arm"]][r["model"]].append(y)
    P = {a: np.mean([v for m in cells[a].values() for v in m]) for a in ARMS}
    n = {a: sum(len(m) for m in cells[a].values()) for a in ARMS}
    rng = np.random.default_rng(0); boots = []
    for _ in range(NB):
        Pb = {}
        for a in ARMS:
            vals = [v for m, ys in cells[a].items() for v in rng.choice(ys, len(ys), replace=True)]
            Pb[a] = np.mean(vals)
        boots.append(effects(Pb))
    B = np.array(boots); est = effects(P)
    ci = [tuple(np.percentile(B[:, k], [2.5, 97.5])) for k in range(3)]
    return P, n, est, ci

def label(est, ci):
    (lc, hc), (lr, hr), (li, hi) = ci
    inC, inR = (-DELTA < lc and hc < DELTA), (-DELTA < lr and hr < DELTA)
    if hc < -DELTA or hr < -DELTA: t = "REVERSED"
    elif lc > DELTA and lr > DELTA: t = "BOTH"
    elif lc > DELTA and inR: t = "COVERAGE"
    elif lr > DELTA and inC: t = "RATIO"
    elif inC and inR: t = "NULL_MATCHED"
    else: t = "INDETERMINATE"
    if li > DELTA or hi < -DELTA: t += "+INTERACTION"
    return t

def main():
    d = load(); print("instances", len(d), collections.Counter(r["arm"] for r in d))
    out = {"prereg_sha256": GOT, "n_instances": len(d)}
    for name, kw in (("primary_-8dB_IR", {}), ("secondary_conc>=0.9", {"need_conc": True}),
                     ("secondary_-6dB_IR", {"snr": "-6.0"}), ("aux_-8dB_OR", {"target": 2})):
        P, n, est, ci = analyse(d, **kw)
        t = label(est, ci) if name == "primary_-8dB_IR" else None
        out[name] = dict(P=P, n=n, dC=est[0], dR=est[1], I=est[2], ci_dC=ci[0], ci_dR=ci[1], ci_I=ci[2], label=t)
        print(f"\n[{name}] " + "  ".join(f"{a}={P[a]:.2f}(n={n[a]})" for a in ARMS))
        print(f"  ΔC={est[0]:+.2f} [{ci[0][0]:+.2f},{ci[0][1]:+.2f}]  ΔR={est[1]:+.2f} [{ci[1][0]:+.2f},{ci[1][1]:+.2f}]  I={est[2]:+.2f} [{ci[2][0]:+.2f},{ci[2][1]:+.2f}]" + (f"  → {t}" if t else ""))
    tab = collections.defaultdict(collections.Counter); conc = collections.defaultdict(list)
    for r in d:
        tab[(r["arm"], r["model"])][CLS[r["observed"]["-8.0"]["top_mode"]]] += 1; conc[(r["arm"], r["model"])].append(r["observed"]["-8.0"]["conc_mean"])
    print("\n=== 逐架构 −8 dB 靶标 ===")
    out["per_arch"] = {}
    for a in ARMS:
        for m in ("cnn1d", "wdcnn", "drsn"):
            k = (a, m); out["per_arch"][f"{a}|{m}"] = dict(targets=dict(tab[k]), conc_min=min(conc[k]), conc_max=max(conc[k]))
            print(f"  {a:9s} {m:6s} {dict(tab[k])}  conc {min(conc[k]):.2f}–{max(conc[k]):.2f}")
    acc10 = {a: np.mean([r["observed"]["10.0"]["acc_mean"] for r in d if r["arm"] == a]) for a in ARMS}
    out["clean_acc_10dB"] = acc10; print("\n+10 dB acc:", {a: round(v, 4) for a, v in acc10.items()})
    json.dump(out, open(RES / "arm2x2_verdict.json", "w"), indent=1, ensure_ascii=False, default=float)
    print("\n>>> 判定:", out["primary_-8dB_IR"]["label"])

if __name__ == "__main__":
    main()
