"""按 PREREG_ext_cnn1d_size_arch.md 分析 E1/E2/E3。运行时校验预注册哈希。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, hashlib, collections, sys, pathlib, argparse
import numpy as np
RES = pathlib.Path((_ROOT + "/results"))
PRE = RES / "PREREG_ext_cnn1d_size_arch.md"
if hashlib.sha256(PRE.read_bytes()).hexdigest() != (RES / "registry/prereg_ext_sha256.txt").read_text().split()[0]:
    sys.exit("预注册哈希不符，拒绝判定")
D, NB, CLS = 0.20, 10000, {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}

def lab(lo, hi):
    if lo > D: return "TOWARD_IR"
    if hi < -D: return "TOWARD_OR"
    if -D < lo and hi < D: return "NO_EFFECT"
    return "INDETERMINATE"

def ci(a): return tuple(np.percentile(a, [2.5, 97.5]).tolist())

def e1():
    d = [r for r in json.load(open(RES / "ext_e1.json"))]
    Yf = lambda r: int(r["final"]["-8.0"]["top_mode"] == 1)
    Yv = lambda r: int(r["valbest"]["-8.0"]["top_mode"] == 1)
    cells = collections.defaultdict(list)
    for r in d: cells[(r["ID"], r["N"], r["S"])].append(r)
    out = {"n": len(d), "cells": {}}
    print(f"E1 instances {len(d)}")
    for k in sorted(cells):
        rs = cells[k]; row = dict(n=len(rs), final=dict(collections.Counter(CLS[r["final"]["-8.0"]["top_mode"]] for r in rs)),
                                  conc_final=[round(r["final"]["-8.0"]["conc_mean"], 2) for r in rs])
        if k[0] == 1:
            row["valbest"] = dict(collections.Counter(CLS[r["valbest"]["-8.0"]["top_mode"]] for r in rs))
            row["best_epoch"] = [r["best_epoch"] for r in rs]
        out["cells"]["ID%dN%dS%d" % k] = row
        print(f"  ID{k[0]}N{k[1]}S{k[2]}  final={row['final']}  " + (f"valbest={row['valbest']} best_ep={row['best_epoch']}" if k[0] else ""))
    e0 = sum(r["final"]["-8.0"]["top_mode"] == 2 for r in cells[(0, 0, 0)]) >= 4
    e1_ = sum(Yv(r) for r in cells[(1, 1, 1)]) >= 4
    out["endpoint"] = dict(base_OR_ge4=e0, natural_valbest_IR_ge4=e1_)
    print("  endpoint checks:", out["endpoint"])
    keys = sorted(cells); Ys = {k: np.array([Yf(r) for r in cells[k]]) for k in keys}
    def mains(Y):
        m = {}
        for f, pos in (("ID", 0), ("N", 1), ("S", 2)):
            on = np.concatenate([Y[k] for k in keys if k[pos] == 1]); off = np.concatenate([Y[k] for k in keys if k[pos] == 0])
            m[f] = on.mean() - off.mean()
        return m
    est = mains(Ys); rng = np.random.default_rng(0); B = collections.defaultdict(list)
    for _ in range(NB):
        Yb = {k: rng.choice(v, len(v), replace=True) for k, v in Ys.items()}
        for f, v in mains(Yb).items(): B[f].append(v)
    out["main"] = {}
    for f in ("ID", "N", "S"):
        c = ci(B[f]); out["main"][f] = dict(est=est[f], ci=c, label=lab(*c)); print(f"  M_{f} = {est[f]:+.2f} [{c[0]:+.2f},{c[1]:+.2f}] {lab(*c)}")
    pairs = {k: np.array([(Yv(r), Yf(r)) for r in cells[k]]) for k in keys if k[0] == 1}
    dv = np.mean(np.concatenate([p[:, 0] - p[:, 1] for p in pairs.values()]))
    Bv = [np.mean(np.concatenate([(lambda q: q[:, 0] - q[:, 1])(p[rng.integers(0, len(p), len(p))]) for p in pairs.values()])) for _ in range(NB)]
    c = ci(Bv); out["V"] = dict(est=dv, ci=c, label=lab(*c)); print(f"  D_V = {dv:+.2f} [{c[0]:+.2f},{c[1]:+.2f}] {lab(*c)}")
    inter = {}
    for (a, pa), (b, pb) in ((("ID", 0), ("N", 1)), (("ID", 0), ("S", 2)), (("N", 1), ("S", 2))):
        g = lambda va, vb: np.concatenate([Ys[k] for k in keys if k[pa] == va and k[pb] == vb]).mean()
        inter[f"{a}x{b}"] = (g(1, 1) - g(0, 1)) - (g(1, 0) - g(0, 0))
    out["interactions_descriptive"] = inter; print("  interactions:", {k: round(v, 2) for k, v in inter.items()})
    traj = {}
    for k in keys:
        rs = cells[k]; ep_keys = sorted(rs[0]["traj"], key=int); E = rs[0]["epochs"]
        traj["ID%dN%dS%d" % k] = {f"{int(e)/E:.2f}": dict(collections.Counter(CLS[collections.Counter(r["traj"][e]["tops"]).most_common(1)[0][0]] for r in rs)) for e in ep_keys}
    out["trajectory"] = traj
    print("  trajectory (fraction → targets):")
    for k, v in traj.items(): print("   ", k, "  ".join(f"{f}:{dict(c)}" for f, c in v.items()))
    return out

def e2():
    d = json.load(open(RES / "ext_e2.json")); Y = lambda r: int(r["final"]["-8.0"]["top_mode"] == 1)
    P = lambda arm, ms: np.mean([Y(r) for r in d if r["cell"] == arm and r["model"] in ms])
    prim = ("wdcnn", "drsn"); out = {"n": len(d), "table": {}}
    for arm in ("S1L3", "S3L1", "S1L1"):
        for m in ("wdcnn", "drsn", "cnn1d"):
            rs = [r for r in d if r["cell"] == arm and r["model"] == m]
            out["table"][f"{arm}|{m}"] = dict(targets=dict(collections.Counter(CLS[r["final"]["-8.0"]["top_mode"]] for r in rs)),
                                               conc=[round(r["final"]["-8.0"]["conc_mean"], 2) for r in rs])
            print(f"  {arm} {m:6s} {out['table'][f'{arm}|{m}']['targets']}  conc {out['table'][f'{arm}|{m}']['conc']}")
    pre = P("S1L3", prim); out["precondition_S1L3_IR"] = pre
    grp = {(arm, m): np.array([Y(r) for r in d if r["cell"] == arm and r["model"] == m]) for arm in ("S1L3", "S3L1") for m in prim}
    est = np.mean(np.concatenate([grp[("S1L3", m)] for m in prim])) - np.mean(np.concatenate([grp[("S3L1", m)] for m in prim]))
    rng = np.random.default_rng(0); B = []
    for _ in range(NB):
        g = {k: rng.choice(v, len(v), replace=True) for k, v in grp.items()}
        B.append(np.mean(np.concatenate([g[("S1L3", m)] for m in prim])) - np.mean(np.concatenate([g[("S3L1", m)] for m in prim])))
    c = ci(B)
    if pre < 0.8: t = "BUDGET_DEPENDENT"
    elif c[0] > D: t = "SIZE_NOT_FILECOUNT"
    elif -D < c[0] and c[1] < D: t = "FILECOUNT_OR_LOAD"
    else: t = "INDETERMINATE"
    out.update(D=est, ci=c, label=t); print(f"  precondition P(IR|S1L3)={pre:.2f}; D={est:+.2f} [{c[0]:+.2f},{c[1]:+.2f}] → {t}")
    return out

def e3():
    d = json.load(open(RES / "arm2x2_lt.json")); A = ["C007_R11", "C007_R31", "CALL_R11", "CALL_R31"]; out = {"n": len(d), "per_arch": {}}
    for m in sorted({r["model"] for r in d}):
        rs = [r for r in d if r["model"] == m]; Y = {a: np.array([int(r["observed"]["-8.0"]["top_mode"] == 1) for r in rs if r["arm"] == a]) for a in A}
        if any(len(v) == 0 for v in Y.values()): print(f"  {m}: incomplete {[(a, len(v)) for a, v in Y.items()]}"); continue
        eff = lambda Y: (0.5 * ((Y["C007_R11"].mean() - Y["CALL_R11"].mean()) + (Y["C007_R31"].mean() - Y["CALL_R31"].mean())),
                         0.5 * ((Y["C007_R31"].mean() - Y["C007_R11"].mean()) + (Y["CALL_R31"].mean() - Y["CALL_R11"].mean())))
        est = eff(Y); rng = np.random.default_rng(0)
        B = np.array([eff({a: rng.choice(v, len(v), replace=True) for a, v in Y.items()}) for _ in range(NB)])
        cC, cR = ci(B[:, 0]), ci(B[:, 1])
        tab = {a: dict(collections.Counter(CLS[r["observed"]["-8.0"]["top_mode"]] for r in rs if r["arm"] == a)) for a in A}
        conc = {a: [round(r["observed"]["-8.0"]["conc_mean"], 2) for r in rs if r["arm"] == a] for a in A}
        out["per_arch"][m] = dict(table=tab, conc=conc, dC=est[0], ci_dC=cC, dR=est[1], ci_dR=cR,
                                  label_descriptive=("COVERAGE" if cC[0] > D and -D < cR[0] and cR[1] < D else
                                                     "RATIO" if cR[0] > D and -D < cC[0] and cC[1] < D else
                                                     "NULL_MATCHED" if -D < cC[0] and cC[1] < D and -D < cR[0] and cR[1] < D else "OTHER"))
        print(f"  {m}: {tab}\n     conc {conc}\n     ΔC={est[0]:+.2f} {cC}  ΔR={est[1]:+.2f} {cR} → {out['per_arch'][m]['label_descriptive']}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("exps", nargs="+"); a = ap.parse_args()
    allout = json.load(open(RES / "ext_verdict.json")) if (RES / "ext_verdict.json").exists() else {}
    for e in a.exps:
        print(f"===== {e.upper()} ====="); allout[e] = {"e1": e1, "e2": e2, "e3": e3}[e]()
    json.dump(allout, open(RES / "ext_verdict.json", "w"), indent=1, ensure_ascii=False, default=float)
