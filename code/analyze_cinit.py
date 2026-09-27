"""按 PREREG_CINIT_confirmation.md 分析。运行时校验预注册哈希。E5 标签规则直接取自 analyze_e5.py 的 label()（源码级复用，不改写）。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, hashlib, sys, collections, pathlib, ast
import numpy as np
R = pathlib.Path((_ROOT + "/results")); C = pathlib.Path(__file__).resolve().parent
if hashlib.sha256((R / "PREREG_CINIT_confirmation.md").read_bytes()).hexdigest() != (R / "registry/prereg_cinit_sha256.txt").read_text().split()[0]:
    sys.exit("预注册哈希不符，拒绝判定")
CLS = {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}
d2 = json.load(open(R / "cinit_2x2.json")) if (R / "cinit_2x2.json").exists() else []
d5 = json.load(open(R / "cinit_e5.json")) if (R / "cinit_e5.json").exists() else []
out = dict(n_2x2=len(d2), n_e5=len(d5), validity={}, c2x2={}, ce5={})
# 有效性
for tag, d in (("C-2x2", d2), ("C-E5", d5)):
    g = collections.defaultdict(set)
    for r in d: g[(r["model"], r["seed"])].add(r["init_sha256"])
    bad = [f"{m}|s{s}" for (m, s), v in g.items() if len(v) != 1]
    out["validity"][tag] = dict(groups=len(g), invalid=bad); print(tag, "init groups", len(g), "INVALID", bad)
inval = {tuple(x.split("|")) for t in out["validity"].values() for x in t["invalid"]}
ok = lambda r: (r["model"], f"s{r['seed']}") not in inval
# C-2×2
share = lambda rs, c: np.mean([r["observed"]["-8.0"]["top_mode"] == c for r in rs]) if rs else np.nan
for m in ("wdcnn", "drsn", "lstm", "cnn1d"):
    rs = {a: [r for r in d2 if r["model"] == m and r["arm"] == a and ok(r)] for a in ("C007_R11", "C007_R31", "CALL_R11", "CALL_R31")}
    if any(len(v) < 5 for v in rs.values()): out["c2x2"][m] = dict(label="NOT_COMPLETE", n={a: len(v) for a, v in rs.items()}); print(m, "NOT_COMPLETE"); continue
    ir = {a: share(v, 1) for a, v in rs.items()}; orr = {a: share(v, 2) for a, v in rs.items()}
    if m == "cnn1d": lab = "REPLICATED" if all(orr[a] >= 0.8 for a in rs) else "NOT_REPLICATED"
    else: lab = "REPLICATED" if (ir["C007_R11"] >= 0.8 and ir["C007_R31"] >= 0.8 and orr["CALL_R11"] >= 0.8 and orr["CALL_R31"] >= 0.8) else "NOT_REPLICATED"
    ratio = "RATIO_NULL_REPLICATED" if (abs(ir["C007_R11"] - ir["C007_R31"]) < 0.4 and abs(ir["CALL_R11"] - ir["CALL_R31"]) < 0.4) else "RATIO_EFFECT"
    dist = {a: dict(collections.Counter(CLS[r["observed"]["-8.0"]["top_mode"]] for r in v)) for a, v in rs.items()}
    out["c2x2"][m] = dict(label=lab, ratio=ratio, dist=dist, n_event090=int(sum(r["observed"]["-8.0"]["conc_mean"] >= 0.9 for v in rs.values() for r in v)))
    print(f"C-2x2 {m:6s} {lab} {ratio} {dist}")
# C-E5：复用 analyze_e5.label
src = (C / "analyze_e5.py").read_text(); tree = ast.parse(src)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "label")
ns = dict(np=np, SETS=["S007", "S014", "S021", "S007+014", "S007+021", "S014+021", "S_ALL"], MULTI=["S007+014", "S007+021", "S014+021"], DRAWS=[0, 1, 2], d=[r for r in d5 if ok(r)])
exec(compile(ast.Module(body=[fn], type_ignores=[]), "analyze_e5.label", "exec"), ns)
old = json.load(open(R / "e5_verdict.json"))["per_arch"]
for m in ("wdcnn", "drsn", "lstm"):
    n_m = sum(r["model"] == m for r in ns["d"])
    if n_m < 63: out["ce5"][m] = dict(label="NOT_COMPLETE", n=n_m); print(m, "E5 NOT_COMPLETE", n_m); continue
    lab, P, pdraw, rng_ = ns["label"](m)
    dist = {s: dict(collections.Counter(CLS[r["observed"]["-8.0"]["top_mode"]] for r in ns["d"] if r["model"] == m and r["set"] == s)) for s in ns["SETS"]}
    same_cells = sum(max(dist[s], key=dist[s].get) == max(old[m]["dominant_dist"][s], key=old[m]["dominant_dist"][s].get) for s in ns["SETS"])
    out["ce5"][m] = dict(label=lab, original_label=old[m]["label"], replicated=lab == old[m]["label"], P_IR=P, dist=dist, original_dist=old[m]["dominant_dist"], cells_same_majority_as_original=f"{same_cells}/7")
    print(f"C-E5 {m:6s} {lab} (orig {old[m]['label']}) cells same majority {same_cells}/7"); [print(f"     {s:9s} new {dist[s]}  orig {old[m]['dominant_dist'][s]}") for s in ns["SETS"]]
prim = [out["c2x2"].get(m, {}).get("label") for m in ("wdcnn", "drsn", "lstm")] + [out["ce5"].get(m, {}).get("replicated") for m in ("wdcnn", "drsn", "lstm")]
out["manuscript_rule"] = "USE_CINIT_AS_PRIMARY" if all(x in ("REPLICATED", True) for x in prim) else ("INCOMPLETE" if any(x in ("NOT_COMPLETE", None) for x in prim) else "NARROW")
print(">>>", out["manuscript_rule"]); json.dump(out, open(R / "cinit_verdict.json", "w"), indent=1, default=float, ensure_ascii=False)
