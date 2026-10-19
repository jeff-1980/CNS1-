"""按 results/PREREG_CINIT_FIX_loader.md 分析。运行时校验预注册哈希。E5 标签规则取自 analyze_e5.py 的 label()（源码级复用）。"""
import json, hashlib, sys, collections, pathlib, ast
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; R = HERE.parent / "results"; C = HERE.parent / "code"
if hashlib.sha256((R / "PREREG_CINIT_FIX_loader.md").read_bytes()).hexdigest() != (R / "registry/prereg_cinit_fix_sha256.txt").read_text().split()[0]:
    sys.exit("预注册哈希不符，拒绝判定")
CLS = {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}; ARMS = ("C007_R11", "C007_R31", "CALL_R11", "CALL_R31")
f2 = json.load(open(R / "cinit_fix_2x2.json")) if (R / "cinit_fix_2x2.json").exists() else []
f5 = json.load(open(R / "cinit_fix_e5.json")) if (R / "cinit_fix_e5.json").exists() else []
o2 = json.load(open(R / "cinit_2x2.json")); o5 = json.load(open(R / "cinit_e5.json"))
out = dict(n_2x2=len(f2), n_e5d1=len(f5), c2x2={}, e5={})
top = lambda r: r["observed"]["-8.0"]["top_mode"]
# init validity: same init hash as C-INIT for same (model, seed)
oi = {(r["model"], r["seed"]): r["init_sha256"] for r in o2 + o5}
out["init_same_as_cinit"] = f"{sum(oi.get((r['model'], r['seed'])) == r['init_sha256'] for r in f2 + f5)}/{len(f2) + len(f5)}"
share = lambda rs, c: float(np.mean([top(r) == c for r in rs])) if rs else float("nan")
for m in ("wdcnn", "drsn", "lstm", "cnn1d"):
    rs = {a: [r for r in f2 if r["model"] == m and r["arm"] == a] for a in ARMS}
    if any(len(v) < 5 for v in rs.values()): out["c2x2"][m] = dict(label="NOT_COMPLETE", n={a: len(v) for a, v in rs.items()}); continue
    ir = {a: share(v, 1) for a, v in rs.items()}; orr = {a: share(v, 2) for a, v in rs.items()}
    if m == "cnn1d": lab = "REPLICATED" if all(orr[a] >= 0.8 for a in ARMS) else "NOT_REPLICATED"
    else: lab = "REPLICATED" if (ir["C007_R11"] >= 0.8 and ir["C007_R31"] >= 0.8 and orr["CALL_R11"] >= 0.8 and orr["CALL_R31"] >= 0.8) else "NOT_REPLICATED"
    ratio = "RATIO_NULL" if (abs(ir["C007_R11"] - ir["C007_R31"]) < 0.4 and abs(ir["CALL_R11"] - ir["CALL_R31"]) < 0.4) else "RATIO_EFFECT"
    od = {(r["arm"], r["seed"]): r for r in o2 if r["model"] == m}
    agree = sum(top(r) == top(od[(r["arm"], r["seed"])]) for v in rs.values() for r in v)
    ev = sum(r["observed"]["-8.0"]["conc_mean"] >= 0.9 for v in rs.values() for r in v); ev_o = sum(r["observed"]["-8.0"]["conc_mean"] >= 0.9 for r in od.values())
    acc = lambda d: float(np.mean([r["observed"]["10.0"]["acc_mean"] for r in d]))
    out["c2x2"][m] = dict(label=lab, ratio=ratio, dist={a: dict(collections.Counter(CLS[top(r)] for r in v)) for a, v in rs.items()},
                          per_instance_agree_with_cinit=f"{agree}/20", events=f"{ev}/20", events_cinit=f"{ev_o}/20",
                          acc10=acc([r for v in rs.values() for r in v]), acc10_cinit=acc(list(od.values())))
p = [out["c2x2"].get(m, {}).get("label") for m in ("wdcnn", "drsn")]
out["primary"] = "NOT_COMPLETE" if "NOT_COMPLETE" in p or None in p else ("LEAK_ROBUST" if all(x == "REPLICATED" for x in p) else "LEAK_SENSITIVE")
# E5: C-INIT draws 0, 2 + FIX draw 1
src = (C / "analyze_e5.py").read_text(); tree = ast.parse(src)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "label")
comb = [r for r in o5 if r["draw"] != 1] + f5
ns = dict(np=np, SETS=["S007", "S014", "S021", "S007+014", "S007+021", "S014+021", "S_ALL"], MULTI=["S007+014", "S007+021", "S014+021"], DRAWS=[0, 1, 2], d=comb)
exec(compile(ast.Module(body=[fn], type_ignores=[]), "analyze_e5.label", "exec"), ns)
for m in ("wdcnn", "drsn", "lstm"):
    n1 = sum(r["model"] == m for r in f5)
    if n1 < 21: out["e5"][m] = dict(label="NOT_COMPLETE", n_draw1=n1); continue
    lab = ns["label"](m)[0]
    def maj(d, s, dr):
        c = collections.Counter(CLS[top(r)] for r in d if r["model"] == m and r["set"] == s and r["draw"] == dr); t, k = c.most_common(1)[0]; return t if k >= 2 else "tie"
    rows = {s: dict(d0=maj(o5, s, 0), d1_fix=maj(f5, s, 1), d1_leaky=maj(o5, s, 1), d2=maj(o5, s, 2)) for s in ns["SETS"]}
    out["e5"][m] = dict(label=lab, label_cinit="ANY_LARGER_SIZE", per_set=rows,
                        draw1_fix_matches_leaky=sum(v["d1_fix"] == v["d1_leaky"] for v in rows.values()),
                        draw1_fix_matches_clean_draws=sum(v["d1_fix"] == v["d0"] == v["d2"] for v in rows.values()))
json.dump(out, open(HERE / "cinit_fix_verdict.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "e5"}, ensure_ascii=False, indent=1)[:3000])
for m, v in out["e5"].items(): print("E5", m, v.get("label"), "| d1_fix==d1_leaky", v.get("draw1_fix_matches_leaky"), "/7 | d1_fix==d0==d2", v.get("draw1_fix_matches_clean_draws"), "/7")
print(">>> PRIMARY:", out["primary"])
