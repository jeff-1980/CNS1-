"""次要结局 S2 早饱和：选模集准确率在前 5% epoch（ceil(0.05·E)，至少 1 轮）内达到 ≥ 0.999 的运行比例。
主口径只含有独立选模集的运行（select_curves/*.json 中 select_acc 非空）；另报用报告集曲线的全部运行作参照。
只计入进口径比较的运行（protocols/<NN>.json 中有的），轮数 E 取自 protocols（稀疏对象按测试点计）。
在 code/ 下运行：python -m audit.main.s2_saturation --objects ../results/main_objects --out ../results/main_analysis/s2.json"""
import argparse, glob, json, math, os


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--objects", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = dict(select=dict(per_object={}), report=dict(per_object={}))
    for f in sorted(glob.glob(os.path.join(a.objects, "select_curves", "*.json"))):
        o = os.path.basename(f)[:-5]
        if not o.isdigit(): continue
        P = {(r["group"], r["seed"]): r for r in json.load(open(os.path.join(a.objects, "protocols", f"{o}.json")))}
        for r in json.load(open(f))["runs"]:
            p = P.get((r["group"], r["seed"]))
            if p is None: continue
            k = max(1, math.ceil(0.05 * p["n_epochs"]))
            for key, curve in (("select", r["select_acc"]), ("report", p["acc_curve"])):
                if not curve: continue
                c = out[key]["per_object"].setdefault(o, [0, 0]); c[0] += max(curve[:k]) >= 0.999; c[1] += 1
    for v in out.values():
        hit = sum(x[0] for x in v["per_object"].values()); n = sum(x[1] for x in v["per_object"].values())
        v.update(runs=n, saturated=hit, share=hit / n, n_objects=len(v["per_object"]))
        print(f"{n} runs in {v['n_objects']} objects: {hit} saturated ({hit / n:.1%})")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
