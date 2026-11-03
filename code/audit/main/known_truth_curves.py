"""Known-truth test: turn the per-run records (~/audit_ext/known_truth/records/<obj>/<group>/seed*.json) into
results/known_truth/curves/<obj>.json, the input of known_truth.py: [{group, seed, acc_curve_S, acc_curve_T, ...}].
Also compares the full-test curve with the main analysis curve of the same group and seed (identical only where
training is seeded and the data are the same as in the main runs).
usage (in code/): python -m audit.main.known_truth_curves --records ~/audit_ext/known_truth/records --protocols ../results/main_objects/protocols --out ../results/known_truth/curves"""
import argparse, glob, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True); ap.add_argument("--protocols", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); summary = {}
    for od in sorted(glob.glob(os.path.join(os.path.expanduser(a.records), "*"))):
        obj = os.path.basename(od)
        main_curves = {(r["group"], r["seed"]): r["acc_curve"] for r in json.load(open(os.path.join(a.protocols, f"{obj}.json")))}
        rows, same = [], []
        for f in sorted(glob.glob(os.path.join(od, "*", "seed*.json"))):
            d = json.load(open(f)); ep = sorted(d["epochs"], key=int)
            if not all({"S", "T", "test"} <= set(d["epochs"][e]) for e in ep):
                raise SystemExit(f"{f}: an epoch lacks S/T/test")
            row = dict(group=d["group"], seed=d["seed"], n_epochs=len(ep),
                       acc_curve_S=[d["epochs"][e]["S"]["acc"] for e in ep], acc_curve_T=[d["epochs"][e]["T"]["acc"] for e in ep],
                       acc_curve_test=[d["epochs"][e]["test"]["acc"] for e in ep],
                       author_check=d.get("author_check") or d.get("author_check_f1"), finished=bool(d.get("finished_at_utc")))
            m = main_curves.get((d["group"], d["seed"]))
            if m is not None and len(m) == len(ep):
                same.append(bool(np.allclose(np.array(m, float), row["acc_curve_test"], atol=1e-9)))
            rows.append(row)
        json.dump(rows, open(os.path.join(a.out, f"{obj}.json"), "w"), indent=1)
        summary[obj] = dict(runs=len(rows), finished=sum(r["finished"] for r in rows),
                            test_curve_identical_to_main=f"{sum(same)}/{len(same)}")
        print(obj, summary[obj])
    # next to curves/, not inside it: known_truth.py reads every *.json in curves/
    json.dump(summary, open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "curves_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
