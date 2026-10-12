"""按 PREREG_C6_crossdomain_pilot.md（已冻结）§5–§7 判定 C6-P。运行时校验预注册哈希。
输入 results/c6p_d1.json、results/c6p_d2.json；输出 results/c6p_verdict.json 与 results/VERDICT_C6P.md 的表格部分（--md）。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, hashlib, sys, collections, pathlib, argparse
import numpy as np
R = pathlib.Path(_ROOT) / "results"
if hashlib.sha256((R / "PREREG_C6_crossdomain_pilot.md").read_bytes()).hexdigest() != (R / "registry/prereg_c6p_sha256.txt").read_text().split()[0]:
    sys.exit("预注册哈希不符，拒绝判定")
ARMS = ["S_a", "S_b", "S_ALL"]; SEEDS = [0, 1, 2, 3, 4]; EPS = 1e-9
MODELS = {"d1": ["wdcnn", "drsn", "lstm"], "d2": ["cnn4", "resnet18", "vit_tiny"]}
DOMAIN_NAME = {"d1": "D1 心电（PTB-XL）", "d2": "D2 自然图像（CIFAR-100）"}


def dom_class(r, snr): return r["observed"][snr]["top_mode"]


def dist(rs, snr):
    c = collections.Counter(dom_class(r, snr) for r in rs); return np.array([c[k] / len(rs) for k in range(4)])


def tv(p, q): return 0.5 * float(np.abs(p - q).sum())


def consistency(rs, snr): return max(collections.Counter(dom_class(r, snr) for r in rs).values()) / len(rs)


def label(cell, snr="-8.0", conc_thr=0.60):
    """§5，按顺序取第一个成立的标签。cell: {arm: [5 条记录]}。conc_thr 用于 §7.2 的 ≥ 0.90 口径。"""
    allr = [r for a in ARMS for r in cell[a]]; info = dict(snr=snr, conc_thr=conc_thr)
    clean = float(np.mean([r["clean_acc"] for r in cell["S_ALL"]])); info["S_ALL_clean_acc_mean"] = clean
    if clean < 0.40 - EPS: return "NOT_LEARNED", info
    frac = np.mean([r["observed"][snr]["conc_mean"] >= conc_thr for r in allr]); info["collapse_frac"] = float(frac)
    if frac < 0.60 - EPS:
        frac12 = np.mean([r["observed"]["-12.0"]["conc_mean"] >= conc_thr for r in allr]); info["collapse_frac_-12"] = float(frac12)
        if frac12 < 0.60 - EPS: return "NO_COLLAPSE", info
        snr = "-12.0"; info["snr"] = snr; info["fallback_-12dB"] = True              # 后续判定全部在 −12 dB 上
    q = {a: dist(cell[a], snr) for a in ARMS}; cons = {a: consistency(cell[a], snr) for a in ARMS}
    T = {"a_ALL": tv(q["S_a"], q["S_ALL"]), "b_ALL": tv(q["S_b"], q["S_ALL"]), "a_b": tv(q["S_a"], q["S_b"])}
    info.update(q={a: q[a].round(3).tolist() for a in ARMS}, consistency=cons, TV={k: round(v, 3) for k, v in T.items()})
    note = "+SUBTYPE_SPECIFIC" if T["a_b"] >= 0.6 - EPS else ""
    if sum(cons[a] < 0.8 - EPS for a in ARMS) >= 2: lab = "UNSTABLE"
    elif max(T["a_ALL"], T["b_ALL"]) >= 0.6 - EPS: lab = "COVERAGE_SHIFT"
    elif all(v <= 0.2 + EPS for v in T.values()): lab = "COVERAGE_INVARIANT"
    else: lab = "MIXED"
    info["note"] = note
    return lab, info


def domain_verdict(labels):
    c = collections.Counter(labels.values())
    if c["COVERAGE_SHIFT"] >= 2: return "SUPPORTS"
    if c["COVERAGE_INVARIANT"] >= 2: return "CONTRADICTS"
    if c["NOT_LEARNED"] + c["NO_COLLAPSE"] >= 2: return "UNINFORMATIVE"
    return "INCONCLUSIVE"


def consequence(v1, v2):
    """§6。"""
    vs = {v1, v2}
    if v1 == v2 == "SUPPORTS": return "BOTH_SUPPORT", "C6′ 值得正式投入：写全规模确认预注册（更多子类型组合、加回比例因子、C-INIT 式复跑、第三领域如音频）；须先对照 §10 价值上限"
    if "SUPPORTS" in vs and "CONTRADICTS" not in vs: return "ONE_SUPPORTS", "加第三个领域（音频，如 Speech Commands 或 ESC-50）再决定，不据此改变投稿定位"
    if "CONTRADICTS" in vs and "SUPPORTS" not in vs: return "CONTRADICTED", "C6′ 不成立：停止跨域线，MSSP 稿件维持“CWRU 内 + 方法学”定位，可把本结果写成局限/讨论"
    if v1 == v2 == "UNINFORMATIVE": return "PILOT_FAILED", "试点设计失败，不能对 C6′ 下结论；修订设计需新的预注册"
    return "NOT_COVERED", "§6 未预先约定此组合（如一个 SUPPORTS、一个 CONTRADICTS，或 INCONCLUSIVE 与非 SUPPORTS 的组合）；不作推论，交 Jeff 决定"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in-dir", default=str(R)); ap.add_argument("--out-dir", default=str(R)); a = ap.parse_args()
    D = {d: json.load(open(pathlib.Path(a.in_dir) / f"c6p_{d}.json")) for d in ("d1", "d2")}
    out = dict(prereg_sha256=(R / "registry/prereg_c6p_sha256.txt").read_text().split()[0], domains={})
    for d, rows in D.items():
        assert len({(r["arm"], r["model"], r["seed"]) for r in rows}) == len(rows)
        cells = {m: {arm: sorted([r for r in rows if r["model"] == m and r["arm"] == arm], key=lambda r: r["seed"]) for arm in ARMS} for m in MODELS[d]}
        missing = {f"{m}|{arm}": 5 - len(cells[m][arm]) for m in MODELS[d] for arm in ARMS if len(cells[m][arm]) != 5}
        names = rows[0]["classes"]
        prim = {m: label(cells[m]) for m in MODELS[d]}
        sec = {"-6dB": {m: label(cells[m], "-6.0")[0] for m in MODELS[d]},
               "conc>=0.90": {m: label(cells[m], conc_thr=0.90)[0] for m in MODELS[d]}}
        # §7.1 纯噪声探针：探针众数类 = −8 dB 主导类别的命中率，对照常数预测器
        snr_used = {m: prim[m][1].get("snr", "-8.0") for m in MODELS[d]}
        t8 = [dom_class(r, "-8.0") for r in rows]; const = collections.Counter(t8).most_common(1)[0][0]   # §7.1 固定用 −8 dB
        probe = dict(hit_rate=float(np.mean([r["probe"]["mode"] == t for r, t in zip(rows, t8)])),
                     constant_class=names[const], constant_hit_rate=float(np.mean([t == const for t in t8])),
                     per_model={m: float(np.mean([r["probe"]["mode"] == dom_class(r, "-8.0") for a_ in ARMS for r in cells[m][a_]])) for m in MODELS[d]})
        tops = [dom_class(r, snr_used[r["model"]]) for r in rows]
        # §7.3 完整分布、干净与 +10 dB 准确率；§7.4 落在基准类的比例
        full = {m: {arm: dict(dominant={names[k]: v for k, v in collections.Counter(dom_class(r, snr_used[m]) for r in cells[m][arm]).items()},
                              clean_acc=float(np.mean([r["clean_acc"] for r in cells[m][arm]])),
                              acc_p10=float(np.mean([r["observed"]["10.0"]["acc_mean"] for r in cells[m][arm]])),
                              conc_mean=float(np.mean([r["observed"][snr_used[m]]["conc_mean"] for r in cells[m][arm]])))
                    for arm in ARMS} for m in MODELS[d]}
        base_frac = float(np.mean([t == 0 for t in tops]))
        labels = {m: prim[m][0] for m in MODELS[d]}
        out["domains"][d] = dict(name=DOMAIN_NAME[d], classes=names, n_instances=len(rows), missing=missing,
                                 labels=labels, label_info={m: prim[m][1] for m in MODELS[d]},
                                 verdict=domain_verdict(labels), secondary_labels=sec, probe=probe, full=full,
                                 base_class_attractor_frac=base_frac)
    v1, v2 = out["domains"]["d1"]["verdict"], out["domains"]["d2"]["verdict"]
    out["consequence"], out["consequence_text"] = consequence(v1, v2)
    json.dump(out, open(pathlib.Path(a.out_dir) / "c6p_verdict.json", "w"), ensure_ascii=False, indent=1)
    for d, o in out["domains"].items():
        print(f"{o['name']}：{o['verdict']}  n={o['n_instances']}  缺失={o['missing'] or '无'}")
        for m in MODELS[d]:
            i = o["label_info"][m]
            print(f"  {m:9s} {o['labels'][m]:19s}{i.get('note','')}  snr={i['snr']}  TV={i.get('TV')}  一致性={i.get('consistency')}  S_ALL干净={i['S_ALL_clean_acc_mean']:.3f}")
        print(f"  次要：−6 dB {o['secondary_labels']['-6dB']}；集中度≥0.90 {o['secondary_labels']['conc>=0.90']}")
        print(f"  探针命中 {o['probe']['hit_rate']:.2f}（常数预测器 {o['probe']['constant_class']} {o['probe']['constant_hit_rate']:.2f}）；吸引子落在基准类 {o['base_class_attractor_frac']:.2f}")
    print(f"§6：{out['consequence']} —— {out['consequence_text']}")


if __name__ == "__main__":
    main()
