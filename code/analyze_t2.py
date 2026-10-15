"""按 PREREG_T2_domain_strength.md（已冻结）§5–§7 判定 T2。运行时校验预注册哈希。
输入 results/t2_cwru.json、results/t2_ecg.json、results/t2_calibration.json；输出 results/t2_verdict.json。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, hashlib, sys, collections, itertools, pathlib, argparse
import numpy as np
R = pathlib.Path(_ROOT) / "results"
if hashlib.sha256((R / "PREREG_T2_domain_strength.md").read_bytes()).hexdigest() != (R / "registry/prereg_t2_sha256.txt").read_text().split()[0]:
    sys.exit("预注册哈希不符，拒绝判定")
ARCHS = ["wdcnn", "drsn", "lstm"]; DOMS = ["cwru", "ecg"]; WEAK = {"steps": "weak_steps", "label": "weak_label"}
DELTA, NBOOT, EPS = 0.15, 10000, 1e-12


def agree_matrix(rows):
    P = np.array([r["probe"]["preds"] for r in sorted(rows, key=lambda r: r["seed"])])
    n = len(P); M = np.ones((n, n))
    for i, j in itertools.combinations(range(n), 2): M[i, j] = M[j, i] = float(np.mean(P[i] == P[j]))
    return M


def A_of(M, idx):
    """两两一致率均值（idx 可含重复：重复的同一模型之间一致率为 1）。"""
    v = [M[a, b] for a, b in itertools.combinations(idx, 2)]; return float(np.mean(v))


def boot(M, rng, keep_self=False):
    """有放回重抽 10 个种子后重算两两一致率。默认排除重抽中"同一模型与自身"配对（否则一致率恒为 1，使不确定格的自助分布系统性偏高）；
    keep_self=True 为字面口径，只作敏感性分析（见 DEVIATIONS_T2.md）。"""
    n = len(M); iu = np.triu_indices(n, 1); out = np.empty(NBOOT)
    for k in range(NBOOT):
        idx = rng.integers(0, n, n); S = M[np.ix_(idx, idx)]; same = (idx[:, None] == idx[None, :])[iu]
        v = S[iu] if keep_self else S[iu][~same]; out[k] = v.mean()
    return out


def label(dS, dD):
    lo = lambda x: np.percentile(x, 2.5); hi = lambda x: np.percentile(x, 97.5)
    ciS, ciD = (lo(dS), hi(dS)), (lo(dD), hi(dD)); inside = lambda c: c[0] > -DELTA and c[1] < DELTA
    if ciS[1] < -DELTA or ciD[1] < -DELTA: lab = "REVERSED"
    elif ciS[0] > DELTA and inside(ciD): lab = "STRENGTH"
    elif ciD[0] > DELTA and inside(ciS): lab = "DATA"
    elif ciS[0] > DELTA and ciD[0] > DELTA: lab = "BOTH"
    elif inside(ciS) and inside(ciD): lab = "NEITHER"
    else: lab = "INDETERMINATE"
    return lab, [round(float(x), 3) for x in ciS], [round(float(x), 3) for x in ciD]


CONSEQ = {"STRENGTH": "吸引子是否确定主要取决于学习强度 → C6′ 值得在强档设置下重新预注册检验（仍是水平增长，天花板不变）",
          "DATA": "主要取决于数据结构 → “覆盖决定吸引子”是 CWRU 类数据的性质；C6 跨域线关闭；CNS1# 稿件写明“结论限于噪声与某类判别特征对齐的数据”，T1 的高频结果作为支持该边界的探索性证据",
          "BOTH": "两者都有贡献 → 写入 CNS1# 讨论；C6′ 不单独重启"}
OTHER = "不作机制推论；T2 只作为 CNS1# 的局限说明"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in-dir", default=str(R)); ap.add_argument("--out-dir", default=str(R)); a = ap.parse_args()
    D = {d: json.load(open(pathlib.Path(a.in_dir) / f"t2_{d}.json")) for d in DOMS}
    cal = json.load(open(pathlib.Path(a.in_dir) / "t2_calibration.json"))
    cell = {(d, t, m): [r for r in D[d] if r["tier"] == t and r["model"] == m] for d in DOMS for t in ("strong", "weak_steps", "weak_label") for m in ARCHS}
    missing = {f"{k[0]}|{k[1]}|{k[2]}": 10 - len(v) for k, v in cell.items() if len(v) != 10}
    M = {k: agree_matrix(v) for k, v in cell.items()}; A = {k: A_of(M[k], range(len(M[k]))) for k in M}
    rng = np.random.default_rng(0); B = {k: boot(M[k], rng) for k in sorted(M)}
    rng2 = np.random.default_rng(0); B_lit = {k: boot(M[k], rng2, keep_self=True) for k in sorted(M)}
    sens = {}
    for m in ARCHS:
        for w, t in WEAK.items():
            dS = 0.5 * ((B_lit[("cwru", "strong", m)] - B_lit[("cwru", t, m)]) + (B_lit[("ecg", "strong", m)] - B_lit[("ecg", t, m)]))
            dD = 0.5 * ((B_lit[("cwru", "strong", m)] - B_lit[("ecg", "strong", m)]) + (B_lit[("cwru", t, m)] - B_lit[("ecg", t, m)]))
            sens[f"{m}|{w}"] = label(dS, dD)[0]
    per_arch = {}
    for m in ARCHS:
        labs = {}
        for w, t in WEAK.items():
            dS = 0.5 * ((B[("cwru", "strong", m)] - B[("cwru", t, m)]) + (B[("ecg", "strong", m)] - B[("ecg", t, m)]))
            dD = 0.5 * ((B[("cwru", "strong", m)] - B[("ecg", "strong", m)]) + (B[("cwru", t, m)] - B[("ecg", t, m)]))
            pS = 0.5 * ((A[("cwru", "strong", m)] - A[("cwru", t, m)]) + (A[("ecg", "strong", m)] - A[("ecg", t, m)]))
            pD = 0.5 * ((A[("cwru", "strong", m)] - A[("ecg", "strong", m)]) + (A[("cwru", t, m)] - A[("ecg", t, m)]))
            lab, ciS, ciD = label(dS, dD); labs[w] = dict(label=lab, dS=round(pS, 3), ciS=ciS, dD=round(pD, 3), ciD=ciD)
        arch = labs["steps"]["label"] if labs["steps"]["label"] == labs["label"]["label"] else "INDETERMINATE"
        per_arch[m] = dict(by_weakening=labs, label=arch)
    c = collections.Counter(v["label"] for v in per_arch.values()); top, n = c.most_common(1)[0]
    verdict = top if n >= 2 else "ARCH_DEPENDENT"
    # 有效性标注
    flags = {}
    flags["T1_NOT_REPLICATED"] = sum(A[("cwru", "strong", m)] < 0.90 for m in ARCHS) >= 2
    ch = {k: v["chosen"] for k, v in cal.items()}
    flags["STRENGTH_GAP"] = {k: v["strength_gap"] for k, v in ch.items() if v["strength_gap"]} or False
    flags["WEAK_GAP"] = {k: [w for w, ok in (("steps", v["weak_steps_in_range"]), ("label", v["rho_in_range"])) if not ok] for k, v in ch.items()
                         if not (v["weak_steps_in_range"] and v["rho_in_range"])} or False
    sec = {}
    for k, rows in cell.items():
        tops = [r["observed"]["-8.0"]["top_mode"] for r in rows]; modes = [r["probe"]["mode"] for r in rows]
        sec["|".join(k)] = dict(A=round(A[k], 3), mode_share_m8=max(collections.Counter(tops).values()) / len(tops),
                                dominant_m8=dict(collections.Counter(tops)), collapse_frac_060=float(np.mean([r["observed"]["-8.0"]["conc_mean"] >= 0.60 for r in rows])),
                                noise_margin_median=float(np.median([r["probe"]["margin_median"] for r in rows])),
                                base_share_noise=float(np.mean([x == 0 for x in modes])), base_share_m8=float(np.mean([x == 0 for x in tops])),
                                low_same_as_white=float(np.mean([r["color"]["low"] == r["color"]["white"] for r in rows])),
                                high_same_as_white=float(np.mean([r["color"]["high"] == r["color"]["white"] for r in rows])),
                                clean_acc=float(np.mean([r["clean_acc"] for r in rows])), acc_p10=float(np.mean([r["observed"]["10.0"]["acc_mean"] for r in rows])))
    flags["NO_COLLAPSE_CELL"] = [k for k, v in sec.items() if v["collapse_frac_060"] < 0.6] or False
    flags["NORM_DOMINATED"] = [k for k, v in sec.items() if v["base_share_noise"] >= 0.8] or False
    out = dict(prereg_sha256=(R / "registry/prereg_t2_sha256.txt").read_text().split()[0], missing=missing, per_arch=per_arch, verdict=verdict,
               consequence=CONSEQ.get(verdict, OTHER), flags=flags, cells=sec, calibration_chosen=ch,
               sensitivity_bootstrap_keep_self_pairs=sens)
    json.dump(out, open(pathlib.Path(a.out_dir) / "t2_verdict.json", "w"), ensure_ascii=False, indent=1)
    for m, v in per_arch.items():
        b = v["by_weakening"]
        print(f"{m:6s} 步数弱化 {b['steps']['label']:13s} ΔS={b['steps']['dS']:+.3f}{b['steps']['ciS']} ΔD={b['steps']['dD']:+.3f}{b['steps']['ciD']}  | "
              f"标签噪声弱化 {b['label']['label']:13s} ΔS={b['label']['dS']:+.3f}{b['label']['ciS']} ΔD={b['label']['dD']:+.3f}{b['label']['ciD']}  → {v['label']}")
    print("跨架构判定：", verdict, "  缺失：", missing or "无"); print("有效性标注：", flags); print("§7：", out["consequence"])
    print("敏感性（字面自助口径，含自身配对）：", sens)
    for k, v in sec.items():
        print(f"  {k:22s} A={v['A']:.3f} −8dB众数占比={v['mode_share_m8']:.1f} 崩溃比例={v['collapse_frac_060']:.1f} 干净={v['clean_acc']:.3f} "
              f"基准类(噪声/−8dB)={v['base_share_noise']:.1f}/{v['base_share_m8']:.1f} 低通/高通同白={v['low_same_as_white']:.1f}/{v['high_same_as_white']:.1f}")


if __name__ == "__main__":
    main()
