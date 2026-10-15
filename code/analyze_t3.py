"""按 PREREG_T3_matched_strength.md（已冻结）§3、§5、§7 判定 T3。运行时校验预注册哈希。
输入 results/t3_main.json、results/t3_calibration.json；输出 results/t3_verdict.json。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, hashlib, sys, collections, itertools, pathlib, argparse
import numpy as np
R = pathlib.Path(_ROOT) / "results"
if hashlib.sha256((R / "PREREG_T3_matched_strength.md").read_bytes()).hexdigest() != (R / "registry/prereg_t3_sha256.txt").read_text().split()[0]:
    sys.exit("预注册哈希不符，拒绝判定")
ARCHS = ["wdcnn", "drsn", "lstm"]; DELTA, NBOOT = 0.15, 10000


def agree_matrix(rows):
    P = np.array([r["probe"]["preds"] for r in sorted(rows, key=lambda r: r["seed"])])
    n = len(P); M = np.ones((n, n))
    for i, j in itertools.combinations(range(n), 2): M[i, j] = M[j, i] = float(np.mean(P[i] == P[j]))
    return M


def A_of(M): return float(M[np.triu_indices(len(M), 1)].mean())


def boot(M, rng):
    """§5：有放回重抽 10 个种子，只对来自不同原始模型的配对求一致率。"""
    n = len(M); iu = np.triu_indices(n, 1); out = np.empty(NBOOT)
    for k in range(NBOOT):
        idx = rng.integers(0, n, n); S = M[np.ix_(idx, idx)]; same = (idx[:, None] == idx[None, :])[iu]
        v = S[iu][~same]; out[k] = v.mean() if len(v) else np.nan
    return out


ci = lambda x: (float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5)))
inside = lambda c: c[0] > -DELTA and c[1] < DELTA


def label(dDhi, dSN):
    cD, cS = ci(dDhi), ci(dSN)
    if cD[1] < -DELTA: lab = "REVERSED"
    elif cD[0] > DELTA: lab = "DATA"
    elif inside(cD) and cS[0] > DELTA: lab = "STRENGTH"
    elif inside(cD): lab = "CONVERGE_ONLY"
    else: lab = "INDETERMINATE"
    return lab, [round(v, 3) for v in cD], [round(v, 3) for v in cS]


CONSEQ = {"STRENGTH": "确定性属于学得好的模型，不是 CWRU 特有 → 在 N 上按高档设置重新预注册检验 C6′",
          "DATA": "强度对齐后 CWRU 仍更确定 → “覆盖决定吸引子”限于 CWRU 这类数据；C6 跨域线关闭；CNS1# 写明边界",
          "CONVERGE_ONLY": "高强度下两领域一致，但强度作用未证实 → 写入 CNS1# 讨论；C6′ 可在 N 上重启，先由 Jeff 决定"}
OTHER = "不作机制推论；C6 跨域线关闭，T2 + T3 一起作为 CNS1# 的局限说明"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in-dir", default=str(R)); ap.add_argument("--out-dir", default=str(R)); a = ap.parse_args()
    cal = json.load(open(pathlib.Path(a.in_dir) / "t3_calibration.json")); N = cal["selected"]
    if N is None:
        out = dict(verdict="NOT_FEASIBLE", consequence=OTHER, feasibility=cal["feasibility"])
        json.dump(out, open(pathlib.Path(a.out_dir) / "t3_verdict.json", "w"), ensure_ascii=False, indent=1); print(out); return
    rows = json.load(open(pathlib.Path(a.in_dir) / "t3_main.json")); ch = cal["chosen"]
    keys = [(d, t) for d in ("cwru", N) for t in ("hi", "lo")] + [("ecg", "natural")]
    cell = {(d, t, m): [r for r in rows if r["domain"] == d and r["tier"] == t and r["model"] == m] for d, t in keys for m in ARCHS}
    missing = {"|".join(k): 10 - len(v) for k, v in cell.items() if len(v) != 10}
    M = {k: agree_matrix(v) for k, v in cell.items()}; A = {k: A_of(M[k]) for k in M}
    rng = np.random.default_rng(0); B = {k: boot(M[k], rng) for k in sorted(M)}
    per_arch = {}
    for m in ARCHS:
        dDhi = B[("cwru", "hi", m)] - B[(N, "hi", m)]; dDlo = B[("cwru", "lo", m)] - B[(N, "lo", m)]; dSN = B[(N, "hi", m)] - B[(N, "lo", m)]
        lab, cD, cS = label(dDhi, dSN)
        hi_gap = ch[f"cwru|{m}"]["hi_gap"] or ch[f"{N}|{m}"]["hi_gap"]
        final = "INDETERMINATE" if hi_gap and lab in ("STRENGTH", "CONVERGE_ONLY") else lab
        per_arch[m] = dict(label=final, raw_label=lab, hi_gap=hi_gap,
                           dD_hi=round(A[("cwru", "hi", m)] - A[(N, "hi", m)], 3), ci_dD_hi=cD,
                           dS_N=round(A[(N, "hi", m)] - A[(N, "lo", m)], 3), ci_dS_N=cS,
                           dD_lo=round(A[("cwru", "lo", m)] - A[(N, "lo", m)], 3), ci_dD_lo=[round(v, 3) for v in ci(dDlo)],
                           ecg_minus_cwru_lo=round(A[("ecg", "natural", m)] - A[("cwru", "lo", m)], 3),
                           ecg_minus_N_lo=round(A[("ecg", "natural", m)] - A[(N, "lo", m)], 3))
    c = collections.Counter(v["label"] for v in per_arch.values()); top, n = c.most_common(1)[0]
    verdict = top if n >= 2 else "ARCH_DEPENDENT"
    sec = {}
    for k, rs in cell.items():
        tops = [r["observed"]["-8.0"]["top_mode"] for r in rs]; modes = [r["probe"]["mode"] for r in rs]
        mc, mn = collections.Counter(modes).most_common(1)[0]; maj = rs[0]["train_majority"]
        sec["|".join(k)] = dict(A=round(A[k], 3), mode_share_m8=max(collections.Counter(tops).values()) / len(tops), dominant_m8=dict(collections.Counter(tops)),
                                collapse_frac_060=float(np.mean([r["observed"]["-8.0"]["conc_mean"] >= 0.60 for r in rs])),
                                noise_margin_median=float(np.median([r["probe"]["margin_median"] for r in rs])),
                                base_share_noise=float(np.mean([x == 0 for x in modes])) if k[0] in ("cwru", "ecg") else None,
                                base_share_m8=float(np.mean([x == 0 for x in tops])) if k[0] in ("cwru", "ecg") else None,
                                noise_mode_top=mc, noise_mode_top_n=mn, train_majority=maj,
                                low_same_as_white=float(np.mean([r["color"]["low"] == r["color"]["white"] for r in rs])),
                                high_same_as_white=float(np.mean([r["color"]["high"] == r["color"]["white"] for r in rs])),
                                clean_acc=float(np.mean([r["clean_acc"] for r in rs])), acc_p10=float(np.mean([r["observed"]["10.0"]["acc_mean"] for r in rs])))
    flags = dict(HI_GAP={k: v["hi_median"] for k, v in ch.items() if v["hi_gap"]} or False,
                 LO_GAP={k: v["lo_median"] for k, v in ch.items() if v.get("lo_gap")} or False,
                 T1_NOT_REPLICATED=sum(A[("cwru", "hi", m)] < 0.90 for m in ARCHS) >= 2,
                 NO_COLLAPSE_CELL=[k for k, v in sec.items() if v["collapse_frac_060"] < 0.6] or False,
                 BASE_DOMINATED=[k for k, v in sec.items() if v["noise_mode_top_n"] >= 8 and v["noise_mode_top"] == v["train_majority"]] or False,
                 K_MISMATCH=N == "slc")
    out = dict(prereg_sha256=(R / "registry/prereg_t3_sha256.txt").read_text().split()[0], N=N, missing=missing, per_arch=per_arch, verdict=verdict,
               consequence=CONSEQ.get(verdict, OTHER), flags=flags, cells=sec, calibration=ch, feasibility=cal["feasibility"])
    json.dump(out, open(pathlib.Path(a.out_dir) / "t3_verdict.json", "w"), ensure_ascii=False, indent=1)
    for m, v in per_arch.items():
        print(f"{m:6s} {v['label']:14s}(原 {v['raw_label']}) ΔD_hi={v['dD_hi']:+.3f}{v['ci_dD_hi']} ΔS_N={v['dS_N']:+.3f}{v['ci_dS_N']} "
              f"ΔD_lo={v['dD_lo']:+.3f}{v['ci_dD_lo']} ECG−CWRU低={v['ecg_minus_cwru_lo']:+.3f} ECG−N低={v['ecg_minus_N_lo']:+.3f}")
    print("N =", N, " 跨架构判定：", verdict, "  缺失：", missing or "无"); print("有效性标注：", flags); print("§7：", out["consequence"])
    for k, v in sec.items():
        print(f"  {k:22s} A={v['A']:.3f} 干净={v['clean_acc']:.3f} −8dB众数占比={v['mode_share_m8']:.1f} 崩溃比例={v['collapse_frac_060']:.1f} "
              f"噪声众数类={v['noise_mode_top']}×{v['noise_mode_top_n']} 低通/高通同白={v['low_same_as_white']:.1f}/{v['high_same_as_white']:.1f}")


if __name__ == "__main__":
    main()
