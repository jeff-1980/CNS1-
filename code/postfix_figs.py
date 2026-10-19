"""IEEE Access 版 Fig. 1、2、5（2026-10-19）：以修正加载器后的实验为主证据（模拟审稿意见 1）。字号比 MSSP 版大一级。
数据：results/cinit_fix_2x2.json；修正后 E5 = cinit_e5 draws 0、2 + cinit_fix_e5 draw 1；E2 未重做（原始运行，图中标注）；
Fig. 5 的 CWRU 来自 ieee_access/analysis/postfix_inference.json（修正后 2×2，B2 重新校准），PU 来自 wp5_main.json 与 wp5b2_summary.json（PU 不含该缺陷）。
由 git_stage/code/mssp_figs.py 改写；所有数值从 json 读取。"""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from figstyle import apply_figure_style, panel_letter, set_frame, META_GREY
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results") + "/"; A = R; OUT = os.path.join(os.path.dirname(R.rstrip("/")), "figures") + "/"; os.makedirs(OUT, exist_ok=True)
apply_figure_style(sizes=(9, 8, 7)); mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]; mpl.rcParams["axes.unicode_minus"] = True
J = lambda f: json.load(open(R + f))
CW = {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}; PU = {0: "Healthy", 1: "OR", 2: "IR"}
COL = {"Normal": "#4a4a4a", "Healthy": "#4a4a4a", "IR": "#1f5fa8", "OR": "#e08a1e", "Ball": "#8e5fb0"}; TXT = {"OR": "black"}
FS = 7.5

def tile(ax, rows, cols, cellfn, rowlab, collab, fs=FS):
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            c = cellfn(row, col)
            if not c: ax.text(j, i, "—", ha="center", va="center", color=META_GREY, fontsize=fs); continue
            n = sum(c.values()); t, k = c.most_common(1)[0]
            ax.add_patch(FancyBboxPatch((j - 0.45, i - 0.41), 0.90, 0.82, boxstyle="round,pad=0,rounding_size=0.06", facecolor=COL[t], edgecolor="white", lw=1.2))
            ax.text(j, i, f"{t} {k}/{n}", ha="center", va="center", color=TXT.get(t, "white"), fontsize=fs, fontweight="bold")
    ax.set_xlim(-0.55, len(cols) - 0.45); ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(collab); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rowlab)
    ax.tick_params(length=0); ax.xaxis.tick_top()
    for s in ax.spines.values(): s.set_visible(False)

def check(fig, path):
    fig.savefig(path, dpi=300)
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    ov = [(a.get_text()[:18], b.get_text()[:18]) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    print(os.path.basename(path), "overlaps:", ov)

A2 = J("cinit_fix_2x2.json"); E2 = J("ext_e2.json"); WP2 = J("wp2_R.json") + J("wp2_A.json") + J("wp2_R_single.json")
E5 = [r for r in J("cinit_e5.json") if r["draw"] in (0, 2)] + J("cinit_fix_e5.json"); assert len(A2) == 80 and len(E5) == 189
M4 = ["wdcnn", "drsn", "lstm", "cnn1d"]; M3 = ["wdcnn", "drsn", "lstm"]

def fig1():
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.1], wspace=0.62, left=0.08, right=0.97, top=0.76, bottom=0.27)
    ax = fig.add_subplot(gs[0]); snr = [10.0, 0.0, -4.0, -6.0, -8.0]
    rs = [r for r in A2 if r["model"] == "wdcnn" and r["coverage"] == "007"]
    for key, col, lab in (("acc_mean", "#1f5fa8", "accuracy"), ("conc_mean", "#b2182b", "concentration")):
        Y = np.array([[r["observed"][f"{s}"][key] for s in snr] for r in rs])
        for y in Y: ax.plot(snr, y, color=col, lw=0.5, alpha=0.25)
        ax.plot(snr, Y.mean(0), color=col, lw=1.8, marker="o", ms=3)
        ax.text(-8.8, Y.mean(0)[-1], lab, color=col, fontsize=FS, ha="left", va="center")
    ax.set_xlim(11, -19.5); ax.set_ylim(0, 1.05); ax.set_xticks([10, 0, -4, -8]); ax.set_xlabel("test SNR (dB)"); ax.set_ylabel("fraction of test windows"); set_frame(ax)
    ax.set_title("Strong noise: predictions\ncollapse onto one class", loc="left", fontsize=9); panel_letter(ax, "a")
    axb = fig.add_subplot(gs[1]); mc = lambda rs, f: collections.Counter(f(r) for r in rs) if rs else None
    rows = [("CWRU · 0.007″ faults only", lambda m: mc([r for r in A2 if r["model"] == m and r["coverage"] == "007"], lambda r: CW[r["observed"]["-8.0"]["top_mode"]])),
            ("CWRU · all three fault sizes", lambda m: mc([r for r in A2 if r["model"] == m and r["coverage"] == "ALL"], lambda r: CW[r["observed"]["-8.0"]["top_mode"]])),
            ("PU · damage extent 1 only", lambda m: mc([r for r in WP2 if r["model"] == m and r["arm"] == "R-L1"], lambda r: PU[r["observed"]["-8.0"]["top_mode"]])),
            ("PU · extent 1 + extent 2", lambda m: mc([r for r in WP2 if r["model"] == m and r["arm"] == "R-L12a"], lambda r: PU[r["observed"]["-8.0"]["top_mode"]]))]
    tile(axb, [r[0] for r in rows], M4, lambda row, m: dict(rows)[row](m), [r[0] for r in rows], M4)
    axb.axvline(2.5, color=META_GREY, lw=0.6, ls=(0, (1, 2)))
    axb.set_title("CWRU (corrected loader): the damage-size composition of the training\nfiles switches the dominant class in wdcnn and drsn. PU: no such shift", loc="left", fontsize=9, pad=20)
    panel_letter(axb, "b")
    fig.text(0.02, 0.012, "a: CWRU, wdcnn trained on 0.007″ faults only, corrected loader, n = 10 = 2 Normal:fault ratios × 5 seeds (thin lines = instances).\n"
             "b: dominant class at −8 dB AWGN; count = instances with that class / instances. CWRU rows: corrected-loader confirmation (C-INIT-FIX), pooled over\n"
             "the 1:1 and 3:1 Normal:fault arms (n = 2 × 5 seeds); PU (not affected by the loading error): real-damage bearings, 1500 rpm, 0.7 Nm, 1000 N (n = 5).", fontsize=6.5, color=META_GREY)
    check(fig, OUT + "fig1_overview.png")

def fig2():
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 2.6], wspace=0.62, left=0.13, right=0.98, top=0.86, bottom=0.61)
    gc = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.3, left=0.20, right=0.98, top=0.45, bottom=0.10)
    ax = fig.add_subplot(gs[0]); arms = ["C007_R11", "C007_R31", "CALL_R11", "CALL_R31"]
    lab = {"C007_R11": "0.007″ only · N 1:1", "C007_R31": "0.007″ only · N 3:1", "CALL_R11": "all sizes · N 1:1", "CALL_R31": "all sizes · N 3:1"}
    tile(ax, arms, M4, lambda a, m: collections.Counter(CW[r["observed"]["-8.0"]["top_mode"]] for r in A2 if r["arm"] == a and r["model"] == m) or None, [lab[a] for a in arms], M4)
    ax.axhline(1.5, color=META_GREY, lw=0.8, ls=(0, (2, 2))); ax.axvline(1.5, color=META_GREY, lw=0.8, ls=(0, (2, 2)))
    ax.set_title("2 × 2 (corrected loader): fault-size composition\nswitches the dominant class in wdcnn and drsn;\nNormal share within the preset no-effect criterion", loc="left", fontsize=9, pad=18)
    panel_letter(ax, "a")
    axb = fig.add_subplot(gs[1]); cells = ["S1L3", "S1L1", "S3L1"]
    lb = {"S1L3": "0.007″ · 3 files", "S1L1": "0.007″ · 1 file", "S3L1": "3 sizes · 1 file each"}
    tile(axb, cells, ["wdcnn", "drsn", "cnn1d"], lambda c, m: collections.Counter(CW[r["final"]["-8.0"]["top_mode"]] for r in E2 if r["cell"] == c and r["model"] == m) or None,
         [lb[c] for c in cells], ["wdcnn", "drsn", "cnn1d"])
    axb.set_title("E2 (original run, not repeated):\nfile count did not drive the switch", loc="left", fontsize=9, pad=18); panel_letter(axb, "b")
    sets = ["S007", "S014", "S021", "S007+014", "S007+021", "S014+021", "S_ALL"]
    sl = {"S007": "0.007″", "S014": "0.014″", "S021": "0.021″", "S007+014": "0.007″ + 0.014″", "S007+021": "0.007″ + 0.021″", "S014+021": "0.014″ + 0.021″", "S_ALL": "all three sizes"}
    axc = fig.add_subplot(gc[0])
    tile(axc, sets, M3, lambda st, m: collections.Counter(CW[r["observed"]["-8.0"]["top_mode"]] for r in E5 if r["set"] == st and r["model"] == m) or None, [sl[x] for x in sets], M3)
    axc.axhline(2.5, color=META_GREY, lw=0.8, ls=(0, (2, 2)))
    axc.set_title("E5 (corrected): single sizes 0.007″ → IR; 0.014″ or 0.021″ leaves IR;\nmixed sets (below the line) are not stable between runs", loc="left", fontsize=9, pad=18)
    panel_letter(axc, "c")
    axd = fig.add_subplot(gc[1]); axd.axis("off"); same = []
    for st in sets:
        for m in M3:
            ks = [collections.Counter(CW[r["observed"]["-8.0"]["top_mode"]] for r in E5 if r["set"] == st and r["model"] == m and r["draw"] == dr).most_common(1)[0][0] for dr in (0, 1, 2)]
            same.append(len(set(ks)) == 1)
    axd.text(0.0, 0.95, f"Draw-level majority\nidentical in all 3 file\ndraws: {sum(same)}/{len(same)} cells\n(a draw-level count,\nnot an instance count)", fontsize=8, va="top")
    fig.text(0.02, 0.012, "CWRU drive end; dominant class at −8 dB on Test_40 (file-disjoint). a: corrected loader (C-INIT-FIX), 1900 windows, n = 5 seeds per cell; N 1:1 = 475 windows per class,\n"
             "N 3:1 = 950 Normal and ≈316 per fault class; inverse-frequency class weights. b: original loader, 920 windows, 5 seeds. c: corrected E5 = fixed-initialization draws 0 and 2\n"
             "(these do not use file 99) + corrected-loader draw 1; n = 3 file draws × 3 seeds per cell, one recording per class and size per draw. All: 3000 steps, last weights.", fontsize=6.5, color=META_GREY)
    check(fig, OUT + "fig2_coverage.png")

def fig5():
    P = json.load(open(A + "postfix_inference.json"))["B"]; W5 = [r for r in J("wp5_main.json") if r["factor"] == "PU"]; S5b = J("wp5b2_summary.json")["cells"]
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(1, 3, wspace=0.55, left=0.09, right=0.97, top=0.92, bottom=0.66)
    gf = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.45, left=0.09, right=0.97, top=0.47, bottom=0.17)
    snr = ["10.0", "4.0", "0.0", "-4.0", "-6.0", "-8.0"]; xs = [float(s) for s in snr]
    cw = [r for r in P if r["model"] in M3]; pu = [r for r in W5 if r["model"] in M3]
    mcw = lambda b, k, s: np.nanmean([r["by_snr"][s][b][k] for r in cw if r["by_snr"][s][b][k] is not None])
    mpu = lambda b, k, s: np.nanmean([r["by_snr"][s][b][k] for r in pu])
    for j, (f, name, L) in enumerate(((mcw, "CWRU", "a"), (mpu, "PU", "b"))):
        ax = fig.add_subplot(gs[j])
        for k, col, lb in (("FAR", "#e08a1e", "false alarm\n(Healthy → fault)"), ("MFR", "#1f5fa8", "missed fault\n(fault → Healthy)")):
            y = [f("B0", k, s) for s in snr]; ax.plot(xs, y, color=col, lw=1.6, marker="o", ms=2.5); ax.text(-8.6, max(y[-1], 0.10), lb, color=col, fontsize=7, va="center", ha="left")
        ax.set_xlim(11, -24); ax.set_xticks([10, 0, -8]); ax.set_ylim(-0.03, 1.05); ax.set_xlabel("test SNR (dB)")
        if j == 0: ax.set_ylabel("error rate (no rejection)")
        set_frame(ax); panel_letter(ax, L); ax.set_title(f"{name}: collapse appears as\n{'false alarms' if name == 'CWRU' else 'missed faults'}", loc="left", fontsize=9)
    ax = fig.add_subplot(gs[2])
    for y, col, lb in (([mcw("B2", "COVERAGE", s) for s in snr], "#1f5fa8", "CWRU"), ([1 - mpu("B2", "REJ", s) for s in snr], "#6a6a6a", "PU")):
        ax.plot(xs, y, color=col, lw=1.6, marker="o", ms=2.5); ax.text(-8.6, y[-1] + (0.05 if lb == "CWRU" else -0.05), lb, color=col, fontsize=7, va="center", ha="left")
    ax.set_xlim(11, -16); ax.set_xticks([10, 0, -8]); ax.set_ylim(-0.03, 1.05); ax.set_xlabel("test SNR (dB)"); ax.set_ylabel("share of windows accepted (B2)"); set_frame(ax)
    ax.set_title("The confidence rule (B2)\naccepts most noisy windows", loc="left", fontsize=9); panel_letter(ax, "c")
    ax = fig.add_subplot(gf[0]); rng = np.random.default_rng(2)
    for x, mm in enumerate(M3):
        for k, (rs, colr, off, key) in enumerate(((cw, "#1f5fa8", -0.18, "SILENT"), (pu, "#6a6a6a", 0.18, "SILENT"))):
            v = np.array([r["by_snr"]["-8.0"]["B2"][key] for r in rs if r["model"] == mm and r["by_snr"]["-8.0"]["B2"][key] is not None], float); v = v[~np.isnan(v)]
            ax.scatter(x + off + rng.uniform(-0.1, 0.1, len(v)), v, s=8, color=colr, alpha=0.6, lw=0)
            if len(v): ax.plot([x + off - 0.12, x + off + 0.12], [np.median(v)] * 2, color="black", lw=1.4)
    ax.set_xticks(range(3)); ax.set_xticklabels(M3); ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("errors accepted by B2 (−8 dB)"); set_frame(ax)
    ax.text(0.02, 0.06, "blue = CWRU, grey = PU", transform=ax.transAxes, fontsize=7, color=META_GREY)
    ax.set_title("Most errors pass the\nconfidence rule", loc="left", fontsize=9); panel_letter(ax, "d")
    RCOL = {"wdcnn": "#1f5fa8", "drsn": "#6f9fd8", "lstm": "#9a9a9a"}
    for j, (name, L) in enumerate((("CWRU", "e"), ("PU", "f"))):
        ax = fig.add_subplot(gf[j + 1])
        for mm in M3:
            if name == "CWRU":
                rs = [r for r in cw if r["model"] == mm]; cov = rs[0]["by_snr"]["-8.0"]["rc_cov"]; rc = np.nanmean([r["by_snr"]["-8.0"]["rc"] for r in rs], 0)
                na = sum(r["by_snr"]["-8.0"]["B2"]["N_ACCEPTED"] for r in rs); ne = sum(r["by_snr"]["-8.0"]["B2"]["N_ERR_ACCEPTED"] for r in rs)
                pt = (np.mean([r["by_snr"]["-8.0"]["B2"]["COVERAGE"] for r in rs]), ne / na if na else None)
            else:
                c = S5b[f"PU_all|{mm}"]["-8.0"]; cov = np.arange(0.05, 1.0001, 0.05); rc = c["rc"]; pt = (c["coverage"], c["risk_pooled"])
            ax.plot(cov, rc, color=RCOL[mm], lw=1.4, label=mm)
            if pt[1] is not None and pt[0] >= 0.05: ax.scatter([pt[0]], [pt[1]], s=18, color=RCOL[mm], edgecolor="black", lw=0.5, zorder=4)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.0); ax.set_xlabel("coverage")
        if j == 0: ax.set_ylabel("error rate among\naccepted windows"); ax.legend(frameon=False, fontsize=7, loc="lower right")
        set_frame(ax); panel_letter(ax, L); ax.set_title(f"{name}, −8 dB:\nrisk–coverage", loc="left", fontsize=9)
    fig.text(0.02, 0.012, "Inference on existing weights, clean-trained wdcnn, drsn, lstm, −8 dB unless stated. CWRU: corrected-loader 2 × 2 models (C-INIT-FIX, n = 20 per architecture),\n"
             "B2 threshold recalibrated on clean validation windows (5th percentile of maximum softmax probability; the validation split was not used to train these models).\n"
             "PU: WP2 instances (not affected by the loading error), calibration on held-out time segments. Curves average per-instance risk with ties in confidence resolved\n"
             "by expected risk; filled circles = rule B2 at its actual coverage and pooled risk.", fontsize=6.5, color=META_GREY)
    check(fig, OUT + "fig5_consequence.png")

if __name__ == "__main__":
    import shutil
    for f in ("fig1_overview.png", "fig2_coverage.png", "fig5_consequence.png"):   # keep the original-run figures for the supplement
        src = OUT + f; dst = OUT + f.replace(".png", "_originalrun.png")
        if os.path.exists(src) and not os.path.exists(dst): shutil.copy(src, dst)
    fig1(); fig2(); fig5()
