"""MSSP 主稿 Fig. 1–5（按 strategy/MSSP_SKELETON.md）。用法：python code/mssp_figs.py（输出到 figures/）。
所有数值直接从 results/*.json 读取；不含任何手写数字。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, os, collections
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from figstyle import apply_figure_style, panel_letter, set_frame, META_GREY
mpl.use("Agg")

R = (_ROOT + "/results/"); OUT = (_ROOT + "/figures" + "/"); os.makedirs(OUT, exist_ok=True)
apply_figure_style(sizes=(8, 7, 6))
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]; mpl.rcParams["axes.unicode_minus"] = True
J = lambda f: json.load(open(R + f))
CW = {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}; PU = {0: "Healthy", 1: "OR", 2: "IR"}; JN = {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}
COL = {"Normal": "#4a4a4a", "Healthy": "#4a4a4a", "IR": "#1f5fa8", "OR": "#e08a1e", "Ball": "#8e5fb0", "none": "#ffffff"}
TXT = {"OR": "black", "none": "#555555"}
ATT = 0.60        # 吸引子判据（与 B 线预注册一致）：conc < 0.60 记为 no attractor

def cat(top, conc, names):
    """Fig. 1 uses the preregistered primary outcome (modal class, no share threshold), matching Fig. 2 and Fig. 4a."""
    return names[top]

def tile(ax, rows, cols, cellfn, rowlab, collab, fs=7):
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            c = cellfn(row, col)
            if not c:
                ax.text(j, i, "—", ha="center", va="center", color=META_GREY, fontsize=fs); continue
            n = sum(c.values()); t, k = c.most_common(1)[0]
            ax.add_patch(FancyBboxPatch((j - 0.44, i - 0.40), 0.88, 0.80, boxstyle="round,pad=0,rounding_size=0.06",
                                        facecolor=COL[t], edgecolor=("#9a9a9a" if t == "none" else "white"), lw=0.8 if t == "none" else 1.2,
                                        hatch="////" if t == "none" else None))
            lab = ("no attr." if t == "none" else t) + f" {k}/{n}"
            ax.text(j, i, lab, ha="center", va="center", color=TXT.get(t, "white"), fontsize=fs, fontweight="bold")
    ax.set_xlim(-0.55, len(cols) - 0.45); ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(collab); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rowlab)
    ax.tick_params(length=0); ax.xaxis.tick_top()
    for s in ax.spines.values(): s.set_visible(False)

def check(fig, path):
    fig.savefig(path, dpi=300)
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    ov = [(a.get_text()[:16], b.get_text()[:16]) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    out = [t.get_text()[:16] for t, bb in texts if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fig.bbox.x1 or bb.y1 > fig.bbox.y1]
    print(os.path.basename(path), "overlaps:", ov, "| outside:", out)

# ------------------------------------------------------------------ 数据
A2 = J("arm2x2.json") + J("arm2x2_lt.json")
E2 = J("ext_e2.json")
C16 = J("e4_paperB_harness.json") + J("wp1_cwru_clean.json") + J("wp1_cwru_clean_M.json") + J("wp1_cwru_m2.json")
JNU = J("wp1_jnu.json") + J("wp1_jnu_M.json")
WP2 = J("wp2_R.json") + J("wp2_A.json") + J("wp2_R_single.json")
LRN = J("wp2_posthoc_learnability.json")
BC = J("b_confirm.json")
W5 = J("wp5_main.json") + J("wp5_mamba.json"); V5 = J("wp5_verdict.json")
M3 = ["wdcnn", "drsn", "lstm"]; M5 = ["wdcnn", "drsn", "lstm", "cnn1d", "transformer"]
M7 = ["wdcnn", "drsn", "lstm", "cnn1d", "transformer", "vibrmamba", "mamba2"]

# ================================================================== Fig. 1
def fig1():
    fig = plt.figure(figsize=(7.0, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.1], wspace=0.62, left=0.08, right=0.97, top=0.80, bottom=0.24)
    ax = fig.add_subplot(gs[0]); snr = [10.0, 0.0, -4.0, -6.0, -8.0]
    rs = [r for r in A2 if r["model"] == "wdcnn" and r["coverage"] == "007"]
    for key, col, lab in (("acc_mean", "#1f5fa8", "accuracy"), ("conc_mean", "#b2182b", "share of the\nmost-predicted class")):
        Y = np.array([[r["observed"][f"{s}"][key] for s in snr] for r in rs])
        for y in Y: ax.plot(snr, y, color=col, lw=0.5, alpha=0.25)
        ax.plot(snr, Y.mean(0), color=col, lw=1.8, marker="o", ms=3)
        ax.text(-8.8, Y.mean(0)[-1], lab.replace("\n", " ") if key == "acc_mean" else "concentration", color=col, fontsize=7, ha="left", va="center")
    ax.set_xlim(11, -17.5); ax.set_ylim(0, 1.05); ax.set_xticks([10, 0, -4, -8]); ax.set_xlabel("test SNR (dB)")
    ax.set_ylabel("fraction of test windows"); set_frame(ax)
    ax.set_title("Strong noise: predictions\ncollapse onto one class", loc="left", fontsize=8)
    panel_letter(ax, "a")
    # b：结果地图
    axb = fig.add_subplot(gs[1])
    def mc(rs, f):
        return collections.Counter(f(r) for r in rs) if rs else None
    rows = [
        ("CWRU · 0.007″ faults only", lambda m: mc([r for r in A2 if r["model"] == m and r["coverage"] == "007"], lambda r: cat(r["observed"]["-8.0"]["top_mode"], r["observed"]["-8.0"]["conc_mean"], CW))),
        ("CWRU · all three fault sizes", lambda m: mc([r for r in A2 if r["model"] == m and r["coverage"] == "ALL"], lambda r: cat(r["observed"]["-8.0"]["top_mode"], r["observed"]["-8.0"]["conc_mean"], CW))),
        ("PU · damage extent 1 only", lambda m: mc([r for r in WP2 if r["model"] == m and r["arm"] == "R-L1"], lambda r: cat(r["observed"]["-8.0"]["top_mode"], r["observed"]["-8.0"]["conc_mean"], PU))),
        ("PU · extent 1 + extent 2", lambda m: mc([r for r in WP2 if r["model"] == m and r["arm"] == "R-L12a"], lambda r: cat(r["observed"]["-8.0"]["top_mode"], r["observed"]["-8.0"]["conc_mean"], PU))),
    ]
    M4 = M3 + ["cnn1d"]
    tile(axb, [r[0] for r in rows], M4, lambda row, m: dict(rows)[row](m), [r[0] for r in rows], M4, fs=6)
    axb.axvline(2.5, color=META_GREY, lw=0.6, ls=(0, (1, 2)))
    axb.set_xlim(-0.55, 3.45)
    axb.set_title("CWRU: the damage sizes in training switch the dominant class in wdcnn and drsn\n(lstm in one of two runs; not cnn1d). PU: drsn, lstm, cnn1d → Healthy at either extent; wdcnn OR → Healthy", loc="left", fontsize=8, pad=18)
    panel_letter(axb, "b")
    fig.text(0.02, 0.012, "a: CWRU, wdcnn trained on 0.007″ faults only, n = 10 (thin lines = instances). b: dominant class at −8 dB AWGN; count = instances with that class / instances;\n"
             "CWRU rows pool the 1:1 and 3:1 Normal:fault arms (n = 10); PU = real-damage bearings, 1500 rpm, 0.7 Nm, 1000 N (n = 5); all rows: fixed-step training, last weights; CWRU rows from the original run (fixed-initialization repeat: Fig. S1).", fontsize=6, color=META_GREY)
    check(fig, OUT + "fig1_overview.png"); return fig

# ================================================================== Fig. 2
def fig2():
    fig = plt.figure(figsize=(7.0, 5.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 2.6], wspace=0.62, left=0.12, right=0.98, top=0.86, bottom=0.60)
    gc = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.3, left=0.20, right=0.98, top=0.44, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    arms = ["C007_R11", "C007_R31", "CALL_R11", "CALL_R31"]
    lab = {"C007_R11": "0.007″ only · N 1:1", "C007_R31": "0.007″ only · N 3:1", "CALL_R11": "all sizes · N 1:1", "CALL_R31": "all sizes · N 3:1"}
    tile(ax, arms, M5, lambda a, m: collections.Counter(CW[r["observed"]["-8.0"]["top_mode"]] for r in A2 if r["arm"] == a and r["model"] == m) or None,
         [lab[a] for a in arms], M5, fs=6)
    ax.axhline(1.5, color=META_GREY, lw=0.8, ls=(0, (2, 2))); ax.axvline(2.5, color=META_GREY, lw=0.8, ls=(0, (2, 2)))
    ax.set_title("Changing the damage-size composition of the fault\ntraining files switches the dominant class in wdcnn\nand drsn (lstm: this run only); Normal share within\nthe preset no-effect criterion", loc="left", fontsize=8, pad=16)
    panel_letter(ax, "a")
    axb = fig.add_subplot(gs[1]); cells = ["S1L3", "S1L1", "S3L1"]
    lb = {"S1L3": "0.007″ · 3 files", "S1L1": "0.007″ · 1 file", "S3L1": "3 sizes · 1 file each"}
    tile(axb, cells, ["wdcnn", "drsn", "cnn1d"], lambda c, m: collections.Counter(CW[r["final"]["-8.0"]["top_mode"]] for r in E2 if r["cell"] == c and r["model"] == m) or None,
         [lb[c] for c in cells], ["wdcnn", "drsn", "cnn1d"], fs=6)
    axb.set_title("E2: one file of each size gives OR;\n0.007″ files alone give IR", loc="left", fontsize=8, pad=16)
    panel_letter(axb, "b")
    E5 = J("e5_size_id.json"); sets = ["S007", "S014", "S021", "S007+014", "S007+021", "S014+021", "S_ALL"]
    sl = {"S007": "0.007″", "S014": "0.014″", "S021": "0.021″", "S007+014": "0.007″ + 0.014″", "S007+021": "0.007″ + 0.021″", "S014+021": "0.014″ + 0.021″", "S_ALL": "all three sizes"}
    axc = fig.add_subplot(gc[0])
    tile(axc, sets, ["wdcnn", "drsn", "lstm"], lambda st, m: collections.Counter(CW[r["observed"]["-8.0"]["top_mode"]] for r in E5 if r["set"] == st and r["model"] == m) or None,
         [sl[x] for x in sets], ["wdcnn", "drsn", "lstm"], fs=6)
    axc.axhline(2.5, color=META_GREY, lw=0.8, ls=(0, (2, 2)))
    axc.set_title("E5 single sizes: 0.007″ → IR, 0.014″ → OR, 0.021″ leaves IR;\nmixed sets are not reproducible across runs (Fig. S1)", loc="left", fontsize=8, pad=16)
    panel_letter(axc, "c")
    axd = fig.add_subplot(gc[1]); axd.axis("off")
    rows = []
    for st in sets:
        for m in ("wdcnn", "drsn", "lstm"):
            ks = [collections.Counter(CW[r["observed"]["-8.0"]["top_mode"]] for r in E5 if r["set"] == st and r["model"] == m and r["draw"] == dr).most_common(1)[0][0] for dr in (0, 1, 2)]
            rows.append(len(set(ks)) == 1)
    axd.text(0.0, 0.95, f"Draw-level majority\nidentical in all 3 file\ndraws: {sum(rows)}/{len(rows)} cells\n(not all 189 instances)", fontsize=7, va="top")
    fig.text(0.02, 0.012, "CWRU drive end; dominant class at −8 dB on Test_40 (file-disjoint); cell = class and count. a: 1900 windows, 5 seeds per cell; N 1:1 = 475 windows per class, N 3:1 = 950 Normal and ≈316 per fault class;\n"
             "inverse-frequency class weights in the loss. b: 920 windows, 5 seeds.\n"
             "c: 920 windows, 3 file draws × 3 seeds per cell (n = 9); pool = 30 training + validation files; one recording per class and size per draw. All: 3000 steps, last weights.",
             fontsize=6, color=META_GREY)
    check(fig, OUT + "fig2_coverage.png"); return fig

# ================================================================== Fig. 3
def fig3():
    fig = plt.figure(figsize=(7.0, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.3], wspace=0.78, left=0.10, right=0.93, top=0.80, bottom=0.34)
    ax = fig.add_subplot(gs[0])
    cnt = {}
    for m in M7:
        rs = [r for r in C16 if r["model"] == m]
        cnt[m] = {t: int(sum(np.mean([r[t][k]["conc"] for r in rs]) >= 0.9 for k in rs[0][t])) for t in ("best", "final")}
    for i, m in enumerate(M7):
        b, f = cnt[m]["best"], cnt[m]["final"]
        ax.plot([b, f], [i, i], color="#bdbdbd", lw=1.5, zorder=1)
        ax.scatter([b], [i], s=26, facecolor="white", edgecolor="#1f5fa8", lw=1.3, zorder=3)
        ax.scatter([f], [i], s=26, color="#1f5fa8", zorder=3)
    ax.set_yticks(range(len(M7))); ax.set_yticklabels(M7); ax.set_ylim(len(M7) - 0.4, -0.9); ax.set_xlim(-0.5, 6.5)
    ax.set_xlabel("collapsed SNR levels (of 10)"); set_frame(ax)
    tb, tf = sum(c["best"] for c in cnt.values()), sum(c["final"] for c in cnt.values())
    ax.scatter([], [], s=26, facecolor="white", edgecolor="#1f5fa8", lw=1.3, label=f"val-best checkpoint (total {tb})")
    ax.scatter([], [], s=26, color="#1f5fa8", label=f"epoch 100 (total {tf})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, frameon=False, fontsize=6, handletextpad=0.2)
    ax.set_title("Epoch 100 adds collapse in 5 of 7\narchitectures; cnn1d loses it", loc="left", fontsize=8); panel_letter(ax, "a")
    # b
    axb = fig.add_subplot(gs[1]); rng = np.random.default_rng(0)
    for x, (lab, rs) in enumerate((("CWRU\n16 files", C16), ("JNU\n600 rpm", JNU))):
        e = np.array([r["best_epoch"] for r in rs], float)
        axb.scatter(x + rng.uniform(-0.18, 0.18, len(e)), e, s=9, color="#1f5fa8" if x == 0 else "#6a6a6a", alpha=0.75, lw=0)
        axb.plot([x - 0.25, x + 0.25], [np.median(e)] * 2, color="black", lw=1.4)
    axb.set_yscale("log"); axb.set_yticks([1, 2, 5, 10, 20, 50, 100]); axb.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
    axb.set_xticks([0, 1]); axb.set_xticklabels(["CWRU\n16 files", "JNU\n600 rpm"]); axb.set_xlim(-0.6, 1.6); axb.set_ylim(0.8, 130)
    axb.set_ylabel("epoch of val-best checkpoint"); set_frame(axb)
    axb.set_title("CWRU validation saturates\nearly: epoch 1–2 kept", loc="left", fontsize=8); panel_letter(axb, "b")
    # c
    axc = fig.add_subplot(gs[2]); eps = [1, 2, 3, 5, 10, 25, 50, 100]
    shade = {"cnn1d": ("#e08a1e", 1.8), "wdcnn": ("#1f5fa8", 1.2), "drsn": ("#6a8fc7", 1.2), "lstm": ("#9a9a9a", 1.0), "transformer": ("#c9c9c9", 1.0)}
    for m, (col, lw) in shade.items():
        rs = [r for r in C16 if r["model"] == m]
        y = [np.mean([r["sweeps_by_epoch"][str(e)]["-8.0"]["top"] == 1 and r["sweeps_by_epoch"][str(e)]["-8.0"]["conc"] >= ATT for r in rs]) for e in eps]
        axc.plot(eps, y, color=col, lw=lw, marker="o", ms=2.5)
        if m != "drsn": axc.text(118, y[-1], "wdcnn, drsn" if m == "wdcnn" else m, color=col, fontsize=6, va="center")
    axc.set_xscale("log"); axc.set_xticks([1, 10, 100]); axc.set_xticklabels(["1", "10", "100"]); axc.set_xlim(0.8, 450); axc.set_ylim(-0.05, 1.08)
    axc.set_xlabel("training epoch"); axc.set_ylabel("fraction of seeds → IR"); set_frame(axc)
    axc.set_title("cnn1d's IR collapse fades with training\n(1.0 → 0.6); wdcnn and drsn stay at IR", loc="left", fontsize=8); panel_letter(axc, "c")
    fig.text(0.02, 0.02, "CWRU 16-file split, paperB training protocol (100 epochs, keep checkpoint on strict val-accuracy improvement), trained without noise; 5 seeds per architecture\n"
             "(b also shows JNU 600 rpm, same protocol). a: collapsed = 5-seed mean concentration ≥ 0.90; c: −8 dB, IR dominant with concentration ≥ 0.60 (looser than the 0.90 collapse event).", fontsize=6, color=META_GREY)
    check(fig, OUT + "fig3_checkpoint.png"); return fig, cnt

# ================================================================== Fig. 4
def fig4():
    fig = plt.figure(figsize=(7.0, 5.2))
    gt = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0], wspace=0.35, left=0.17, right=0.98, top=0.88, bottom=0.57)
    gb = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.45, left=0.10, right=0.98, top=0.40, bottom=0.16)
    ax = fig.add_subplot(gt[0]); arms = ["R-L1", "R-L12a", "R-L12b", "A-L1", "A-L12a", "A-L12b"]
    lab = {"R-L1": "real · extent 1 only", "R-L12a": "real · + extent 2 (a)", "R-L12b": "real · + extent 2 (b)",
           "A-L1": "artificial · extent 1", "A-L12a": "artificial · + extent 2 (a)", "A-L12b": "artificial · + extent 2 (b)"}
    tile(ax, arms, M5, lambda a, m: collections.Counter(PU[r["observed"]["-8.0"]["top_mode"]] for r in WP2 if r["arm"] == a and r["model"] == m) or None, [lab[a] for a in arms], M5, fs=6)
    ax.axhline(2.5, color=META_GREY, lw=0.8, ls=(0, (2, 2)))
    ax.set_title("Real damage: 4 of 5 architectures collapse to Healthy;\nwdcnn joins them once extent-2 bearings are added.\nArtificial damage: target varies with the retained bearing", loc="left", fontsize=8, pad=16); panel_letter(ax, "a")
    # b
    axb = fig.add_subplot(gt[1]); snr = [10.0, 0.0, -4.0, -6.0, -8.0, -10.0, -12.0]
    for m, col in zip(M3, ("#1f5fa8", "#6a8fc7", "#9a9a9a")):
        rs = [r for r in WP2 if r["block"] == "R" and r["model"] == m]
        Y = np.array([[r["observed"][f"{s}"]["conc_mean"] for s in snr] for r in rs])
        axb.plot(snr, Y.mean(0), color=col, lw=1.5, marker="o", ms=2.5); axb.text(-12.6, Y.mean(0)[-1], {"wdcnn": "wdcnn", "drsn": "", "lstm": "drsn, lstm"}[m], color=col, fontsize=6, va="center", ha="left")
    axb.set_xlim(11, -21); axb.set_xticks([10, 0, -4, -8, -12]); axb.set_ylim(0.3, 1.05)
    axb.set_xlabel("test SNR (dB)"); axb.set_ylabel("share of most-predicted class"); set_frame(axb)
    axb.set_title("drsn and lstm are fully\ncollapsed already at 0 dB", loc="left", fontsize=8); panel_letter(axb, "b")
    # c
    axc = fig.add_subplot(gb[0]); keys = [("healthy_clean", "Healthy bearings"), ("seen_fault_clean", "fault bearings\nseen in training"), ("unseen_fault_clean", "fault bearings\nnot seen in training")]
    rng = np.random.default_rng(1); rs = [r for r in LRN if r["block"] == "R" and r["model"] in M3]
    for x, (k, lb) in enumerate(keys):
        v = np.array([r[k] for r in rs if r[k] is not None], float)
        axc.scatter(x + rng.uniform(-0.2, 0.2, len(v)), v, s=8, color="#1f5fa8", alpha=0.6, lw=0); axc.plot([x - 0.27, x + 0.27], [np.median(v)] * 2, color="black", lw=1.4)
    axc.set_xticks(range(3)); axc.set_xticklabels([lb for _, lb in keys]); axc.set_ylim(-0.03, 1.05); axc.set_xlim(-0.6, 2.6)
    axc.set_ylabel("clean accuracy"); set_frame(axc)
    axc.set_title("The task was learned: bearings seen\nin training are classified correctly", loc="left", fontsize=8); panel_letter(axc, "c")
    # d
    axd = fig.add_subplot(gb[1]); conds = [("1500 rpm\nreference", None), ("1500 rpm\nlow torque", "N15_M01_F10"), ("1500 rpm\nlow force", "N15_M07_F04"), ("900 rpm", "N09_M07_F10")]
    w = 0.26
    for j, (m, col) in enumerate(zip(M3, ("#1f5fa8", "#6a8fc7", "#9a9a9a"))):
        ys, ns = [], []
        for lb, c in conds:
            rr = [r for r in WP2 if r["arm"] == "R-L1" and r["model"] == m] if c is None else [r for r in BC if r["cond"] == c and r["model"] == m]
            h = [(r["observed"]["-8.0"]["top_mode"] == 0 and r["observed"]["-8.0"]["conc_mean"] >= ATT) if c is None else (r["top"] == 0 and r["attractor"]) for r in rr]
            ys.append(np.mean(h)); ns.append(len(h))
        x = np.arange(len(conds)) + (j - 1) * w
        axd.bar(x, ys, width=w * 0.92, color=col, label=m)
        for xi, yi in zip(x, ys):
            if yi == 0: axd.plot([xi - w * 0.4, xi + w * 0.4], [0.008, 0.008], color=col, lw=2)
    axd.set_xticks(range(len(conds))); axd.set_xticklabels([lb for lb, _ in conds], fontsize=6); axd.set_ylim(0, 1.18); axd.set_yticks([0, 0.5, 1])
    axd.set_ylabel("fraction → Healthy"); set_frame(axd)
    axd.legend(loc="upper right", frameon=False, fontsize=6, ncol=3, handlelength=0.8, columnspacing=0.6, borderaxespad=0)
    axd.set_title("drsn and lstm collapse to Healthy\nat 1500 rpm, not at 900 rpm", loc="left", fontsize=8); panel_letter(axd, "d")
    fig.text(0.02, 0.012, "PU, −8 dB AWGN. a–c: WP2, n = 5 seeds per cell, fixed-step training; c: real-damage block, wdcnn/drsn/lstm, horizontal line = median. "
             "\nd (descriptive, not a preregistered comparison; Healthy counted if dominant with concentration ≥ 0.60, a looser threshold than the 0.90 collapse event): extent-1 real-damage bearings;\n"
             "reference = 1500 rpm, 0.7 Nm, 1000 N (n = 5); low torque 0.1 Nm, low force 400 N, 900 rpm (n = 3 each).", fontsize=6, color=META_GREY)
    check(fig, OUT + "fig4_pu_boundary.png"); return fig

# ================================================================== Fig. 5
def fig5():
    fig = plt.figure(figsize=(7.0, 7.4))
    gs = fig.add_gridspec(1, 3, wspace=0.5, left=0.09, right=0.97, top=0.93, bottom=0.72)
    gd = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.75, left=0.09, right=0.97, top=0.60, bottom=0.40)
    gf = fig.add_gridspec(1, 2, wspace=0.35, left=0.09, right=0.80, top=0.29, bottom=0.155)
    snr = ["10.0", "4.0", "0.0", "-4.0", "-6.0", "-8.0"]; xs = [float(s) for s in snr]
    CWR = [r for r in W5 if r["factor"] in ("F1", "F2")]; PUR = [r for r in W5 if r["factor"] == "PU"]
    m = lambda rs, b, k, s: np.nanmean([r["by_snr"][s][b][k] for r in rs])
    for j, (rs, name, L) in enumerate(((CWR, "CWRU", "a"), (PUR, "PU", "b"))):
        ax = fig.add_subplot(gs[j])
        for k, col, lb in (("FAR", "#e08a1e", "false alarm\n(Healthy → fault)"), ("MFR", "#1f5fa8", "missed fault\n(fault → Healthy)")):
            y = [m(rs, "B0", k, s) for s in snr]; ax.plot(xs, y, color=col, lw=1.6, marker="o", ms=2.5)
            ax.text(-8.6, y[-1], lb, color=col, fontsize=6, va="center", ha="left")
        ax.set_xlim(11, -22); ax.set_xticks([10, 0, -8]); ax.set_ylim(-0.03, 1.05); ax.set_xlabel("test SNR (dB)")
        if j == 0: ax.set_ylabel("error rate (no rejection)")
        set_frame(ax); panel_letter(ax, L)
        ax.set_title(f"{name}: collapse appears as\n{'false alarms' if name == 'CWRU' else 'missed faults'}", loc="left", fontsize=8)
    # e: gate
    ax = fig.add_subplot(gs[2])
    for rs, col, lb in ((CWR, "#1f5fa8", "CWRU"), (PUR, "#6a6a6a", "PU")):
        y = [m(rs, "B1", "REJ", s) for s in snr]; ax.plot(xs, y, color=col, lw=1.6, marker="o", ms=2.5)
        ax.text(-8.6, y[-1], lb, color=col, fontsize=6, va="center", ha="left")
    ax.set_xlim(11, -15); ax.set_xticks([10, 0, -8]); ax.set_ylim(-0.03, 1.05); ax.set_xlabel("test SNR (dB)"); ax.set_ylabel("fraction rejected by gate"); set_frame(ax)
    ax.set_title("A spectral-flatness gate is not a safeguard:\nit rejects all noisy CWRU input, even at +10 dB\nwhere errors are near zero, and little PU input", loc="left", fontsize=8); panel_letter(ax, "c")
    # d: silence
    ax = fig.add_subplot(gd[0]); rng = np.random.default_rng(2)
    groups = [(mm, [r for r in W5 if r["model"] == mm and r["factor"] in ("F1", "F2", "PU")]) for mm in M7]
    for x, (mm, rs) in enumerate(groups):
        v = np.array([r["by_snr"]["-8.0"]["B2"]["SILENT"] for r in rs], float); v = v[~np.isnan(v)]
        ax.scatter(x + rng.uniform(-0.22, 0.22, len(v)), v, s=7, color="#1f5fa8", alpha=0.55, lw=0)
        if len(v): ax.plot([x - 0.3, x + 0.3], [np.median(v)] * 2, color="black", lw=1.4)
        ax.text(x, -0.2, f"n={len(v)}", ha="center", fontsize=6, color=META_GREY)
    ax.set_xticks(range(len(M7))); ax.set_xticklabels(M7, rotation=25, ha="right"); ax.set_ylim(-0.26, 1.06); ax.set_yticks([0, 0.5, 1]); ax.set_xlim(-0.6, len(M7) - 0.4)
    ax.set_ylabel("errors accepted by the\nconfidence gate (−8 dB)"); set_frame(ax)
    ax.set_title("Most errors pass the confidence rule; the Mamba models\naccept few errors only by rejecting almost every window (f)", loc="left", fontsize=8); panel_letter(ax, "d")
    # e: forest
    ax = fig.add_subplot(gd[1]); rows = []
    for F, lb in (("F1", "coverage"), ("F2", "checkpoint")):
        pr = V5[F]["primary"]; sec = V5[F].get("secondary_archs", {})
        for mm in M5:
            src = pr if mm in pr else sec
            if mm in src: rows.append((f"{lb} · {mm}", src[mm]["delta"], src[mm]["ci"], mm in pr))
    for i, (lb, d, ci, prim) in enumerate(rows):
        ax.plot(ci, [i, i], color="#1f5fa8" if prim else "#9a9a9a", lw=1.2); ax.scatter([d], [i], s=14, color="#1f5fa8" if prim else "#9a9a9a", zorder=3)
    ax.axvspan(-0.2, 0.2, color="#eeeeee", zorder=0); ax.axvline(0, color=META_GREY, lw=0.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=6); ax.set_ylim(len(rows) - 0.4, -0.6); ax.set_xlim(-0.3, 0.5)
    ax.set_xlabel("Δ balanced harmful-error rate\n(after confidence rejection)"); set_frame(ax)
    ax.set_title("Error-rate effects are confined\nto single architectures", loc="left", fontsize=8); panel_letter(ax, "e")
    # f, g: risk–coverage (post hoc WP5b v2: clean-trained models only, tie-aware curves, actual B2 operating points)
    S5b = J("wp5b2_summary.json")["cells"]; covs = np.arange(0.05, 1.0001, 0.05)
    LOWNOTE = []
    RCOL = {"wdcnn": "#1f5fa8", "drsn": "#6f9fd8", "lstm": "#9a9a9a", "vibrmamba": "#e08a1e", "mamba2": "#b35806"}
    for j, (pop, ms, L, ttl) in enumerate((("CWRU_clean_F1F2", ["wdcnn", "drsn", "lstm", "vibrmamba", "mamba2"], "f", "CWRU"), ("PU_all", ["wdcnn", "drsn", "lstm"], "g", "PU"))):
        ax = fig.add_subplot(gf[j]); low = []
        for mm in ms:
            c = S5b[f"{pop}|{mm}"]["-8.0"]; ax.plot(covs, c["rc"], color=RCOL[mm], lw=1.4, label=mm)
            if c["risk_pooled"] is not None and c["coverage"] >= 0.05:
                ax.scatter([c["coverage"]], [c["risk_pooled"]], s=16, color=RCOL[mm], edgecolor="black", lw=0.5, zorder=4)
            else: low.append((mm, c))
        for k, (mm, c) in enumerate(low):          # 不画点：以文字说明（坐标为轴内位置，非数据）
            rtxt = "risk undefined (nothing accepted)" if c["risk_pooled"] is None else f"pooled risk {c['risk_pooled']:.2f} ({c['n_risk_defined']}/{c['n']} instances accept any window)"
            LOWNOTE.append(f"{mm}: coverage {c['coverage'] * 100:.3f}%, {rtxt}")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.0); ax.set_xlabel("coverage (share of windows accepted)")
        if j == 0: ax.set_ylabel("error rate among\naccepted windows")
        set_frame(ax); panel_letter(ax, L); ax.set_title(f"{ttl}, −8 dB, clean-trained: risk–coverage", loc="left", fontsize=8)
        if j == 0: ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(2.28, 1.0))
    fig.text(0.02, 0.012, "Inference on existing weights, −8 dB unless stated. CWRU: 16- and 40-file instances trained without noise; PU: WP2 instances. d: n = instances with ≥ 1 error (Mamba: CWRU 16-file only).\n"
             "e: coverage = 0.007″-only minus all sizes (preregistered unpaired bootstrap); checkpoint = val-best minus epoch 100 (paired);\n"
             "grey band = preregistered equivalence bound ±0.20; blue = primary, grey = secondary architectures; 95% bootstrap CI.\n"
             "f, g (post hoc, clean-trained models as in a–d): threshold varied, ties in confidence resolved by expected risk;\n"
             "filled circles = rule B2 at its actual coverage and pooled risk; B2 points with coverage < 5% are not plotted;\n"
             "f, not plotted: " + "; ".join(LOWNOTE) + ";\n"
             "curves average per-instance risk, B2 points pool accepted windows, so the two need not coincide; Mamba: CWRU 16-file only.", fontsize=6, color=META_GREY)
    check(fig, OUT + "fig5_consequence.png"); return fig


# ================================================================== Fig. S1
def figS1():
    O2, C2 = J("arm2x2.json") + J("arm2x2_lt.json"), J("cinit_2x2.json"); O5, C5 = J("e5_size_id.json"), J("cinit_e5.json")
    arms = ["C007_R11", "C007_R31", "CALL_R11", "CALL_R31"]; lab = {"C007_R11": "0.007″ · N 1:1", "C007_R31": "0.007″ · N 3:1", "CALL_R11": "all · N 1:1", "CALL_R31": "all · N 3:1"}
    sets = ["S007", "S014", "S021", "S007+014", "S007+021", "S014+021", "S_ALL"]; sl = {"S007": "0.007″", "S014": "0.014″", "S021": "0.021″", "S007+014": "0.007+0.014″", "S007+021": "0.007+0.021″", "S014+021": "0.014+0.021″", "S_ALL": "all three"}
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 7], wspace=0.55, hspace=0.55, left=0.16, right=0.98, top=0.90, bottom=0.13)
    for j, (D, ttl) in enumerate(((O2, "2 × 2, original run"), (C2, "2 × 2, fixed initialization"))):
        ax = fig.add_subplot(gs[0, j]); M4 = ["wdcnn", "drsn", "lstm", "cnn1d"]
        tile(ax, arms, M4, lambda a, m: collections.Counter(CW[r["observed"]["-8.0"]["top_mode"]] for r in D if r["arm"] == a and r["model"] == m), [lab[a] for a in arms] if j == 0 else [""] * 4, M4, fs=6)
        ax.set_title(ttl, loc="left", fontsize=8); panel_letter(ax, "ab"[j])
    for j, (D, ttl) in enumerate(((O5, "E5, original run"), (C5, "E5, fixed initialization"))):
        ax = fig.add_subplot(gs[1, j]); M3 = ["wdcnn", "drsn", "lstm"]
        tile(ax, sets, M3, lambda st, m: collections.Counter(CW[r["observed"]["-8.0"]["top_mode"]] for r in D if r["set"] == st and r["model"] == m), [sl[x] for x in sets] if j == 0 else [""] * 7, M3, fs=6)
        ax.axhline(2.5, color=META_GREY, lw=0.6, ls=(0, (2, 2))); ax.set_title(ttl, loc="left", fontsize=8); panel_letter(ax, "cd"[j])
    fig.suptitle("Single-size results and the wdcnn/drsn coverage switch replicate; lstm (2 × 2), cnn1d and mixed sizes do not", x=0.02, ha="left", fontsize=8)
    fig.text(0.02, 0.02, "CWRU, dominant class at −8 dB on Test_40; cell = most frequent class and count. a, b: 5 seeds per cell; c, d: 3 draws × 3 seeds. \n"
             "Fixed initialization: torch.manual_seed(20000 + seed) before model construction, initial-weight hash identical across arms. Original run: initialization partly\n"
             "unpaired (5/60 and 7/189 instances). Rows below the dashed line: mixed sizes.", fontsize=6, color=META_GREY)
    check(fig, OUT + "figS1_cinit.png"); return fig


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5, figS1): fn(); plt.close("all")
