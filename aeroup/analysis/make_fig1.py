"""Fig. 1: study design (a) timeline of retrofit and analysis windows; (b) reference-cell estimator.
Nature-style layout: 180 mm wide, sans-serif 6-7 pt, bold lower-case panel letters. Usage: python make_fig1.py <FIGDIR>"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

F = sys.argv[1]
BLUE, ORANGE, GREY, LIGHT, INK, MUTED = "#2a78d6", "#eb6834", "#9a9993", "#e6e5e0", "#0b0b0b", "#52514e"
AQUA = "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 400})
mm = 1 / 25.4
fig = plt.figure(figsize=(180 * mm, 118 * mm))
axa = fig.add_axes([0.16, 0.56, 0.81, 0.40])
axb = fig.add_axes([0.0, 0.0, 1.0, 0.50])
D = pd.Timestamp

# ---------------- (a) timeline
rows = [("T13", 5), ("Other 20 turbines", 4), ("T13 window analysis\n(main estimate)", 2.9),
        ("2022–23 rollout\nreplication (D1)", 1.8), ("Known-null check\n2020→2021 (D2)", 0.7)]


def bar(y, a, b, color, h=0.42, alpha=1.0, hatch=None, ec="none"):
    axa.add_patch(Rectangle((mdates.date2num(D(a)), y - h / 2), mdates.date2num(D(b)) - mdates.date2num(D(a)), h,
                            fc=color, alpha=alpha, hatch=hatch, ec=ec, lw=0.5))


# T13 state
bar(5, "2020-01-01", "2021-09-23", GREY)
bar(5, "2021-09-23", "2021-09-30", ORANGE)
bar(5, "2021-09-30", "2023-12-31", BLUE)
# others
bar(4, "2020-01-01", "2022-07-14", GREY)
bar(4, "2022-07-14", "2023-05-26", ORANGE, alpha=0.55)
bar(4, "2023-05-26", "2023-12-31", BLUE)
axa.text(mdates.date2num(D("2022-12-15")), 4, "works: 12 done by Oct 2022,\n8 by May 2023", ha="center", va="center", fontsize=5.5, color=INK)
# T13 window
bar(2.9, "2020-01-01", "2021-09-22", "#ffffff", ec=INK)
bar(2.9, "2021-09-30", "2022-07-13", "#ffffff", ec=INK, hatch="////")
axa.text(mdates.date2num(D("2020-11-10")), 2.9, "pre (21 months)", ha="center", va="center", fontsize=5.5)
axa.text(mdates.date2num(D("2022-02-20")), 3.27, "post (9.5 months)", ha="center", va="bottom", fontsize=5.5)
# replication
bar(1.8, "2021-10-01", "2022-07-13", "#ffffff", ec=INK)
for a, b in [("2023-06-01", "2023-07-31"), ("2023-10-01", "2023-12-31")]:
    bar(1.8, a, b, "#ffffff", ec=INK, hatch="////")
axa.text(mdates.date2num(D("2022-02-20")), 1.8, "pre (ref = T13)", ha="center", va="center", fontsize=5.5)
axa.text(mdates.date2num(D("2023-09-15")), 2.12, "post: Jun–Jul, Oct–Dec 2023", ha="center", va="bottom", fontsize=5.5)
# null
for a, b in [("2020-06-01", "2020-07-31"), ("2020-10-01", "2020-12-31")]:
    bar(0.7, a, b, "#ffffff", ec=INK)
for a, b in [("2021-06-01", "2021-07-31"), ("2021-10-01", "2021-12-31")]:
    bar(0.7, a, b, "#ffffff", ec=INK, hatch="////")
axa.text(mdates.date2num(D("2021-03-01")), 1.02, "Jun–Jul, Oct–Dec in both years; ref = T01 or T21", ha="center", va="bottom", fontsize=5.5)
axa.set_yticks([r[1] for r in rows])
axa.set_yticklabels([r[0] for r in rows], fontsize=6.5)
axa.set_ylim(0.2, 5.5)
axa.set_xlim(mdates.date2num(D("2020-01-01")), mdates.date2num(D("2023-12-31")))
axa.xaxis.set_major_locator(mdates.YearLocator())
axa.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
axa.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for s in ["top", "right", "left"]:
    axa.spines[s].set_visible(False)
axa.tick_params(axis="y", length=0)
axa.axvline(mdates.date2num(D("2021-09-26")), color=ORANGE, lw=0.6, ls=":")
axa.axvline(mdates.date2num(D("2022-07-14")), color=ORANGE, lw=0.6, ls=":")
# legend (row of the null check, right side, which is empty)
items = [(GREY, "original blades", None), (ORANGE, "VG works", None), (BLUE, "retrofitted", None),
         ("#ffffff", "pre window", None), ("#ffffff", "post window", "////")]
for i, (c, lab, h) in enumerate(items):
    x0 = mdates.date2num(D("2022-03-01")) + (i % 3) * 200
    yy = 0.95 - 0.42 * (i // 3)
    axa.add_patch(Rectangle((x0, yy - 0.11), 28, 0.22, fc=c, ec=INK if c == "#ffffff" else "none", hatch=h, lw=0.5))
    axa.text(x0 + 36, yy, lab, va="center", fontsize=5.8, color=INK)
fig.text(0.01, 0.97, "a", fontsize=9, fontweight="bold")

# ---------------- (b) estimator schematic
axb.set_xlim(0, 180)
axb.set_ylim(0, 55)
axb.axis("off")
fig.text(0.01, 0.48, "b", fontsize=9, fontweight="bold")


def box(x, y, w, h, title, body, color, fs=5.5):
    axb.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5", fc="#ffffff", ec=color, lw=0.9))
    axb.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=6.5, fontweight="bold", color=INK)
    axb.text(x + w / 2, y + h - 7.2, body, ha="center", va="top", fontsize=fs, color=INK, linespacing=1.4)


def arrow(x0, y0, x1, y1):
    axb.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=MUTED))


box(3, 26, 38, 24, "1  Farm state", "for every 10-min stamp:\nmean power of reference\nturbines (≥10 operating),\n50 kW bins  ×  circular-mean\nyaw, 8 sectors  →  cell c", GREY)
box(49, 26, 38, 24, "2  Counterfactual", "target turbine's own\npre-window records in c\n→ expected P, time at\nrated, torque T = P/ω\nraised to m", GREY)
box(95, 26, 38, 24, "3  Effect", r"post records: observed $-$ expected" "\n" r"$E=\Sigma P/\Sigma\hat{P}-1$" "\n" r"$R=\Delta$ share with $P\geq0.95P_r$" "\n" r"$D_m=(\Sigma T^m/\Sigma\hat{T}^m)^{1/m}-1$", BLUE)
box(141, 26, 36, 24, "4  Load cost ratio", r"$C=D_m/E$" "\n\ncompared with pure\npower scaling at\nunchanged rotor speed", BLUE)
arrow(41.8, 38, 48.2, 38)
arrow(87.8, 38, 94.2, 38)
arrow(133.8, 38, 140.2, 38)
box(46, 2, 90, 17, "Drift correction and inference", "same estimator and dates on each of the 20 unmodified turbines = null drift\n"
    "corrected effect = T13 effect − median of the 20;  rank of T13 among 21 = permutation test\n"
    "known-truth check: +4 % power (capped) injected into each unmodified turbine", ORANGE, fs=5.2)
arrow(114, 19.6, 114, 25.4)
box(3, 2, 38, 17, "Reference set", "all turbines except T13\nand the target\n(T13 never a reference\nin the main estimate)", GREY)
box(141, 2, 36, 17, "Outputs", "E, R, D_m (m = 3 … 10),\nrotor speed, pitch,\n10-min power std,\nbearing temperatures", GREY)
fig.savefig(f"{F}/fig1_design.png")
fig.savefig(f"{F}/fig1_design.pdf")
print("fig1 written")
