"""Fig. S5: T13 net uplift by 1 m/s wind-speed bin, reference-turbine wind speed vs T13's own anemometer.
Usage: python make_figS5.py <ws_bins.csv> <FIGDIR>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

B = pd.read_csv(sys.argv[1])
F = sys.argv[2]
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 400})
mm = 1 / 25.4
fig, ax = plt.subplots(1, 2, figsize=(180 * mm, 62 * mm))
B = B[(B.freq >= 0.005)]
for ws, c, lab in [("refWS", BLUE, "reference-turbine wind speed"), ("ownWS", ORANGE, "T13 nacelle anemometer")]:
    b = B[B.ws == ws]
    ax[0].plot(b.bin + 0.5, b.uplift_pct, "-o", ms=2.5, lw=1, color=c, label=lab)
    ax[1].bar(b.bin + (0.3 if ws == "refWS" else 0.7), b.share_E, width=0.4, color=c, label=lab)
for a in ax:
    a.axhline(0, color=MUTED, lw=0.5)
    a.axvspan(10, 18, color="#e6e5e0", zorder=0)
    a.set_xlabel("Wind speed bin (m/s)")
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
ax[0].set_ylabel("Net power uplift of T13 (% of expected)")
ax[1].set_ylabel("Share of T13 energy gain")
ax[0].legend(frameon=False, fontsize=6)
ax[1].text(14, ax[1].get_ylim()[1] * 0.92, "10–18 m/s", ha="center", fontsize=6, color=MUTED)
for a, l in zip(ax, "ab"):
    a.text(-0.16, 1.02, l, transform=a.transAxes, fontsize=9, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{F}/figS5_wind_speed_bins.png")
fig.savefig(f"{F}/figS5_wind_speed_bins.pdf")
