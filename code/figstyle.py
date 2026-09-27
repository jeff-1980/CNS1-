"""Figure style used for the manuscript figures (matplotlib rcParams and two small helpers)."""
import matplotlib as mpl

META_GREY = "#888888"

def apply_figure_style(sizes=(8, 7, 6)):
    big, mid, small = sizes
    mpl.rcParams.update({
        "font.size": big, "axes.titlesize": big, "axes.labelsize": big, "legend.fontsize": mid,
        "xtick.labelsize": small, "ytick.labelsize": small, "axes.titlelocation": "left",
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 3.0, "xtick.major.width": 0.6, "ytick.major.size": 3.0, "ytick.major.width": 0.6,
        "lines.linewidth": 1.2, "patch.linewidth": 0.6, "legend.frameon": False,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42})

def panel_letter(ax, letter, dx=-0.18, dy=1.02):
    ax.text(dx, dy, letter.lower(), transform=ax.transAxes, fontweight="bold",
            fontsize=mpl.rcParams["font.size"] + 1, va="bottom", ha="left")

def set_frame(ax):
    for side, vis in (("top", False), ("right", False), ("bottom", True), ("left", True)):
        ax.spines[side].set_visible(vis)
        if vis: ax.spines[side].set_linewidth(0.6)
    ax.tick_params(direction="out", length=3, width=0.6)
