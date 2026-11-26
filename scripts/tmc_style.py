from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out", "figures")
FIGSIZE = (5, 3.4)
K = (5.35 * 72.0) / (0.49 * 252.0)
FS_LABEL, FS_TICK, FS_LEG, FS_NOTE = (round(v * K, 1) for v in (7.0, 6.5, 6.3, 6.3))

RED, BLUE, TEAL = "#D62728", "#2E86AB", "#70C1B3"
ORANGE, GREEN, PURPLE, GRAY = "#FF7F0E", "#2CA02C", "#9467BD", "#7F7F7F"
COLOR = {"CIARA": RED, "Incumbent": BLUE, "I1-S": TEAL, "I1-Oracle": ORANGE, "Mix-or-Split": GREEN, "RACE": PURPLE, "RecServe": GRAY, "Bandit": "#8C564B"}
HATCH = {"CIARA": "", "Incumbent": "///", "I1-S": "...", "I1-Oracle": "\\\\\\", "Mix-or-Split": "xxx", "RACE": "+++", "RecServe": "---", "Bandit": "ooo"}

CONTROLLERS = (("CIARA", "CIARA"), ("Incumbent", "I1-E"), ("I1-S", "I1-S"), ("I1-Oracle", "I1-oracle"), ("Mix-or-Split", "I3"), ("RACE", "I2"), ("RecServe", "RecServe"), ("Bandit", "CB"))

ROWS = (("Y_L_f1", "Tail 1"), ("Y_L_f2", "Tail 2"), ("Y_D", "Deadline"), ("Y_E_A", "Energy A"), ("Y_E_B", "Energy B"), ("Y_E_C", "Energy C"), ("Y_C", "Cloud"))

BAR = {"edgecolor": "black", "linewidth": 1.5, "alpha": 0.9}
ERR = {"capsize": 4, "capthick": 2.0, "elinewidth": 2.0, "ecolor": "black"}
LINE = {"linewidth": 3.0, "markersize": 10, "markeredgewidth": 1.5, "markeredgecolor": "white"}


def fnum(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_csv(name, data_dir=None):
    p = os.path.join(data_dir or DATA, name)
    opener = gzip.open if p.endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--out-dir", default=OUT)
    return ap.parse_args()


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for p in fm.findSystemFonts(fontext="ttf"):
        if os.path.basename(p).lower() in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
            fm.fontManager.addfont(p)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": FS_TICK, "font.weight": "normal",
        "axes.labelsize": FS_LABEL, "axes.labelweight": "normal", "axes.titleweight": "normal", "axes.linewidth": 2.0,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "xtick.major.width": 1.5, "ytick.major.width": 1.5, "xtick.major.size": 6, "ytick.major.size": 6,
        "legend.fontsize": FS_LEG, "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "black", "legend.fancybox": False,
        "grid.alpha": 0.3, "grid.linewidth": 0.8, "grid.linestyle": "--",
        "axes.grid": True, "axes.axisbelow": True, "axes.unicode_minus": True,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def new_fig(plt):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.grid(axis="x", visible=False)
    return fig, ax


def legend(ax, **kw):
    leg = ax.legend(**kw)
    leg.get_frame().set_linewidth(1.5)
    leg.set_zorder(10)
    return leg


def budget_line(ax, y, label="Budget", margin=0.0):
    if margin:
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + margin)
    ax.axhline(y, color=GRAY, ls="--", lw=2.5, zorder=1)
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, 4), textcoords="offset points", ha="right", va="bottom", color=GRAY, fontweight="normal", fontsize=FS_NOTE)


def rotate_ticks(ax, deg=30):
    for t in ax.get_xticklabels():
        t.set_rotation(deg)
        t.set_ha("right")
        t.set_rotation_mode("anchor")


def save(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name + ".pdf")
    fig.savefig(p, format="pdf", bbox_inches="tight", pad_inches=0.05, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("saved", p)
    return p
