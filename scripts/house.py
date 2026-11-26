from __future__ import annotations

import csv
import gzip
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
FIGS = os.path.join(OUT, "figures")
TABS = os.path.join(OUT, "tables")
sys.path.insert(0, HERE)
import tmc_style as S

A_INC = "I1-E"
A_PRIO = "q35-4b-ih-w8a8|b8|c32|f1prio"
A_FIFO = "q35-4b-ih-w8a8|b8|c32|fifo"
BUDGETS = {"Y_L_f1": 0.10, "Y_L_f2": 0.10, "Y_D": 0.10, "Y_E_A": 0.5, "Y_E_B": 0.5, "Y_E_C": 0.5, "Y_C": 0.30}
TARGET = {1: 9.504, 2: 44.331}

CONTROLLERS = S.CONTROLLERS
ARM = {A_INC: ("Incumbent", S.BLUE, "s", "-"), A_PRIO: ("Cascade, priority", S.RED, "^", "-"), A_FIFO: ("Cascade, FIFO", S.ORANGE, "v", "--")}
MARK = {"CIARA": "o", "Incumbent": "s", "I1-S": "D", "I1-Oracle": "P", "Mix-or-Split": "X", "RACE": "v", "RecServe": "^", "Bandit": "<"}
LSTYLE = {"CIARA": "-", "Incumbent": "-", "I1-S": "--", "I1-Oracle": ":", "Mix-or-Split": "-.", "RACE": "--", "RecServe": ":", "Bandit": "-."}


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read(name):
    p = os.path.join(DATA, name)
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


PRINT_W_PT = 0.49 * 252.0
SAVED_W_PT = 5.35 * 72.0
K = SAVED_W_PT / PRINT_W_PT
PRINTED = {"label": 7.0, "tick": 6.5, "legend": 6.3, "note": 6.3}
FS_LABEL, FS_TICK, FS_LEG, FS = (round(PRINTED[k] * K, 1) for k in ("label", "tick", "legend", "note"))


def setup():
    plt = S.setup()
    plt.rcParams.update({"font.weight": "normal", "axes.labelweight": "normal", "axes.titleweight": "normal", "axes.labelsize": FS_LABEL,
                         "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEG})
    return plt


FIGSIZE = (5, 3.4)

CANVAS = (SAVED_W_PT / 72.0, SAVED_W_PT / 72.0 / 1.6)
PAD_IN = 0.04
BOTTOM_IN = 0.78
TOP_IN = 0.06


def new(plt, figsize=None):
    canvas = figsize is None
    fig, ax = plt.subplots(figsize=CANVAS if canvas else figsize)
    ax.grid(axis="x", visible=False)
    fig._house_canvas = canvas
    if canvas:
        fig.subplots_adjust(left=0.18, right=0.97, bottom=BOTTOM_IN / CANVAS[1], top=1 - TOP_IN / CANVAS[1])
    return fig, ax


def fit(fig, bottom_in=None):
    W, Hh = fig.get_size_inches()
    sp = fig.subplotpars
    left, right = sp.left, sp.right
    if bottom_in is not None:
        fig.subplots_adjust(bottom=bottom_in / Hh)
    for _ in range(6):
        fig.subplots_adjust(left=left, right=right)
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        dl, dr = PAD_IN - bb.x0, bb.x1 - (W - PAD_IN)
        if abs(dl) < 0.003 and abs(dr) < 0.003:
            break
        left, right = left + dl / W, right - dr / W
    if bb.y0 < -0.005 or bb.y1 > Hh + 0.005:
        print(f"WARNING text outside the canvas vertically: bottom {bb.y0:.3f} in, top {bb.y1 - Hh:+.3f} in")
    return {"left_in": left * W, "right_in": (1 - right) * W, "bottom_text_in": bb.y0, "top_text_in": Hh - bb.y1}


def save(fig, name, bottom_in=None):
    os.makedirs(FIGS, exist_ok=True)
    pdf = os.path.join(FIGS, name + ".pdf")
    if getattr(fig, "_house_canvas", False):
        m = fit(fig, bottom_in)
        print(f"canvas {name}: left {m['left_in']:.2f} in, right {m['right_in']:.2f} in, text {m['bottom_text_in']:.3f} in above the bottom, {m['top_text_in']:.3f} in below the top")
        kw = {"bbox_inches": fig.bbox_inches}
    else:
        kw = {"bbox_inches": "tight", "pad_inches": 0.05}
    fig.savefig(pdf, format="pdf", facecolor="white", **kw)
    fig.savefig(os.path.join(FIGS, name + ".png"), format="png", dpi=110, facecolor="white", **kw)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("saved", pdf)


def hline(ax, y, label="Budget", ha="right", x=1.0, dy=4):
    ax.axhline(y, color=S.GRAY, ls="--", lw=2.5, zorder=1)
    ax.annotate(label, xy=(x, y), xycoords=("axes fraction", "data"), xytext=(-4 if ha == "right" else 4, dy), textcoords="offset points", ha=ha,
                va="bottom" if dy >= 0 else "top", color=S.GRAY, fontsize=FS)


def vline(ax, x, label=None, y=0.97, ha="right", color=None, ls=":", lw=2.5, fs=None):
    ax.axvline(x, color=color or S.GRAY, ls=ls, lw=lw, zorder=1)
    if label:
        ax.annotate(label, xy=(x, y), xycoords=("data", "axes fraction"), xytext=(-5 if ha == "right" else 5, 0), textcoords="offset points", ha=ha, va="top",
                    color=color or S.GRAY, fontsize=fs or FS)


def legend(ax, **kw):
    kw["fontsize"] = FS_LEG
    kw.setdefault("handletextpad", 0.4)
    kw.setdefault("borderpad", 0.35)
    kw.setdefault("labelspacing", 0.3)
    return S.legend(ax, **kw)


def note(ax, text, xy, ha="left", va="top", color="#222222", fs=None, bold=False, coords="axes fraction"):
    ax.annotate(text, xy=xy, xycoords=coords, ha=ha, va=va, fontsize=fs or FS, fontweight="bold" if bold else "normal", color=color, zorder=8,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85))
