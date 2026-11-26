import numpy as np

import house as H

S = H.S


def device(plt):
    rows = {(r["unit"], r["class"], r["variant"]): r for r in H.read("energy_device_by_class.csv") if r["run"] == "e4"}
    fig, ax = H.new(plt)
    x = np.arange(3)
    w = 0.36
    for j, (unit, nm) in enumerate((("I1-E", "Incumbent"), ("CIARA", "CIARA"))):
        nom = np.array([H.fnum(rows[(unit, c, "nom")]["level"]) for c in "ABC"])
        lo = np.array([H.fnum(rows[(unit, c, "low")]["level"]) for c in "ABC"])
        hi = np.array([H.fnum(rows[(unit, c, "high")]["level"]) for c in "ABC"])
        dev = np.array([H.fnum(rows[(unit, c, "dev")]["level"]) for c in "ABC"])
        xx = x + (j - 0.5) * w
        ax.bar(xx, nom, w, yerr=[np.maximum(nom - lo, 0), np.maximum(hi - nom, 0)], color=S.COLOR[nm], hatch=S.HATCH[nm], error_kw=S.ERR, label=nm, zorder=3, **S.BAR)
        ax.plot(xx, dev, "D", markersize=9, markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.8, zorder=5,
                label="Measured" if j == 1 else None)
    ax.axhline(0.5, color=S.GRAY, ls="--", lw=2.5, zorder=1, label="Budget")
    ax.set_xticks(x)
    ax.set_xticklabels(["A\nNB-IoT", "B\nLTE-M", "C\n5G"])
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Device Energy")
    H.legend(ax, loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8)
    H.save(fig, "energy_device_band")


def ranks(plt):
    rows = H.read("e9_ranks.csv")
    inv = H.read("e9_inversion.csv")[0]
    fig, ax = H.new(plt)
    ax.plot([0.5, 6.5], [0.5, 6.5], color=S.GRAY, ls="--", lw=2.5, zorder=1)
    for r in rows:
        x, y = int(r["i1_rank"]), int(r["measured_rank"])
        prec = "INT8" if "w8a8" in r["item"] else "INT4"
        kappa = r["item"].split("|")[-1].replace("k", "κ")
        ax.plot(x, y, "o", color=S.RED if x == y else S.BLUE, markersize=14, markeredgecolor="black", markeredgewidth=1.3, zorder=3)
        ax.annotate(f"{prec}, {kappa}", xy=(x, y), xytext=(10, -2), textcoords="offset points", ha="left", va="center", fontsize=H.FS, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))
    ax.annotate(f"{inv['inversions']} of {inv['pairs']} pairs\ninverted", xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top", fontsize=H.FS, linespacing=1.0,
                fontweight="normal")
    ax.set_xlim(0.5, 8.6)
    ax.set_ylim(0.5, 7.6)
    ax.set_xticks(range(1, 7))
    ax.set_yticks(range(1, 7))
    ax.set_xlabel("Rank by Isolated Profile")
    ax.set_ylabel("Rank Measured")
    H.save(fig, "e9_rank_scatter")


def main():
    plt = H.setup()
    device(plt)
    ranks(plt)


main()
