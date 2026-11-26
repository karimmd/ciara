import numpy as np

import house as H

S = H.S
CTX = (("hi", "−25%"), ("campaign", "Nominal"), ("lo", "+25%"), ("mps", "MPS"))


def plot(plt, c, outcome, ylabel, name, ymax):
    fig, ax = H.new(plt)
    x = np.arange(len(CTX))
    w = 0.36
    for j, (key, nm) in enumerate((("J_a0", "Incumbent"), ("J_pi", "CIARA"))):
        vals = [c[(lvl, outcome)][key] / H.BUDGETS[outcome] for lvl, _ in CTX]
        xx = x + (j - 0.5) * w
        bars = ax.bar(xx, vals, w, color=S.COLOR[nm], hatch=S.HATCH[nm], label=nm, zorder=3, **S.BAR)
        if nm == "CIARA":
            for b, v in zip(bars, vals):
                if v > 1.0:
                    b.set_linewidth(3.5)
                    ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=H.FS,
                                fontweight="normal", color=S.RED)
    ax.axhspan(1.0, ymax, color="#D62728", alpha=0.06, lw=0, zorder=0)
    ax.axhline(1.0, color=S.GRAY, ls="--", lw=2.5, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in CTX])
    ax.set_xlabel("Capacity Context")
    ax.set_xlim(-0.6, len(CTX) - 0.4)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hs = [Patch(facecolor=S.COLOR[n], hatch=S.HATCH[n], edgecolor="black", linewidth=1.5, label=n) for n in ("Incumbent", "CIARA")]
    hs.append(Line2D([0], [0], color=S.GRAY, ls="--", lw=2.5, label="Budget"))
    H.legend(ax, handles=hs, loc="upper right", ncol=2, columnspacing=0.6, handlelength=1.4)
    H.save(fig, name)


def main():
    plt = H.setup()
    c = {}
    for r in H.read("e8v9_contrast.csv"):
        c[(r["level"], r["outcome"])] = {"J_pi": H.fnum(r["J_pi"]), "J_a0": H.fnum(r["J_a0"])}
    for lvl, lab in CTX:
        print(lab, {k: (round(c[(lvl, k)]["J_pi"], 3), round(c[(lvl, k)]["J_a0"], 3)) for k in ("Y_C", "Y_L_f1", "Y_Q")})
    plot(plt, c, "Y_C", "Cloud / Budget", "load_cloud", 2.6)
    plot(plt, c, "Y_L_f1", "Tail 1 / Budget", "load_tail1", 2.6)


main()
