import house as H

S = H.S


def curve(rows, scope):
    pts = sorted((H.fnum(r["factor"]), H.fnum(r["quality_cost_bound"])) for r in rows if r["scope"] == scope and r["switch"] == "1")
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    plt = H.setup()
    rows = H.read("campaign_budget_sweep.csv")
    fig, ax = H.new(plt)
    ax.axvline(1.0, color=S.GRAY, ls=":", lw=2.2, zorder=1)
    ax.annotate("Planned", xy=(1.0, 0.985), xycoords=("data", "axes fraction"), xytext=(-5, 0), textcoords="offset points", ha="right", va="top", color=S.GRAY,
                fontsize=H.FS)
    others = (("Y_L_f2", "Tail 2", "v"), ("Y_D", "Deadline", "^"), ("energy", "Energy", "D"))
    xs, ys = curve(rows, "Y_L_f2")
    ax.plot(xs, ys, color="#8C8C8C", lw=3.0, zorder=2)
    for sc, lab, mk in others:
        cx, cy = curve(rows, sc)
        ax.plot(cx[0], cy[0], mk, color="#8C8C8C", markersize=11, markeredgecolor="black", markeredgewidth=1.0, zorder=4, linestyle="none", label=f"{lab} {cx[0]:.2f}")
        print(f"{lab:9s} tightest certifiable scale {cx[0]:.2f}, quality cost bound {cy[0]:.4f}")
    ax.annotate("Others", xy=(1.6, ys[-1]), xytext=(0, 4), textcoords="offset points", ha="right", va="bottom", fontsize=H.FS, color="#666666")
    for sc, lab, col, ls, dy in (("Y_C", "Cloud", S.RED, "-", -4), ("Y_L_f1", "Tail 1", S.BLUE, "--", -4)):
        cx, cy = curve(rows, sc)
        ax.plot(cx, cy, color=col, ls=ls, lw=3.4, zorder=3)
        ax.plot(cx[0], cy[0], "o", color=col, markersize=11, markeredgecolor="black", markeredgewidth=1.0, zorder=5, linestyle="none", label=f"{lab} {cx[0]:.2f}")
        ax.annotate(lab, xy=(1.6, cy[-1]), xytext=(0, dy), textcoords="offset points", ha="right", va="bottom" if dy > 0 else "top", fontsize=H.FS, color=col)
        print(f"{lab:9s} tightest certifiable scale {cx[0]:.2f}, quality cost bound {cy[0]:.4f}, at 1.6: {cy[-1]:.4f}")
    h, lab = ax.get_legend_handles_labels()
    order = [lab.index(x) for x in lab if x.startswith("Cloud")] + [lab.index(x) for x in lab if x.startswith("Tail 1")] + \
            [lab.index(x) for x in lab if x.split()[0] in ("Energy", "Deadline", "Tail") and not x.startswith("Tail 1")]
    H.legend(ax, handles=[h[i] for i in order], labels=[lab[i] for i in order], loc="lower center", ncol=2, handlelength=0.9, handletextpad=0.3, labelspacing=0.15,
             columnspacing=0.6, bbox_to_anchor=(0.5, 0.0))
    ax.set_xlim(0.45, 1.62)
    ax.set_ylim(0.012, 0.101)
    ax.set_yticks([0.06, 0.07, 0.08, 0.09])
    ax.set_xlabel("Budget Scale Factor")
    ax.set_ylabel("Quality Cost Bound")
    H.save(fig, "op_curve")


main()
