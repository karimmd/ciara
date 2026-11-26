import house as H

S = H.S


def bias_plot(plt):
    cells = [r for r in H.read("e2_cells.csv") if r["look_day"] == "7"]
    std = {(r["arm"], r["outcome"]): r for r in H.read("e2_energy_standardized.csv") if r["look_day"] == "7"}
    cov = next(r for r in H.read("e2_coverage_by_level.csv") if abs(H.fnum(r["delta"]) - 0.05) < 1e-9)
    fig, ax = H.new(plt)
    lim = 0.09
    dark = "#333333"
    ax.fill_between([0, 0.06], [0, 0.06], 0, color="#2CA02C", alpha=0.07, lw=0, zorder=0)
    ax.plot([0, 0.06], [0, 0.06], color=S.GRAY, ls="--", lw=2.5, zorder=1)
    ax.annotate("Bias = radius", xy=(0.0510, 0.033), ha="right", va="top", color=S.GRAY, fontsize=H.FS)
    pts = [(H.fnum(r["radius"]), abs(H.fnum(r["bias"]))) for r in cells if r["covered_radius"] == "1"]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color="#9E9E9E", markersize=7, markeredgecolor="#555555", markeredgewidth=0.8, zorder=3)
    tips = []
    for r in cells:
        if r["covered_radius"] == "1":
            continue
        x, y = H.fnum(r["radius"]), abs(H.fnum(r["bias"]))
        ax.plot(x, y, "s", markerfacecolor="white", markeredgecolor=dark, markeredgewidth=2.2, markersize=10, zorder=4)
        s = std.get((r["arm"], r["outcome"]))
        if s and s["covered_std_radius"] == "1":
            ys = abs(H.fnum(s["bias_std"]))
            ax.annotate("", xy=(x, ys), xytext=(x, y), arrowprops=dict(arrowstyle="-|>", color=dark, lw=1.6, shrinkA=6, shrinkB=3), zorder=3)
            ax.plot(x, ys, "s", color=dark, markersize=7, zorder=5)
            tips.append((x, y))
    if tips:
        tx = max(t[0] for t in tips)
        ty = sum(t[1] for t in tips) / len(tips)
        ax.annotate("Class A energy:\nlink mix", xy=(tx + 0.0012, ty), xytext=(0.0315, 0.080), textcoords="data", ha="left", va="center",
                    fontsize=H.FS, linespacing=1.0, color=dark, arrowprops=dict(arrowstyle="-", color="#777777", lw=1.0))
    ax.annotate(f"{cov['covered_planned']} of {cov['cells']} covered\n({cov['covered_standardized']} standardized)", xy=(0.0315, 0.060),
                xycoords="data", ha="left", va="center", fontsize=H.FS, linespacing=1.05, color="#222222")
    ax.set_xlim(0.015, 0.052)
    ax.set_ylim(0, 0.088)
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.set_xlabel("Certified Radius")
    ax.set_ylabel("Absolute Bias")
    H.save(fig, "val_bias_std")


def coverage_plot(plt):
    rows = H.read("e2_coverage_by_level.csv")
    x = [H.fnum(r["nominal_level"]) for r in rows]
    fig, ax = H.new(plt)
    ax.plot([0.5, 1.0], [0.5, 1.0], color=S.GRAY, ls="--", lw=2.5, zorder=1, label="Nominal")
    ax.plot(x, [H.fnum(r["share_planned"]) for r in rows], color=S.RED, lw=3.2, marker="o", markersize=8, markeredgecolor="black", markeredgewidth=1.0,
            zorder=3, label="Planned")
    ax.plot(x, [H.fnum(r["share_standardized"]) for r in rows], color=S.RED, ls=":", lw=3.0, marker="o", markersize=9, markerfacecolor="white",
            markeredgecolor=S.RED, markeredgewidth=2.0, zorder=3, label="Link-standardized")
    ax.axvline(0.95, color=S.BLUE, ls=":", lw=2.2, zorder=1)
    ax.annotate("\u03b4 = 0.05", xy=(0.95, 0.785), xycoords="data", xytext=(-5, 0), textcoords="offset points", ha="right", va="center",
                color=S.BLUE, fontweight="normal", fontsize=H.FS)
    ax.set_xlim(0.48, 1.005)
    ax.set_ylim(0.40, 1.04)
    ax.set_xlabel("Nominal Level (1 − δ)")
    ax.set_ylabel("Cells Covered")
    H.legend(ax, loc="lower right", fontsize=H.FS)
    H.save(fig, "val_coverage")


def main():
    plt = H.setup()
    bias_plot(plt)
    coverage_plot(plt)


main()
