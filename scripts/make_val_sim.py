import house as H

S = H.S
FLOOR = 1e-4
SERIES = (("per-epoch", "Per-epoch randomization", S.BLUE, "o", "--"), ("switchback-no-burnin", "Blocks, no burn-in", S.ORANGE, "s", ":"),
          ("safe-block", "Blocks with burn-in", S.RED, "^", "-"))


def main():
    plt = H.setup()
    rows = H.read("sim_estimators.csv")
    fig, ax = H.new(plt)
    for d, lab, c, m, ls in SERIES:
        rr = sorted((r for r in rows if r["design"] == d), key=lambda r: float(r["t_mix"]))
        x = [float(r["t_mix"]) for r in rr]
        y = [max(abs(float(r["dm_bias_exceed_non"])), FLOOR) for r in rr]
        print("val_sim", d, [f"{abs(float(r['dm_bias_exceed_non'])):.2e}" for r in rr])
        ax.plot(x, y, marker=m, color=c, ls=ls, label=lab, lw=3.0, markersize=10, markeredgewidth=1.3, markeredgecolor="black", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 20])
    ax.set_xticklabels(["2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlim(1.7, 23)
    ax.set_ylim(5e-5, 1.5e2)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xlabel("Mixing Time (Epochs)")
    ax.set_ylabel("Absolute Bias")
    H.legend(ax, loc="upper left", ncol=1)
    H.save(fig, "val_sim")


main()
