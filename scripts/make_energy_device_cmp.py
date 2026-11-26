import numpy as np

import house as H

S = H.S
ARMS = (("CIARA", "CIARA"), ("Incumbent", "I1-E"), ("Mix-or-Split", "I3"), ("RACE", "I2"), ("RecServe", "RecServe"), ("Bandit", "CB"))
ROWS = (("Y_E_A", "A\nNB-IoT"), ("Y_E_B", "B\nLTE-M"), ("Y_E_C", "C\n5G"))


def main():
    plt = H.setup()
    e = {(r["arm"], r["outcome"]): (H.fnum(r["mean"]), H.fnum(r["se"])) for r in H.read("e1e5_estimates.csv")}
    fig, ax = H.new(plt)
    x = np.arange(len(ROWS))
    w = 0.8 / len(ARMS)
    ymax = 2.9
    ax.axhspan(1.0, ymax, color="#D62728", alpha=0.06, lw=0, zorder=0)
    for j, (nm, arm) in enumerate(ARMS):
        m = np.array([e[(arm, k)][0] / 0.5 for k, _ in ROWS])
        up = np.array([2 * e[(arm, k)][1] / 0.5 for k, _ in ROWS])
        ax.bar(x + (j - (len(ARMS) - 1) / 2) * w, m, w, yerr=[np.zeros_like(up), up], color=S.COLOR[nm], hatch=S.HATCH[nm], label=nm, zorder=3,
               error_kw={**S.ERR, "capsize": 2, "elinewidth": 1.4, "capthick": 1.4}, edgecolor="black", linewidth=1.0, alpha=0.9)
        print(f"{nm:13s}", " ".join(f"{k}: {mm:.3f} (+2SE {mm + uu:.3f})" for (k, _), mm, uu in zip(ROWS, m, up)))
    ax.axhline(1.0, color=S.GRAY, ls="--", lw=2.5, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in ROWS])
    ax.set_xlim(-0.5, len(ROWS) - 0.5)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_ylabel("Energy / Budget")
    H.legend(ax, loc="upper left", ncol=2, columnspacing=0.4, handlelength=1.0, handletextpad=0.25)
    H.save(fig, "energy_device_cmp")


main()
