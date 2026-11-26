import house as H

S = H.S


def plot(plt, rows, series, ylabel, name, ylim):
    pts = sorted((int(r["epoch"]), H.fnum(r["mean_deviation"]), H.fnum(r["se"])) for r in rows if r["group"] == "pooled" and r["series"] == series)
    x = [p[0] for p in pts]
    m = [p[1] for p in pts]
    lo = [p[1] - 2 * p[2] for p in pts]
    hi = [p[1] + 2 * p[2] for p in pts]
    fig, ax = H.new(plt)
    ax.axvspan(-0.5, 24.5, color="#bbbbbb", alpha=0.25, lw=0, zorder=0)
    ax.annotate("Burn-in", xy=(16, 0.97), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=H.FS, fontweight="normal", color="#555555")
    ax.fill_between(x, lo, hi, color=S.RED, alpha=0.2, lw=0, zorder=2, label="Two standard errors")
    ax.plot(x, m, color=S.RED, lw=3.0, marker="o", markersize=6, zorder=3, label="Mean deviation")
    ax.axhline(0, color="black", lw=1.5, zorder=1)
    ax.axvline(5, color=S.BLUE, ls="--", lw=2.5, zorder=1)
    ax.annotate("$\\bar{t}=5$", xy=(5, 0.97), xycoords=("data", "axes fraction"), xytext=(5, 0), textcoords="offset points", ha="left", va="top", fontsize=H.FS,
                fontweight="normal", color=S.BLUE)
    ax.set_xlim(-0.5, 29.5)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Epochs after the Switch")
    ax.set_ylabel(ylabel)
    H.legend(ax, loc="lower right")
    print(series, "largest |mean| after epoch 5:", round(max(abs(v) for e, v in zip(x, m) if e > 5), 4))
    H.save(fig, name)


def main():
    plt = H.setup()
    rows = H.read("campaign_settling_curves.csv")
    plot(plt, rows, "cloud_share", "Deviation", "settle_cloud", (-0.075, 0.055))
    plot(plt, rows, "tail_f1", "Deviation", "settle_tail1", (-0.05, 0.035))


main()
