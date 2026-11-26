import math

import house as H

S = H.S


def curve(rows, unit, fam):
    pts = sorted((H.fnum(r["p"]), r["latency_s"]) for r in rows if r["run"] == "e1e5" and r["unit"] == unit and r["family"] == str(fam))
    xs, ys = [], []
    for p, v in pts:
        if v == "inf" or p is None:
            continue
        xs.append(float(v))
        ys.append(max(1.0 - p, 1e-6))
    return xs, ys


NEUTRAL = {"Mix-or-Split": ("#555555", (0, (6, 1.5, 1.5, 1.5))), "RACE": ("#222222", (0, (5, 2))), "RecServe": ("#777777", (0, (1.2, 1.5))),
           "Bandit": ("#999999", (0, (9, 2.5)))}


def at_target(rows, arm, fam):
    r = next(r for r in rows if r["run"] == "e1e5" and r["unit"] == arm and r["family"] == str(fam))
    return float(r["share_exceed"])


def plot(plt, rows, fam, name, xlim):
    fig, ax = H.new(plt)
    ax.grid(axis="x", visible=True)
    order = ("CIARA", "Incumbent", "Mix-or-Split", "RACE", "RecServe", "Bandit")
    arms = dict(H.CONTROLLERS)
    for nm in reversed(order):
        xs, ys = curve(rows, arms[nm], fam)
        if nm in NEUTRAL:
            c, ls = NEUTRAL[nm]
            ax.step(xs, ys, where="post", color=c, ls=ls, lw=2.6, label=nm, zorder=3)
        else:
            ax.step(xs, ys, where="post", color=S.COLOR[nm], ls="-", lw=4.2 if nm == "CIARA" else 2.8, label=nm, zorder=5 if nm == "CIARA" else 4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    tk = [0.1, 1, 10, 60]
    ax.xaxis.set_major_locator(FixedLocator(tk))
    ax.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10", "60"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(1e-5, 2.0)
    ax.axhline(0.10, color=S.GRAY, ls="--", lw=2.5, zorder=1, label="Budget")
    t = H.TARGET[fam]
    ax.axvline(t, color="black", ls=":", lw=2.5, zorder=1)
    pub = {nm: at_target(rows, arms[nm], fam) for nm in ("Mix-or-Split", "RACE", "RecServe", "Bandit")}
    worst = max(pub, key=pub.get)
    yc, yw = at_target(rows, arms["CIARA"], fam), pub[worst]
    ax.plot([t], [yc], "o", color=S.RED, markersize=12, markeredgecolor="black", markeredgewidth=1.3, zorder=7)
    ax.plot([t], [yw], "o", color=NEUTRAL[worst][0], markersize=12, markeredgecolor="black", markeredgewidth=1.3, zorder=7)
    box = dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.9)
    ax.annotate(f"{yw:.2f}", xy=(t, yw), xytext=(9, 3), textcoords="offset points", ha="left", va="bottom", fontsize=H.FS, color="#222222", zorder=8, bbox=box)
    ax.annotate(f"{yc:.2f}", xy=(t, yc), xytext=(9, -3), textcoords="offset points", ha="left", va="top", fontsize=H.FS, color=S.RED, zorder=8, bbox=box)
    ax.set_xlabel(f"Latency (s), Target {t:.1f}")
    ax.set_ylabel("Share Slower")
    h, lab = ax.get_legend_handles_labels()
    idx = [lab.index(n) for n in order + ("Budget",)]
    H.legend(ax, handles=[h[i] for i in idx], labels=[lab[i] for i in idx], loc="lower left", ncol=1, handlelength=1.6, handletextpad=0.35, borderpad=0.3,
             labelspacing=0.15, borderaxespad=0.25)
    print(f"{name}: at the target CIARA {yc:.4f}, worst published {worst} {yw:.4f}")
    H.save(fig, name)


def main():
    plt = H.setup()
    rows = H.read("latency_quantiles.csv")
    for nm, arm in H.CONTROLLERS:
        for fam in (1, 2):
            r = next(r for r in rows if r["run"] == "e1e5" and r["unit"] == arm and r["family"] == str(fam))
            print(nm, fam, "share exceed", round(float(r["share_exceed"]), 4))
    plot(plt, rows, 1, "ccdf_family1", (0.025, 100))
    plot(plt, rows, 2, "ccdf_family2", (0.025, 100))


main()
