import csv
import os

import numpy as np

import house as H

S = H.S
COST = ("Y_L_f1", "Y_L_f2", "Y_D", "Y_E_A", "Y_E_B", "Y_E_C", "Y_C")
ARMS = (("CIARA", "CIARA", True), ("Incumbent", "I1-E", True), ("I1-S", "I1-S", True), ("I1-Oracle", "I1-oracle", True), ("Mix-or-Split", "I3", True), ("RACE", "I2", True),
        ("RecServe", "RecServe", True), ("RecServe-ft", "RecServe-ft", False), ("Bandit", "CB", True), ("Inc.+cascade", "I1-E|c32", False), ("Inc.+priority", "I1-E|f1prio", False))
COLOR_EXTRA = {"RecServe-ft": S.GRAY, "Inc.+cascade": S.BLUE, "Inc.+priority": S.BLUE}
MARK_EXTRA = {"RecServe-ft": "^", "Inc.+cascade": "s", "Inc.+priority": "s"}
LAB = {"CIARA": (0, -21, "center")}


DIRECT = {"CIARA": (0, 14, "center", "bottom"), "Bandit": (0, 12, "center", "bottom"), "RecServe": (11, 0, "left", "center"), "RACE": (0, 12, "center", "bottom")}


def worst(e, arm):
    return max((e[(arm, k)][0] + 2 * e[(arm, k)][1]) / H.BUDGETS[k] for k in COST)


def plane(plt, e):
    qb = next(H.fnum(r["quality_cost_bound"]) for r in H.read("campaign_looks.csv") if r["look_day"] == "7" and r["band"] == "planned")
    q0 = e[("I1-E", "Y_Q")][0]
    fig, ax = H.new(plt)
    ax.fill_between([0.5, 1.0], -0.045, qb, color="#2CA02C", alpha=0.10, lw=0, zorder=0)
    ax.axvline(1.0, color=S.GRAY, ls="--", lw=2.5, zorder=1)
    ax.axhline(qb, color=S.GRAY, ls=":", lw=2.5, zorder=1)
    ax.annotate("Certified", xy=(0.52, -0.035), xycoords="data", ha="left", va="bottom", fontsize=H.FS, color="#2CA02C")
    ax.annotate(f"Certified bound {qb:.3f}", xy=(6.0, qb), xycoords="data", xytext=(-3, 3), textcoords="offset points", ha="right", va="bottom", fontsize=H.FS,
                color=S.GRAY, zorder=6, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.9))
    pts = {}
    for name, arm, filled in ARMS:
        if name in ("RecServe-ft", "Inc.+priority", "Inc.+cascade"):
            continue
        x, (q, qse) = worst(e, arm), e[(arm, "Y_Q")]
        y = q0 - q
        pts[name] = (x, y)
        col = S.COLOR.get(name, COLOR_EXTRA.get(name))
        mk = H.MARK.get(name, MARK_EXTRA.get(name))
        big = name == "CIARA"
        ax.plot(x, y, mk, markersize=19 if big else 14, markerfacecolor=col if filled else "white", markeredgecolor=col if not filled else "black",
                markeredgewidth=2.4 if not filled else 1.5, zorder=5 if big else 4, linestyle="none", label=name)
        if name in DIRECT:
            dx, dy, ha, va = DIRECT[name]
            ax.annotate(name, xy=(x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=H.FS, color=S.RED if big else "#222222", zorder=6)
    cl = [pts[n] for n in ("Incumbent", "I1-S", "I1-Oracle", "Mix-or-Split")]
    ax.annotate("Incumbent,\nselectors", xy=(sum(p[0] for p in cl) / 4, min(p[1] for p in cl)), xytext=(0, -12), textcoords="offset points", ha="center", va="top",
                fontsize=H.FS, linespacing=1.0, zorder=6)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2", "4"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.5, 6.0)
    ax.set_ylim(-0.045, 0.105)
    ax.set_xlabel("Worst Level / Budget")
    ax.set_ylabel("Quality Cost")
    for n, (x, y) in pts.items():
        print(f"{n:14s} worst {x:.3f} quality cost {y:+.4f}")
    H.save(fig, "cmp_plane")


def ablation(plt, e):
    off = {}
    for r in H.read("e5off_realized.csv"):
        off[(r["policy"], r["outcome"])] = (H.fnum(r["level"]), H.fnum(r["se"]))
    fifo = {r["path"]: r["value"] for r in H.read("fifoonly_result.csv")}
    inc_ref = {k: H.fnum(fifo.get(f"policies/1/levels/{k}")) for k in COST}
    rows = [("CIARA", "comparison run", worst(e, "CIARA")),
            ("Plug-in program", "offline, reference levels", max((off[("plug_in_lp", k)][0] + 2 * off[("plug_in_lp", k)][1]) / H.BUDGETS[k] for k in COST)),
            ("FIFO only", "offline, reference levels (incumbent kept)", max(inc_ref[k] / H.BUDGETS[k] for k in COST)),
            ("Inc.+cascade", "comparison run", worst(e, "I1-E|c32")),
            ("Inc.+priority", "comparison run", worst(e, "I1-E|f1prio"))]
    os.makedirs(H.TABS, exist_ok=True)
    with open(os.path.join(H.TABS, "ablation_rows.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "source", "worst_level_over_budget"])
        for n, src, v in rows:
            w.writerow([n, src, f"{v:.6g}"])
    fig, ax = H.new(plt)
    y = np.arange(len(rows))[::-1]
    for yi, (n, src, v) in zip(y, rows):
        is_c = n == "CIARA"
        ax.barh(yi, v, 0.62, color=S.RED if is_c else S.BLUE, hatch="" if is_c else "///", zorder=3, **S.BAR)
        ax.annotate(f"{v:.2f}", xy=(max(v, 1.0), yi), xytext=(6, 0), textcoords="offset points", va="center", ha="left", fontsize=H.FS, fontweight="normal",
                    color=S.RED if is_c else "#222222")
        print(f"{n:16s} {v:.3f} ({src})")
    ax.axvline(1.0, color=S.GRAY, ls="--", lw=2.5, zorder=1)
    ax.annotate("Budget", xy=(1.0, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -3), textcoords="offset points", ha="left", va="top", color=S.GRAY,
                fontweight="normal", fontsize=H.FS)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in rows])
    ax.set_xlim(0, 2.1)
    ax.set_ylim(-0.6, len(rows) + 0.1)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Worst Level / Budget")
    H.save(fig, "cmp_ablation")


def main():
    plt = H.setup()
    e = {(r["arm"], r["outcome"]): (H.fnum(r["mean"]), H.fnum(r["se"])) for r in H.read("e1e5_estimates.csv")}
    H.S.COLOR.setdefault("Inc.+cascade", S.BLUE)
    plane(plt, e)
    ablation(plt, e)


main()
