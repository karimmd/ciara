import house as H

S = H.S


def plot(plt, rows, col, ylabel, name, ymax, budget):
    fig, ax = H.new(plt)
    ax.axvspan(10, 30, color=S.RED, alpha=0.07, lw=0, zorder=0)
    for pol, nm, mk in (("incumbent", "Incumbent", "s"), ("certified", "CIARA", "o")):
        for rnd in sorted({int(r["round"]) for r in rows if r["policy"] == pol}):
            pts = sorted((H.fnum(r["hour_mid"]), H.fnum(r[col])) for r in rows if r["policy"] == pol and int(r["round"]) == rnd)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=S.COLOR[nm], lw=2.5, marker=mk, markersize=7, markeredgecolor="black", markeredgewidth=0.8,
                    zorder=3, label=nm if rnd in (0, 1) else None)
    if col == "Y_C":
        est = {r["policy"]: H.fnum(r["mean"]) for r in H.read("e4_estimates.csv") if r["outcome"] == "Y_C"}
        box = dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85)
        for pol, nm, spans, xl, dy, va in (("C", "CIARA", ((10, 30),), 20, -18, "top"), ("I", "Incumbent", ((0, 10), (30, 40)), 5, -14, "top")):
            for a, b in spans:
                ax.plot([a, b], [est[pol]] * 2, color=S.COLOR[nm], ls="--", lw=2.0, zorder=2)
            ax.annotate(f"Mean {est[pol]:.3f}", xy=(xl, est[pol]), xytext=(0, dy), textcoords="offset points", ha="center", va=va, fontsize=H.FS, color=S.COLOR[nm],
                        zorder=8, bbox=box)
        print("trace_cloud: policy-level means", est)
    ax.axhline(budget, color=S.GRAY, ls="--", lw=2.5, zorder=1)
    ax.annotate("Budget", xy=(1.0, budget), xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points", ha="right", va="bottom", color=S.GRAY,
                fontweight="normal", fontsize=H.FS)
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Hours of Deployment")
    ax.set_ylabel(ylabel)
    H.legend(ax, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    over = sum(1 for r in rows if r["policy"] == "certified" and H.fnum(r[col]) > budget)
    print(col, "certified hours above the budget:", over, "of", sum(1 for r in rows if r["policy"] == "certified"))
    H.save(fig, name)


def quality(plt, rows):
    con = H.read("e4_contrast.csv")[0]
    bound, cost = H.fnum(con["certified_cost_bound"]), H.fnum(con["measured_cost"])
    q = {pol: [(H.fnum(r["Y_Q"]), int(r["requests_q"])) for r in rows if r["policy"] == pol] for pol in ("incumbent", "certified")}
    mean = {pol: sum(v * n for v, n in xs) / sum(n for _, n in xs) for pol, xs in q.items()}
    fig, ax = H.new(plt)
    ax.axvspan(10, 30, color=S.RED, alpha=0.07, lw=0, zorder=0)
    lo = mean["incumbent"] - bound
    ax.axhspan(lo, mean["incumbent"], color=S.GREEN, alpha=0.10, lw=0, zorder=0)
    ax.axhline(mean["incumbent"], color=S.BLUE, ls="--", lw=2.0, zorder=1)
    ax.axhline(lo, color=S.GRAY, ls="--", lw=2.5, zorder=1)
    ax.annotate(f"Certified cost bound {bound:.3f}", xy=(1.0, lo), xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points", ha="right", va="bottom",
                color=S.GRAY, fontsize=H.FS)
    for pol, nm, mk in (("incumbent", "Incumbent", "s"), ("certified", "CIARA", "o")):
        for rnd in sorted({int(r["round"]) for r in rows if r["policy"] == pol}):
            pts = sorted((H.fnum(r["hour_mid"]), H.fnum(r["Y_Q"])) for r in rows if r["policy"] == pol and int(r["round"]) == rnd)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=S.COLOR[nm], lw=2.5, marker=mk, markersize=7, markeredgecolor="black", markeredgewidth=0.8,
                    zorder=3, label=nm if rnd in (0, 1) else None)
    ax.annotate(f"Measured cost {cost:.3f}", xy=(20, mean["certified"]), xytext=(0, -16), textcoords="offset points", ha="center", va="top", color=S.RED,
                fontsize=H.FS, zorder=8, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_ylim(0.58, 0.78)
    ax.set_yticks([0.6, 0.65, 0.7, 0.75])
    ax.set_xlabel("Hours of Deployment")
    ax.set_ylabel("Quality")
    H.legend(ax, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    inside = sum(1 for v, _ in q["certified"] if v >= lo)
    print(f"trace_quality: incumbent mean {mean['incumbent']:.4f}, certified mean {mean['certified']:.4f}, band lower edge {lo:.4f}, certified hours inside the band {inside} of "
          f"{len(q['certified'])}, lowest certified hour {min(v for v, _ in q['certified']):.4f}")
    H.save(fig, "trace_quality")


def main():
    plt = H.setup()
    rows = H.read("e4_timeseries.csv")
    quality(plt, rows)
    plot(plt, rows, "Y_C", "Cloud Share", "trace_cloud", 0.82, 0.30)
    plot(plt, rows, "Y_L_f1", "Family 1 Tail", "trace_tail1", 0.25, 0.10)


main()
