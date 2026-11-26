import house as H

S = H.S
LOOKS = ((230, "Day 4"), (403, "Day 7"))


def load():
    b = {}
    for r in H.read("campaign_trajectory_bounds.csv"):
        if r["band"] == "planned" and r["outcome"] in ("Y_C", "Y_L_f1"):
            b[(int(r["k"]), r["arm"], r["outcome"])] = H.fnum(r["upper_cert"])
    d = {int(r["k"]): r for r in H.read("campaign_trajectory_decision.csv") if r["band"] == "planned"}
    ks = sorted({k for k, _, _ in b})
    return b, d, ks


def policy_bound(b, d, k, outcome):
    r = d[k]
    if r["switch"] != "1":
        return None
    eta = H.fnum(r["eta_dep"])
    return sum(H.fnum(r[f"w_{a}"]) * (b[(k, a, outcome)] - eta) for a in (H.A_INC, H.A_PRIO, H.A_FIFO)) + eta


def decorate(ax, ks, first, ymax, ytext, text="Policy\ncertified"):
    ax.axvspan(first, 415, color="#D62728", alpha=0.08, lw=0, zorder=0)
    if ytext is not None:
        ax.annotate(text, xy=((first + 415) / 2, ytext), xycoords="data", ha="center", va="center", fontsize=H.FS, fontweight="normal", color=S.RED,
                    linespacing=1.0)
    ax.axhline(1.0, color=S.GRAY, ls="--", lw=2.5, zorder=1)
    ax.annotate("Budget", xy=(0.0, 1.0), xycoords=("axes fraction", "data"), xytext=(4, -4), textcoords="offset points", ha="left", va="top", color=S.GRAY,
                fontweight="normal", fontsize=H.FS)
    for k, lab in LOOKS:
        ax.axvline(k, color="#444444", ls=":", lw=2.2, zorder=1)
        ax.annotate(lab, xy=(k, 0.985), xycoords=("data", "axes fraction"), xytext=(-5, 0), textcoords="offset points", ha="right", va="top", fontsize=H.FS,
                    fontweight="normal", color="#444444", linespacing=1.0)
    ax.set_xlim(20, 415)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel("Completed Rounds")


def single(plt, b, d, ks, outcome, ylabel, name, ymax, ytext, legend_kw):
    bud = H.BUDGETS[outcome]
    first = min(k for k in ks if d[k]["switch"] == "1")
    fig, ax = H.new(plt)
    for arm in (H.A_INC, H.A_PRIO, H.A_FIFO):
        lab, col, mk, ls = H.ARM[arm]
        ax.plot(ks, [b[(k, arm, outcome)] / bud for k in ks], color=col, ls=ls, lw=3.2, label=f"{lab} {b[(403, arm, outcome)] / bud:.2f}", zorder=3)
    decorate(ax, ks, first, ymax, ytext, "Certified")
    ax.set_yticks(range(0, int(round(ymax))))
    ax.set_ylabel(ylabel)
    H.legend(ax, **legend_kw)
    H.save(fig, name)


def combined(plt, b, d, ks):
    first = min(k for k in ks if d[k]["switch"] == "1")
    fig, ax = H.new(plt)
    for outcome, ls, lab in (("Y_C", "-", "cloud"), ("Y_L_f1", "--", "tail 1")):
        bud = H.BUDGETS[outcome]
        ax.plot(ks, [b[(k, H.A_INC, outcome)] / bud for k in ks], color=S.BLUE, ls=ls, lw=3.0, label=f"Incumbent, {lab}", zorder=3)
        kp = [k for k in ks if k >= first]
        ax.plot(kp, [policy_bound(b, d, k, outcome) / bud for k in kp], color=S.RED, ls=ls, lw=4.0, label=f"CIARA, {lab}", zorder=5)
    decorate(ax, ks, first, 2.6, 0.45)
    ax.set_ylabel("Certified Bound / Budget")
    H.legend(ax, loc="upper right", bbox_to_anchor=(0.80, 0.80), fontsize=H.FS)
    H.save(fig, "cert_traj_combined")


def main():
    plt = H.setup()
    b, d, ks = load()
    first = min(k for k in ks if d[k]["switch"] == "1")
    for k in (first, 403):
        print("k", k, "policy bound / budget: cloud", round(policy_bound(b, d, k, "Y_C") / 0.30, 4), "tail1", round(policy_bound(b, d, k, "Y_L_f1") / 0.10, 4))
    print("incumbent cloud bound / budget at 230 and 403:", round(b[(230, H.A_INC, "Y_C")] / 0.3, 3), round(b[(403, H.A_INC, "Y_C")] / 0.3, 3))
    single(plt, b, d, ks, "Y_C", "Bound / Budget", "cert_traj_cloud", 5.0, 0.42, {"loc": "upper right", "bbox_to_anchor": (0.995, 0.87)})
    single(plt, b, d, ks, "Y_L_f1", "Bound / Budget", "cert_traj_tail1", 5.6, None, {"loc": "upper right", "bbox_to_anchor": (0.995, 0.87)})


main()


def best(plt):
    rows = sorted(H.read("campaign_trajectory_best.csv"), key=lambda r: int(r["k"]))
    ks = [int(r["k"]) for r in rows]
    fig, ax = H.new(plt)
    ax.axhspan(0, 1.0, color="#2CA02C", alpha=0.08, lw=0, zorder=0)
    ax.axhline(1.0, color=S.GRAY, ls="--", lw=2.5, zorder=1)
    i7 = H.fnum(rows[-1]["incumbent_worst_ratio"])
    ax.plot(ks, [H.fnum(r["incumbent_worst_ratio"]) for r in rows], color=S.BLUE, lw=3.2, label=f"Incumbent {i7:.2f}", zorder=3)
    single = [min(H.fnum(r[f"worst_ratio_{H.A_PRIO}"]), H.fnum(r[f"worst_ratio_{H.A_FIFO}"])) for r in rows]
    ax.plot(ks, single, color=S.ORANGE, ls=":", lw=3.0, label=f"Single config. {single[-1]:.2f}", zorder=3)
    kb = [(k, H.fnum(r["best_mixture_worst_ratio"])) for k, r in zip(ks, rows) if r["best_mixture_worst_ratio"]]
    ax.plot([k for k, _ in kb], [v for _, v in kb], color=S.RED, lw=4.2, label=f"CIARA {kb[-1][1]:.2f}", zorder=4)
    for k, lab in ((230, "Day 4"), (403, "Day 7")):
        ax.axvline(k, color="#444444", ls=":", lw=2.2, zorder=1)
        ax.annotate(lab, xy=(k, 0.03), xycoords=("data", "axes fraction"), xytext=(-5, 0), textcoords="offset points", ha="right", va="bottom", fontsize=H.FS,
                    color="#444444")
    ax.set_xlim(60, 415)
    ax.set_ylim(0.0, 3.5)
    ax.set_xlabel("Completed Rounds")
    ax.set_ylabel("Bound / Budget")
    ax.annotate("Certified", xy=(0.03, 0.17), xycoords="axes fraction", ha="left", va="bottom", fontsize=H.FS, color="#2CA02C")
    H.legend(ax, loc="upper right", handlelength=1.5, bbox_to_anchor=(1.0, 1.0))
    print("best mixture at 230, 290, 403:", [round(v, 3) for k, v in kb if k in (230, 290, 403)], "incumbent at 403:", round(i7, 3))
    H.save(fig, "cert_best")


best(H.setup())
