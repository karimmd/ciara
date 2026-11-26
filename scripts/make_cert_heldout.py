import numpy as np

import house as H

S = H.S
ROWS = (("Y_C", "Cloud"), ("Y_L_f1", "Tail 1"), ("Y_L_f2", "Tail 2"), ("Y_D", "Deadline"), ("Y_E_A", "Energy A"), ("Y_E_B", "Energy B"), ("Y_E_C", "Energy C"))


def main():
    plt = H.setup()
    e = {(r["policy"], r["outcome"]): r for r in H.read("e4_estimates.csv") if r["unit"] == "constant"}
    cert = {r["row"]: r for r in H.read("campaign_certificate_rows.csv") if r["look_day"] == "7" and r["band"] == "planned"}
    fig, ax = H.new(plt)
    y = np.arange(len(ROWS))[::-1]
    h = 0.38
    for j, (pol, name) in enumerate((("I", "Incumbent"), ("C", "CIARA"))):
        vals = [H.fnum(e[(pol, k)]["mean"]) / H.BUDGETS[k] for k, _ in ROWS]
        errs = [2 * H.fnum(e[(pol, k)]["se"]) / H.BUDGETS[k] for k, _ in ROWS]
        yy = y + (h / 2 if pol == "I" else -h / 2)
        ax.barh(yy, vals, h, xerr=errs, color=S.COLOR[name], hatch=S.HATCH[name], error_kw=S.ERR, label=name, zorder=3, **S.BAR)
        ub = [H.fnum(cert[k]["incumbent_upper" if pol == "I" else "policy_upper"]) for k, _ in ROWS]
        ax.plot([u / H.BUDGETS[k] if u is not None else np.nan for u, (k, _) in zip(ub, ROWS)], yy, "D", markersize=9, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=1.8, zorder=5, label="Certified bound" if pol == "C" else None)
    ax.axvspan(1.0, 3.7, color="#D62728", alpha=0.06, lw=0, zorder=0)
    ax.axvline(1.0, color=S.GRAY, ls="--", lw=2.5, zorder=1, label="Budget")
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in ROWS])
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_xlim(0, 3.7)
    ax.set_xticks([0, 1, 2, 3])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Held-Out Level / Budget")
    inc_c, cia_c = H.fnum(e[("I", "Y_C")]["mean"]), H.fnum(e[("C", "Y_C")]["mean"])
    print("held-out cloud share: incumbent", round(inc_c, 3), "CIARA", round(cia_c, 3))
    H.legend(ax, loc="lower right", bbox_to_anchor=(1.0, 0.0), handlelength=1.4)
    H.save(fig, "cert_heldout_bounds")


main()
