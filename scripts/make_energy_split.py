import numpy as np

import house as H

S = H.S
PARTS = (("edge_idle_j_per_req", "Edge idle", S.TEAL, "xxx"), ("edge_active_j_per_req", "Edge active", S.TEAL, ""),
         ("cloud_idle_j_per_req", "Cloud idle", S.BLUE, "xxx"), ("cloud_active_j_per_req", "Cloud active", S.BLUE, ""))


def main():
    plt = H.setup()
    srv = {r["unit"]: r for r in H.read("energy_serving_by_arm.csv") if r["run"] == "e1e5"}
    fig, ax = H.new(plt)
    names = [n for n, _ in H.CONTROLLERS]
    y = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    for col, lab, c, h in PARTS:
        v = np.array([H.fnum(srv[arm][col]) for _, arm in H.CONTROLLERS])
        ax.barh(y, v, 0.66, left=left, color=c, hatch=h, zorder=3, edgecolor="black", linewidth=1.2, alpha=0.95 if h else 0.55)
        left += v
    for yi, tot, (n, _) in zip(y, left, H.CONTROLLERS):
        ax.annotate(f"{tot:.0f}", xy=(tot, yi), xytext=(4, 0), textcoords="offset points", va="center", ha="left", fontsize=H.FS, fontweight="normal",
                    color=S.RED if n == "CIARA" else "#222222")
        print(f"{n:13s} total serving J/request {tot:.1f}")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    for t in ax.get_yticklabels():
        if t.get_text() == "CIARA":
            t.set_color(S.RED)
    ax.set_xlim(0, 1050)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Energy per Request (J)")
    ax.set_xticks([0, 200, 400, 600])
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=S.TEAL, alpha=0.75, edgecolor="black", label="Edge"), Patch(facecolor=S.BLUE, alpha=0.75, edgecolor="black", label="Cloud"),
          Patch(facecolor="white", hatch="xxx", edgecolor="black", label="Idle")]
    H.legend(ax, handles=hs, loc="upper right", ncol=1, handlelength=1.2)
    H.save(fig, "energy_split")


main()
