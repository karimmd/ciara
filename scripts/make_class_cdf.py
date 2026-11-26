import house as H

S = H.S
DEADLINE = {"B": 10.0, "C": 5.0}
RADIO = {"B": "LTE-M", "C": "5G"}


def plot(plt, rows, cls, name):
    fig, ax = H.new(plt)
    ax.grid(axis="x", visible=True)
    miss = {}
    for pol, nm, ls in (("incumbent", "Incumbent", "--"), ("certified", "CIARA", "-")):
        pts = sorted((H.fnum(r["p"]), float(r["latency_s"])) for r in rows if r["run"] == "e4_class" and r["unit"] == f"{pol}|{cls}" and r["family"] == "1")
        ax.plot([v for _, v in pts], [p for p, _ in pts], color=S.COLOR[nm], ls=ls, lw=4.0 if nm == "CIARA" else 3.0, label=nm, zorder=4 if nm == "CIARA" else 3,
                drawstyle="steps-post")
        miss[nm] = next(H.fnum(r["share_exceed"]) for r in rows if r["run"] == "e4_class" and r["unit"] == f"{pol}|{cls}" and r["family"] == "1")
    d = DEADLINE[cls]
    y0 = 0.54 if cls == "B" else 0.06
    ax.axvline(d, color="black", ls=":", lw=2.5, zorder=1)
    ax.annotate(f"Deadline {d:.0f} s", xy=(d, y0 + 0.24), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", ha="left", va="bottom", fontsize=H.FS,
                fontweight="normal")
    ax.annotate(f"Missed: {miss['CIARA'] * 100:.1f}%", xy=(d, y0 + 0.12), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", ha="left",
                va="bottom", fontsize=H.FS, fontweight="normal", color=S.RED)
    ax.annotate(f"Missed: {miss['Incumbent'] * 100:.1f}%", xy=(d, y0), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", ha="left",
                va="bottom", fontsize=H.FS, fontweight="normal", color=S.BLUE)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 30]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2", "5", "10", "30"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.4, 40)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"Latency, Class {cls} ({RADIO[cls]}) (s)")
    ax.set_ylabel("CDF")
    def x_at(nm, q):
        pts = sorted((H.fnum(r["p"]), float(r["latency_s"])) for r in rows if r["run"] == "e4_class" and r["unit"] == f"{dict(Incumbent='incumbent', CIARA='certified')[nm]}|{cls}" and r["family"] == "1")
        return next(v for p, v in pts if p >= q)
    ax.annotate("Incumbent", xy=(x_at("Incumbent", 0.75), 0.75), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=H.FS, color=S.BLUE)
    ax.annotate("CIARA", xy=(x_at("CIARA", 0.30), 0.30), xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=H.FS, color=S.RED)
    print(cls, {k: round(v, 4) for k, v in miss.items()})
    H.save(fig, name)


def main():
    plt = H.setup()
    rows = [r for r in H.read("latency_quantiles.csv") if r["run"] == "e4_class"]
    plot(plt, rows, "B", "cdf_classB")
    plot(plt, rows, "C", "cdf_classC")


main()
