import os, sys, csv, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ciara_core import CAPS, NOUT
from clopper_pearson import cp_upper

SCENS, TS = ["K3X1", "small", "null", "trap"], (3, 5, 8)
ZM, DAYS, TOL = ("T", "E", "Z"), np.array([4.0, 7.0, 10.0]), 1e-6
OUTN = ("quality", "exceed", "deadline", "cloud")


def evaluate(d, gi, zm, which):
    sw, J = d[f"g{gi}_BAL1_{zm}_switch"], d[f"g{gi}_BAL1_{zm}_J"]
    R = sw.shape[0]
    Jinc, Jor = d[f"truth_J_inc_g{gi}"], d[f"truth_J_or_g{gi}"]
    if which == "campaign":
        use = sw.any(1)
        look = np.where(use, sw.argmax(1), 2)
    else:
        look = np.full(R, which)
        use = sw[:, which]
    Jd = np.where(use[:, None], J[np.arange(R), look], Jinc[None, :])
    viol = (Jd[:, 1:] > np.array(CAPS)[None, :] + TOL).any(1)
    worse = Jd[:, 0] < Jinc[0] - TOL
    fc = use & (viol | worse)
    og = Jor[0] - Jinc[0]
    frac = (Jd[:, 0] - Jinc[0]) / og if og >= 0.01 else np.full(R, np.nan)
    x = int(fc.sum())
    return dict(fc=x, fc_ub=cp_upper(x, R), switch=float(use.mean()), frac=float(np.nanmean(frac)) if og >= 0.01 else float("nan"),
                stop_day=float(DAYS[look].mean()), R=R)


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(raw, out):
    os.makedirs(out, exist_ok=True)
    rows, wrows, zrows = [], [], []
    for s in SCENS:
        for t in TS:
            p = os.path.join(raw, f"raw_{s}_t{t}.npz")
            if not os.path.exists(p):
                continue
            d = dict(np.load(p))
            for gi in range(3):
                z = d[f"g{gi}_BAL1_z"]
                R = z.shape[0]
                m1 = int((z > 1).any(1).sum())
                m1d = int((z[:, [15, 27, 39]] > 1).any(1).sum())
                m2 = int((d[f"g{gi}_BAL2_z"] > 1).any(1).sum())
                zT = d[f"truth_zeta_g{gi}"]
                zh = d[f"g{gi}_zhat"]
                zrows.append(dict(scen=s, t_mix=t, level=gi, zeta_true_exceed=float(zT[1]), zeta_true_quality=float(zT[0]),
                                  zhat_exceed_mean=float(zh[:, 1].mean()), frac_zhat_below_true=float((zh[:, 1] < zT[1]).mean())))
                rb, r2, ri = d[f"g{gi}_BAL1_rad"], d[f"g{gi}_BAL2_rad"], d[f"g{gi}_IID2_rad"]
                med = lambda r: np.median(np.where(np.isfinite(r), r, np.nan), axis=0)
                for li, day in enumerate((4, 7, 10)):
                    for o in range(NOUT):
                        for a in range(3):
                            wrows.append(dict(scen=s, t_mix=t, level=gi, day=day, outcome=OUTN[o], arm=a,
                                              r_bal_onesided=float(med(rb)[li, o, a]), r_bal_twosided=float(med(r2)[li, o, a]),
                                              r_iid_twosided=float(med(ri)[li, o, a]),
                                              ratio_bal1_to_iid2=float(med(rb)[li, o, a] / med(ri)[li, o, a])))
                for zm in ZM:
                    for which, name in (("campaign", "campaign"), (0, "day4"), (1, "day7"), (2, "day10")):
                        ev = evaluate(d, gi, zm, which)
                        rows.append(dict(scen=s, t_mix=t, level=gi, zeta_exceed=float(zT[1]), zmode=zm, design=name, **ev,
                                         miss_any=m1, miss_any_ub=cp_upper(m1, R), miss_dec=m1d, miss_two_sided_diag=m2))
    write(os.path.join(out, "summary.csv"), rows)
    write(os.path.join(out, "width.csv"), wrows)
    write(os.path.join(out, "zeta.csv"), zrows)
    prow = []
    for t in TS:
        p = os.path.join(raw, f"raw_probe_t{t}.npz")
        if os.path.exists(p):
            d = dict(np.load(p))
            for gi, g in enumerate((0.0, 0.2, 0.4)):
                z = d[f"g{gi}_z"]
                m = int((z > 1).any(1).sum())
                prow.append(dict(t_mix=t, gamma=g, R=z.shape[0], misses=m, miss_ub=cp_upper(m, z.shape[0]),
                                 median_radius_day10_cloud_a1=float(np.median(d[f'g{gi}_rad'][:, 2, 3, 1]))))
    if prow:
        write(os.path.join(out, "probe.csv"), prow)
    if os.path.exists(os.path.join(raw, "stress_balanced.csv")):
        shutil.copy(os.path.join(raw, "stress_balanced.csv"), out)
    T = [r for r in rows if r["zmode"] == "T"]
    fc_ub = max(r["fc_ub"] for r in T)
    fc_tot = sum(r["fc"] for r in T if r["design"] == "campaign")
    n_camp = sum(r["R"] for r in T if r["design"] == "campaign")
    seen = {(r["scen"], r["t_mix"], r["level"]): r for r in T}
    miss_ub = max(r["miss_any_ub"] for r in seen.values())
    miss_tot = sum(r["miss_any"] for r in seen.values())
    n_cfg = len(seen)
    n_rep = sum(r["R"] for r in seen.values())
    E = [r for r in rows if r["zmode"] == "E" and r["design"] == "campaign"]
    Z = [r for r in rows if r["zmode"] == "Z" and r["design"] == "campaign"]
    pm = sum(r["misses"] for r in prow)
    pn = sum(r["R"] for r in prow)
    pub = max([r["miss_ub"] for r in prow] or [float("nan")])
    st = []
    if os.path.exists(os.path.join(out, "stress_balanced.csv")):
        st = list(csv.DictReader(open(os.path.join(out, "stress_balanced.csv"))))
    wr = [r for r in wrows if r["scen"] == "K3X1" and r["day"] == 7 and r["outcome"] != "quality"]
    lines = ["# Balanced allocation and one-sided bounds: simulation tables", "",
             f"**Campaign composition (2,1,1), N = 4:** {n_cfg} configurations (4 scenarios x 3 t_mix x 3 coupling levels), {n_rep} replicated campaigns in all.", "",
             "**Verdict inputs (zeta true, decision rule of the campaign, looks day 4, 7, 10):**", "",
             f"* Interval coverage, one-sided events of the certificate and the gate, any look: {miss_tot} misses in {n_rep} replicates; largest per-configuration Clopper-Pearson 95% upper bound {miss_ub:.4f} (criterion 0.05).",
             f"* False certification (campaign design, zeta true): {fc_tot} in {n_camp} campaigns; largest per-configuration upper bound over all designs {fc_ub:.4f} (criterion 0.05).",
             f"* With the Stage 1 estimate of zeta (E): {sum(r['fc'] for r in E)} false certifications; with zeta = 0 (Z): {sum(r['fc'] for r in Z)} (reference only; the coupling slack is part of the decision rule).",
             f"* Probe composition (1,1,1,1), four arms, interval coverage: {pm} misses in {pn} replicates, largest upper bound {pub:.4f}.", ""]
    if st:
        lines += ["**Within-round dependence stress (balanced rounds, target 1/2, outcome-level events, all looks):**", "",
                  "| composition | dependence | replicates | misses, one-sided L | misses, two-sided L |", "|---|---|---|---|---|"]
        for r in st:
            lines.append(f"| {r['composition']} | {r['dependence']} | {r['R']} | {r['miss_one_sided']} ({int(r['miss_one_sided'])/int(r['R']):.4f}) | {r['miss_two_sided']} ({int(r['miss_two_sided'])/int(r['R']):.4f}) |")
        lines.append("")
    if wr:
        rr = np.array([r["ratio_bal1_to_iid2"] for r in wr])
        r2 = np.array([r["r_bal_twosided"] / r["r_iid_twosided"] for r in wr])
        lines += [f"**Width (K3X1, day 7, cost outcomes, all arms and levels):** median radius ratio balanced one-sided to independent draws (two-sided) {np.median(rr):.3f} (range {rr.min():.3f} to {rr.max():.3f}); balanced two-sided to independent draws {np.median(r2):.3f}.", ""]
        for a in range(3):
            x = np.array([r["ratio_bal1_to_iid2"] for r in wr if r["arm"] == a])
            lines.append(f"* arm {a}: median ratio {np.median(x):.3f}")
    open(os.path.join(out, "tables.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
