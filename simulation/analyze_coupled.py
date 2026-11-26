import os, sys, csv, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ciara_core import CAPS, NOUT
from clopper_pearson import cp_upper

SCENS, TS, NS = ["K3X1", "small", "null", "trap"], (3, 5, 8), (5, 6)
RADII, ZMODES = ("NAIVE", "C1P", "C1R"), ("T", "E", "Z")
DEC, DAYS = (15, 27, 39), np.array([4.0, 7.0, 10.0])
TOL = 1e-6
OUTN = ("quality", "exceed", "deadline", "cloud")


def evaluate(d, gi, mode, zm, which):
    sw = d[f"g{gi}_{mode}_{zm}_switch"]
    J = d[f"g{gi}_{mode}_{zm}_J"]
    R = sw.shape[0]
    Jinc, Jor = d["truth_J_inc"][gi], d["truth_J_or"][gi]
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
    return dict(fc=x, fc_ub=cp_upper(x, R), switch=float(use.mean()),
                frac=float(np.nanmean(frac)) if og >= 0.01 else np.nan, stop_day=float(DAYS[look].mean()), og=float(og), R=R)


def main(dirn):
    rows, wrows, zrows = [], [], []
    for scen in SCENS:
        for t in TS:
            for N in NS:
                p = os.path.join(dirn, f"raw_{scen}_t{t}_N{N}.npz")
                if not os.path.exists(p):
                    continue
                d = dict(np.load(p))
                ng = d["truth_gamma"].shape[0]
                for gi in range(ng):
                    zP, zR = d["truth_zeta_P"][gi], d["truth_zeta_R"][gi]
                    zh = d[f"g{gi}_zhat"]
                    zrows.append(dict(scen=scen, t_mix=t, N=N, level=gi, gamma=float(d["truth_gamma"][gi]),
                                      zeta_P_exceed=float(zP[1]), zeta_R_exceed=float(zR[1]), zeta_P_quality=float(zP[0]),
                                      zhat_exceed_mean=float(zh[:, 1].mean()), zhat_exceed_p10=float(np.percentile(zh[:, 1], 10)),
                                      zhat_quality_mean=float(zh[:, 0].mean()), rho_c=float(d[f"g{gi}_rho_c"][0]),
                                      frac_zhat_below_true=float((zh[:, 1] < zP[1]).mean())))
                    for mode in RADII:
                        z = d[f"g{gi}_{mode}_z"]
                        R = z.shape[0]
                        miss_all = int((z > 1).any(1).sum())
                        miss_dec = int((z[:, list(DEC)] > 1).any(1).sum())
                        rad = d[f"g{gi}_{mode}_rad"]
                        rm = np.median(np.where(np.isfinite(rad), rad, np.nan), axis=0)
                        rn = np.median(np.where(np.isfinite(d[f"g{gi}_NAIVE_rad"]), d[f"g{gi}_NAIVE_rad"], np.nan), axis=0)
                        for li, day in enumerate((4, 7, 10)):
                            for o in range(NOUT):
                                for a in range(3):
                                    wrows.append(dict(scen=scen, t_mix=t, N=N, level=gi, radius=mode, day=day, outcome=OUTN[o], arm=a,
                                                      median_radius=float(rm[li, o, a]), ratio_to_naive=float(rm[li, o, a] / rn[li, o, a])))
                        for zm in ZMODES:
                            for which, name in (("campaign", "campaign"), (0, "day4"), (1, "day7"), (2, "day10")):
                                ev = evaluate(d, gi, mode, zm, which)
                                rows.append(dict(scen=scen, t_mix=t, N=N, level=gi, zeta_exceed=float(zP[1]), radius=mode, zmode=zm, design=name, **ev,
                                                 miss_any=miss_all, miss_any_ub=cp_upper(miss_all, R), miss_dec=miss_dec,
                                                 z_med7=float(np.median(z[:, 27])), z_p95_7=float(np.percentile(z[:, 27], 95)), z_max=float(z.max())))
    def write(name, rs):
        with open(os.path.join(dirn, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader()
            for r in rs:
                w.writerow(r)
    write("summary.csv", rows)
    write("width.csv", wrows)
    write("zeta.csv", zrows)
    select(rows, dirn)


def select(rows, dirn):
    idx = {(r["scen"], r["t_mix"], r["N"], r["level"], r["radius"], r["zmode"], r["design"]): r for r in rows}
    out = []
    n_cfg = len({(r["scen"], r["t_mix"], r["N"], r["level"]) for r in rows})
    for mode in RADII:
        sub = [r for r in rows if r["radius"] == mode and r["zmode"] == "T"]
        fc_ub = max(r["fc_ub"] for r in sub)
        miss_ub = max(r["miss_any_ub"] for r in sub)
        fc_tot = sum(r["fc"] for r in sub if r["design"] == "campaign")
        V = fc_ub <= 0.05 and miss_ub <= 0.05
        Uc = [idx[("K3X1", t, N, lv, mode, "T", "campaign")]["frac"] for t in TS for N in NS for lv in (0, 1) if ("K3X1", t, N, lv, mode, "T", "campaign") in idx]
        U = (len(Uc) > 0) and all(x >= 0.5 for x in Uc)
        out.append(dict(radius=mode, configs=n_cfg, max_fc_ub_any_design=fc_ub, max_miss_ub=miss_ub, campaign_false_certs=fc_tot, V=V,
                        U_min_frac_K3X1_levels01=min(Uc) if Uc else np.nan, U=U if mode != "NAIVE" else "n/a"))
    with open(os.path.join(dirn, "selection.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow(r)
    for r in out:
        print(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("CIARA_RUNS", "runs"), "coupled"))
    main(ap.parse_args().dir)
