import os, sys, time, argparse
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ciara_core import CAPS, NOUT, System, ARMS, LOADS, persistence_slack
from ciara_scenarios import design
from ciara_round_radius import c1_intervals
from ciara_coupled import CoupledSystem, simulate_coupled, estimate_zeta
from ciara_balanced import (BalSys, simulate_balanced, side_mask, c1r_balanced, one_sided_violation, one_sided_level,
                            two_sided_level, draw_balanced)
from run_coupled import looks, gammas_for, decide, DEC, DELTA, L_DEP, MC, STAGE1_DAYS, SCENS

N, COMP = 4, (2, 1, 1)
TS = (3, 5, 8)
ZMODES = ("T", "E", "Z")


def truth_bal(cs, bs, comp):
    caps = np.array(CAPS)
    grid = np.array([[i / 100, j / 100, 1 - i / 100 - j / 100] for i in range(101) for j in range(101 - i)])
    Jg = cs.policy_values(grid)
    feas = (Jg[:, 1:] <= caps[None, :] + 1e-12).all(1)
    i_or = np.argmax(np.where(feas, Jg[:, 0], -np.inf))
    mu = bs.level_bal(comp)
    zb = np.zeros(NOUT)
    n = 20
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pi = np.array([i, j, n - i - j]) / n
            zb = np.maximum(zb, np.abs(cs.level_pair(pi) - mu).max(1))
    return dict(mu=mu, zeta=zb, J_inc=cs.policy_values(np.array([[1.0, 0, 0]]))[0], J_or=Jg[i_or], pi_or=grid[i_or])


def run_chunk(task):
    scen, t, chunk, R, seed, gam = task
    l, b = design(t)
    lk = looks(l)
    nb = int(lk[-1])
    k7 = int(lk[27])
    css = [CoupledSystem(scen, t, N, g) for g in gam]
    cs0 = css[0]
    q = cs0.q
    caps = np.array(CAPS)
    eta = persistence_slack(t, L_DEP)
    L1, L2 = one_sided_level(MC, DELTA), two_sided_level(MC, DELTA)
    up = side_mask(3)
    bss = [BalSys(c.sys, N, g, c.t_cloud) for c, g in zip(css, gam)]
    Z, Fs = simulate_balanced(bss[0], COMP, l, b, nb, R, gam, seed)
    Zi, Fi = simulate_coupled(cs0, q, l, b, nb, R, gam, seed + 31)
    nb0 = int(STAGE1_DAYS * 2880 // l)
    Z0, F0s = simulate_coupled(cs0, np.full(3, 1.0 / 3.0), l, b, nb0, R, gam, seed + 777)
    cl = cs0.cl
    comp = np.array(COMP, float)
    e_bal = (comp @ cl - cl) / (N - 1)
    dmax = float(max((cl.max() - e_bal).max(), (e_bal - cl.min()).max()))
    res, tr = {}, {}
    for gi, g in enumerate(gam):
        cs, bs = css[gi], bss[gi]
        T = truth_bal(cs, bs, COMP) if chunk == 0 else None
        if chunk == 0:
            for k_, v in T.items():
                tr[f"truth_{k_}_g{gi}"] = np.asarray(v)
        tb = bs.level_bal(COMP)
        zT = np.zeros(NOUT)
        n = 20
        for i in range(n + 1):
            for j in range(n + 1 - i):
                zT = np.maximum(zT, np.abs(cs.level_pair(np.array([i, j, n - i - j]) / n) - tb).max(1))
        zhat = estimate_zeta(Z0, F0s[gi], cs.cl, N) * dmax
        res[f"g{gi}_zhat"] = zhat.astype(np.float32)
        F = Fs[gi]
        cen, rad, _ = c1r_balanced(Z, F, N, l, b, t, L1, lk, k7, 3)
        z = one_sided_violation(cen, rad, tb, up)
        res[f"g{gi}_BAL1_z"] = z.astype(np.float32)
        res[f"g{gi}_BAL1_rad"] = rad[:, DEC].astype(np.float32)
        _, rad2, _ = c1r_balanced(Z, F, N, l, b, t, L2, lk, k7, 3)
        res[f"g{gi}_BAL2_rad"] = rad2[:, DEC].astype(np.float32)
        zb2 = (np.abs(cen - tb[None, None]) / np.where(np.isfinite(rad2), rad2, 1.0)).max((2, 3))
        res[f"g{gi}_BAL2_z"] = zb2.astype(np.float32)
        ceni, radi, _ = c1_intervals(Zi, Fi[gi], N, l, b, t, L2, lk, k7, q, "R")
        res[f"g{gi}_IID2_rad"] = radi[:, DEC].astype(np.float32)
        mh = np.clip(cen, 0, 1)
        for zm, zrows in (("T", np.broadcast_to(zT[None], (R, NOUT))), ("E", zhat), ("Z", np.zeros((R, NOUT)))):
            d = decide(mh, rad, zrows, caps, eta, cs)
            for k_, v in d.items():
                res[f"g{gi}_BAL1_{zm}_{k_}"] = v
    return scen, t, chunk, res, tr


def probe_system(t):
    arms = ARMS[[1, 5, 6, 6]].copy()
    arms[3, 0] = 0.80
    arms[3, 4] = 0.4
    return System(t_mix=t, arms=arms, loads=LOADS[[1]], safe=[[0, 1, 2, 3]], incumbent=[0])


def run_probe_chunk(task):
    t, chunk, R, seed, gam = task
    l, b = design(t)
    lk = looks(l)
    nb = int(lk[-1])
    k7 = int(lk[27])
    sys_ = probe_system(t)
    L1 = one_sided_level(MC, DELTA)
    up = side_mask(4)
    bss = [BalSys(sys_, 4, g, t) for g in gam]
    comp = (1, 1, 1, 1)
    for bs in bss:
        bs.check_range(comp)
        top = sys_.m.max(axis=(1, 2, 3, 4)) + np.maximum(bs.g, 0) * bs.cl.max()
        assert (top <= 1 + 1e-12).all(), top
    Z, Fs = simulate_balanced(bss[0], comp, l, b, nb, R, gam, seed)
    res = {}
    for gi in range(len(gam)):
        tb = bss[gi].level_bal(comp)
        cen, rad, _ = c1r_balanced(Z, Fs[gi], 4, l, b, t, L1, lk, k7, 4)
        res[f"g{gi}_z"] = one_sided_violation(cen, rad, tb, up).astype(np.float32)
        res[f"g{gi}_rad"] = rad[:, DEC].astype(np.float32)
    return t, chunk, res


def stress_cell(args):
    comp, dep, R, H, seed, M_c = args
    rng = np.random.default_rng(seed)
    K, Nn = len(comp), int(sum(comp))
    l, b = design(5)
    Z = draw_balanced(rng, R, comp, H)
    if dep == "common":
        F = np.broadcast_to(rng.integers(0, 2, size=(R, 1, H, 1)).astype(np.float32), (R, Nn, H, NOUT)).copy()
    elif dep == "half":
        sh = rng.integers(0, 2, size=(R, 1, H, 1)).astype(np.float32)
        own = rng.integers(0, 2, size=(R, Nn, H, NOUT)).astype(np.float32)
        F = np.where(rng.random((R, Nn, H, 1)) < 0.5, sh, own)
    else:
        F = rng.integers(0, 2, size=(R, Nn, H, NOUT)).astype(np.float32)
    lk = np.arange(10, H + 1, 10)
    L1, L2 = one_sided_level(M_c, 0.05), two_sided_level(M_c, 0.05)
    up = side_mask(K)
    tgt = np.full((NOUT, K), 0.5)
    cen, rad, _ = c1r_balanced(Z, F, Nn, l, b, 5, L1, lk, H, K)
    m1 = int((one_sided_violation(cen, rad, tgt, up) > 1).any(1).sum())
    cen2, rad2, _ = c1r_balanced(Z, F, Nn, l, b, 5, L2, lk, H, K)
    m2 = int((np.abs(cen2 - 0.5) > rad2).any((1, 2, 3)).sum())
    return comp, dep, R, m1, m2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--scen", nargs="+", default=SCENS)
    ap.add_argument("--t", nargs="+", type=int, default=list(TS))
    ap.add_argument("--mode", default="campaign")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.mode == "campaign":
        gam_tab = {t: gammas_for(t, N) for t in a.t}
        print("gamma levels", {k: [round(x, 4) for x in v] for k, v in gam_tab.items()}, flush=True)
        tasks = [(s, t, c, a.chunk, 1000003 * (SCENS.index(s) + 1) + 1009 * t + 17 * c + 90001, gam_tab[t])
                 for s in a.scen for t in a.t for c in range(a.R // a.chunk)]
        store, trs = {}, {}
        with Pool(a.procs) as p:
            for s, t, c, res, tr in p.imap_unordered(run_chunk, tasks):
                store.setdefault((s, t), {})[c] = res
                if tr:
                    trs[(s, t)] = tr
                print(f"done {s} t={t} chunk={c} at {time.time()-t0:.0f}s", flush=True)
        for (s, t), d in store.items():
            out = {k: np.concatenate([d[c][k] for c in sorted(d)], axis=0) for k in d[0].keys()}
            out.update(trs[(s, t)])
            np.savez_compressed(os.path.join(a.out, f"raw_{s}_t{t}.npz"), **out)
    elif a.mode == "probe":
        gam = [0.0, 0.2, 0.4]
        tasks = [(t, c, a.chunk, 5000011 + 1009 * t + 17 * c, gam) for t in a.t for c in range(a.R // a.chunk)]
        store = {}
        with Pool(a.procs) as p:
            for t, c, res in p.imap_unordered(run_probe_chunk, tasks):
                store.setdefault(t, {})[c] = res
                print(f"done probe t={t} chunk={c} at {time.time()-t0:.0f}s", flush=True)
        for t, d in store.items():
            out = {k: np.concatenate([d[c][k] for c in sorted(d)], axis=0) for k in d[0].keys()}
            np.savez_compressed(os.path.join(a.out, f"raw_probe_t{t}.npz"), **out)
    else:
        tasks = []
        for comp in ((2, 1, 1), (1, 1, 1, 1)):
            for dep in ("indep", "half", "common"):
                for c in range(a.R // a.chunk):
                    tasks.append((comp, dep, a.chunk, 400, 777 + 31 * c + 7 * len(comp), NOUT * len(comp)))
        agg = {}
        with Pool(a.procs) as p:
            for comp, dep, R, m1, m2 in p.imap_unordered(stress_cell, tasks):
                x = agg.setdefault((comp, dep), [0, 0, 0])
                x[0] += R; x[1] += m1; x[2] += m2
        with open(os.path.join(a.out, "stress_balanced.csv"), "w") as f:
            f.write("composition,dependence,R,miss_one_sided,miss_two_sided\n")
            for (comp, dep), (R, m1, m2) in sorted(agg.items()):
                f.write(f"{'-'.join(map(str, comp))},{dep},{R},{m1},{m2}\n")
    print("all done %.0f s" % (time.time() - t0), flush=True)
