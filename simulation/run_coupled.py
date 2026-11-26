import os, sys, time, argparse
import numpy as np
from math import log, ceil
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ciara_core import CAPS, NOUT, persistence_slack
from ciara_scenarios import design
from ciara_radius import eb_intervals, eb_level
from ciara_round_radius import c1_intervals
from ciara_coupled import CoupledSystem, simulate_coupled, estimate_zeta

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("CIARA_RUNS", "runs"))
DELTA, EPS, L_DEP, MC = 0.05, 0.5, 400, 24
SCENS, TS, NS = ["K3X1", "small", "null", "trap"], (3, 5, 8), (5, 6)
ZETA_TARGETS = (0.0, 0.01, 0.03)
EPOCH_PER_LOOK, NL = 720, 40
DEC = (15, 27, 39)
RADII = ("NAIVE", "C1P", "C1R")
ZMODES = ("T", "E", "Z")
STAGE1_DAYS = 3
NONINC_MIN = 0.01


def looks(l):
    return np.array([(EPOCH_PER_LOOK * j) // l for j in range(1, NL + 1)])


def gammas_for(t, N):
    cs = CoupledSystem("K3X1", t, N, 1.0)
    zp, _ = cs.zeta(step=0.05)
    unit = zp[1]
    return [z / unit for z in ZETA_TARGETS]


def lp3_rows(U, Cup, cap, slack):
    B = U.shape[0]
    nc = Cup.shape[1]
    nl = 3 + nc
    Nn = np.zeros((B, nl, 3))
    d = np.zeros((B, nl))
    for i in range(3):
        Nn[:, i, i] = 1.0
    Nn[:, 3:, :] = Cup
    d[:, 3:] = cap - slack
    pairs = [(p, q) for p in range(nl) for q in range(p + 1, nl)]
    A = np.zeros((B, len(pairs), 3, 3))
    rhs = np.zeros((B, len(pairs), 3))
    A[:, :, 0, :] = 1.0
    rhs[:, :, 0] = 1.0
    for j, (p, q) in enumerate(pairs):
        A[:, j, 1, :] = Nn[:, p]
        A[:, j, 2, :] = Nn[:, q]
        rhs[:, j, 1] = d[:, p]
        rhs[:, j, 2] = d[:, q]
    det = np.linalg.det(A)
    bad = np.abs(det) < 1e-12
    A[bad] = np.eye(3)
    sol = np.linalg.solve(A, rhs[..., None])[..., 0]
    ok = ~bad
    ok &= (sol >= -1e-9).all(-1)
    viol = np.einsum("bkc,bpc->bpk", Cup, sol) + slack - cap[:, None, :]
    ok &= (viol <= 1e-9).all(-1)
    val = np.where(ok, np.einsum("bc,bpc->bp", U, sol), -np.inf)
    jbest = val.argmax(1)
    feas = ok.any(1)
    pi = np.clip(sol[np.arange(B), jbest], 0, None)
    pi /= pi.sum(1, keepdims=True)
    pi[~feas] = np.array([1.0, 0.0, 0.0])
    return pi, feas


def decide(mh, rad, zeta_rows, caps, eta, cs):
    R = mh.shape[0]
    m = mh[:, DEC]
    r = np.where(np.isfinite(rad[:, DEC]), rad[:, DEC], 1.0)
    nd = len(DEC)
    Ulo = np.clip(m[:, :, 0, :] - r[:, :, 0, :], 0, 1)
    Cup = np.clip(m[:, :, 1:, :] + r[:, :, 1:, :], 0, 1)
    cap = np.broadcast_to(np.asarray(caps)[None, None, :] - zeta_rows[:, None, 1:], (R, nd, 3))
    Bt = R * nd
    pi, feas = lp3_rows(Ulo.reshape(Bt, 3), Cup.reshape(Bt, 3, 3), cap.reshape(Bt, 3), eta)
    pi = pi.reshape(R, nd, 3)
    feas = feas.reshape(R, nd)
    dlb = (pi[..., 1:] * ((m[:, :, 0, 1:] - r[:, :, 0, 1:]) - (m[:, :, 0, :1] + r[:, :, 0, :1]))).sum(-1) - eta - 2.0 * zeta_rows[:, None, 0]
    switch = feas & ((1 - pi[..., 0]) >= NONINC_MIN) & (dlb > 0)
    J = cs.policy_values(pi.reshape(Bt, 3)).reshape(R, nd, NOUT)
    return dict(pi=pi.astype(np.float32), feas=feas, switch=switch, J=J)


def truth(scen, t, N, gam_list):
    caps = np.array(CAPS)
    g = np.arange(0, 1.0001, 0.01)
    grid = np.array([[i / 100, j / 100, 1 - i / 100 - j / 100] for i in range(101) for j in range(101 - i)])
    out = {k: [] for k in ("gamma", "mu_tilde", "mu_rw", "zeta_P", "zeta_R", "J_inc", "J_or", "pi_or", "eta")}
    for gm in gam_list:
        cs = CoupledSystem(scen, t, N, gm)
        cs.check_range()
        zp, zr = cs.zeta(step=0.02)
        Jg = cs.policy_values(grid)
        feas = (Jg[:, 1:] <= caps[None, :] + 1e-12).all(1)
        i_or = np.argmax(np.where(feas, Jg[:, 0], -np.inf))
        out["gamma"].append(gm)
        out["mu_tilde"].append(cs.level_pair(cs.q))
        out["mu_rw"].append(cs.level_round(cs.q))
        out["zeta_P"].append(zp)
        out["zeta_R"].append(zr)
        out["J_inc"].append(cs.policy_values(np.array([[1.0, 0.0, 0.0]]))[0])
        out["J_or"].append(Jg[i_or])
        out["pi_or"].append(grid[i_or])
        out["eta"].append(persistence_slack(t, L_DEP))
    return {("truth_" + k): np.asarray(v) for k, v in out.items()}


def run_chunk(task):
    scen, t, N, chunk, R, seed, gam = task
    l, b = design(t)
    nb = int(looks(l)[-1])
    lk = looks(l)
    k7 = int(lk[27])
    assert k7 == 20160 // l
    css = [CoupledSystem(scen, t, N, g) for g in gam]
    cs0 = css[0]
    q = cs0.q
    caps = np.array(CAPS)
    eta = persistence_slack(t, L_DEP)
    L = eb_level(MC, DELTA, 1)
    tr = truth(scen, t, N, gam) if chunk == 0 else None
    Z, Fs = simulate_coupled(cs0, q, l, b, nb, R, gam, seed)
    nb0 = int(STAGE1_DAYS * 2880 // l)
    Z0, F0s = simulate_coupled(cs0, np.full(3, 1.0 / 3.0), l, b, nb0, R, gam, seed + 777)
    dmax = max(cs0.cl.max() - float(q @ cs0.cl), float(q @ cs0.cl) - cs0.cl.min())
    res = {}
    for gi, g in enumerate(gam):
        cs = css[gi]
        F = Fs[gi]
        zhat = estimate_zeta(Z0, F0s[gi], cs.cl, N) * dmax
        mt, mr = cs.level_pair(q), cs.level_round(q)
        zp, zr = cs.zeta(step=0.05)
        zp = np.maximum(zp, 0)
        resid = F[..., 1].astype(np.float64) - mt[1][Z.astype(int)]
        s1 = resid.sum(1)
        cross = ((s1 ** 2) - (resid ** 2).sum(1)).sum() / (R * nb * N * (N - 1))
        res[f"g{gi}_rho_c"] = np.full(R, cross / max((resid ** 2).mean(), 1e-12))
        res[f"g{gi}_zhat"] = zhat.astype(np.float32)
        for mode in RADII:
            if mode == "NAIVE":
                n_plan = N * k7 * q
                cen, rad, _ = eb_intervals(Z, F, N, l, b, t, L, lk, n_plan)
                tgt = mt
            elif mode == "C1P":
                cen, rad, _ = c1_intervals(Z, F, N, l, b, t, L, lk, k7, q, "P")
                tgt = mt
            else:
                cen, rad, _ = c1_intervals(Z, F, N, l, b, t, L, lk, k7, q, "R")
                tgt = mr
            radf = np.where(np.isfinite(rad), rad, 1.0)
            z = (np.abs(cen - tgt[None, None]) / radf).max((2, 3))
            res[f"g{gi}_{mode}_z"] = z.astype(np.float32)
            res[f"g{gi}_{mode}_rad"] = rad[:, DEC].astype(np.float32)
            mh = np.clip(cen, 0, 1)
            zT = np.broadcast_to((zr if mode == "C1R" else zp)[None], (R, NOUT))
            for zm, zrows in (("T", zT), ("E", zhat), ("Z", np.zeros((R, NOUT)))):
                d = decide(mh, rad, zrows, caps, eta, cs)
                for k_, v in d.items():
                    res[f"g{gi}_{mode}_{zm}_{k_}"] = v
    return scen, t, N, chunk, res, tr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--scen", nargs="+", default=SCENS)
    ap.add_argument("--t", nargs="+", type=int, default=list(TS))
    ap.add_argument("--N", nargs="+", type=int, default=list(NS))
    ap.add_argument("--out", default=os.path.join(RUNS, "coupled"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gam_tab = {(t, N): gammas_for(t, N) for t in a.t for N in a.N}
    print("gamma levels", {k: [round(x, 4) for x in v] for k, v in gam_tab.items()}, flush=True)
    tasks = []
    for scen in a.scen:
        si = SCENS.index(scen)
        for t in a.t:
            for N in a.N:
                for c in range(a.R // a.chunk):
                    tasks.append((scen, t, N, c, a.chunk, 1000003 * (si + 1) + 1009 * t + 211 * N + 17 * c + 5, gam_tab[(t, N)]))
    t0 = time.time()
    store, trs = {}, {}
    with Pool(a.procs) as p:
        for scen, t, N, c, res, tr in p.imap_unordered(run_chunk, tasks):
            store.setdefault((scen, t, N), {})[c] = res
            if tr is not None:
                trs[(scen, t, N)] = tr
            print(f"done {scen} t={t} N={N} chunk={c} at {time.time()-t0:.0f}s", flush=True)
    for (scen, t, N), d in store.items():
        keys = d[0].keys()
        out = {k: np.concatenate([d[c][k] for c in sorted(d)], axis=0) for k in keys}
        out.update(trs[(scen, t, N)])
        np.savez_compressed(os.path.join(a.out, f"raw_{scen}_t{t}_N{N}.npz"), **out)
    print("all done %.0f s" % (time.time() - t0), flush=True)
