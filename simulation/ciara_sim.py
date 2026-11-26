import numpy as np
from math import log, ceil
from ciara_core import (dm_bias, dm_radius, arm_levels_stationary, System, seg_len, arm_levels, cycle_values, block_ops, solve_policy,
                        radii, carry_bias, persistence_slack, CAPS, NOUT)


def simulate(sys_, eps, l, b, Tmax, R, N, ckpt_blocks, seed, G0=400):
    rng = np.random.default_rng(seed)
    X, K, S, M, D = sys_.X, sys_.K, sys_.S, sys_.M, sys_.D
    q = sys_.propensities(eps)
    cumq = np.cumsum(q, axis=1)
    cumq[:, -1] = 1.0
    G = seg_len(l, D, G0)
    C = R * N
    nblk = Tmax // l
    beta = b / l
    s = np.zeros(C, dtype=int)
    ht = np.zeros((C, X * K, NOUT))
    dm = np.zeros((C, X * K, NOUT))
    dm2 = np.zeros((C, X * K, NOUT))
    hd = np.zeros((C, X * K))
    cnt = np.zeros((C, X * K))
    ar = np.arange(C)
    Zprev = None
    xprev = None
    snaps = {}
    Zlog = np.zeros((C, nblk), dtype=np.int8)
    Fcur = np.zeros((NOUT, C))
    Bcur = np.zeros((NOUT, C))
    for i in range(nblk):
        x = (i * l // G) % X
        Z = (rng.random(C)[:, None] > cumq[x][None, :]).sum(1)
        Z = np.minimum(Z, K - 1)
        Fcur[:] = 0
        Bcur[:] = 0
        for u in range(l):
            ph = u % D
            p = sys_.m[:, ph, x][:, Z, s]
            Y = rng.binomial(M, p) / M
            if u < b:
                Bcur += Y
            else:
                Fcur += Y
            cp = sys_.cumP[ph, x][Z, s]
            s = np.minimum((rng.random(C)[:, None] > cp).sum(1), S - 1)
        Fb = (Fcur / l).T
        Bb = (Bcur / l).T
        cell = x * K + Z
        wf = 1.0 / q[x, Z]
        if i > 0:
            stay = (Z == Zprev).astype(float)
            wb = stay / (q[xprev, Zprev] * q[x, Z])
        else:
            wb = np.zeros(C)
        ht[ar, cell] += wf[:, None] * Fb + wb[:, None] * Bb
        hd[ar, cell] += wf * (1 - beta) + wb * beta
        dm[ar, cell] += Fb
        dm2[ar, cell] += Fb ** 2
        cnt[ar, cell] += 1
        Zprev, xprev = Z.copy(), x
        Zlog[:, i] = Z
        if (i + 1) in ckpt_blocks:
            snaps[i + 1] = dict(
                ht=ht.reshape(R, N, X * K, NOUT).sum(1),
                hd=hd.reshape(R, N, X * K).sum(1),
                dm=dm.reshape(R, N, X * K, NOUT).sum(1),
                dm2=dm2.reshape(R, N, X * K, NOUT).sum(1),
                cnt=cnt.reshape(R, N, X * K).sum(1))
    return snaps, Zlog.reshape(R, N, nblk)


def blocks_per_context(nb, l, X, G):
    ctx = (np.arange(nb) * l // G) % X
    return np.bincount(ctx, minlength=X).astype(float)


def evaluate(sys_, eps, l, b, N, snaps, Zlog, mu, delta, l_dep, caps=CAPS, G0=400, ops_dep=None,
             do_cert=True, max_cert_reps=None):
    X, K = sys_.X, sys_.K
    t = sys_.t_mix
    q = sys_.propensities(eps)
    qlow = np.where(q > 0, q, np.inf).min(axis=0)
    qlow[np.isinf(qlow)] = 0
    G = seg_len(l, sys_.D, G0)
    beta = b / l
    nu = np.ones(X) / X
    rows = []
    safe_mask = (q > 0)
    inc_mask = np.zeros((X, K), bool)
    for x in range(X):
        inc_mask[x, sys_.inc[x]] = True
    eta = persistence_slack(t, l_dep)
    ops_d = ops_dep if ops_dep is not None else block_ops(sys_, l_dep)
    pi_or = solve_policy(sys_, nu, mu[0], [mu[1], mu[2], mu[3]], caps)
    inc_pi = np.zeros((X, K))
    for x in range(X):
        inc_pi[x, sys_.inc[x]] = 1
    if pi_or is None:
        pi_or = inc_pi
    JQ_or = cycle_values(sys_, l_dep, pi_or, G0, ops=ops_d)[1][0]
    JQ_inc = cycle_values(sys_, l_dep, inc_pi, G0, ops=ops_d)[1][0]
    ncell = int(safe_mask.sum()) * NOUT
    dcell = delta / ncell
    for nb, sn in sorted(snaps.items()):
        R = sn["ht"].shape[0]
        kx = blocks_per_context(nb, l, X, G)
        kxx = kx[:, None]
        ht = sn["ht"].reshape(R, X, K, NOUT)
        hd = sn["hd"].reshape(R, X, K)
        dm = sn["dm"].reshape(R, X, K, NOUT)
        cnt = sn["cnt"].reshape(R, X, K)
        dm2 = sn["dm2"].reshape(R, X, K, NOUT)
        est = {}
        est["ht"] = np.moveaxis(ht / (N * kxx)[None, :, :, None], -1, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            est["hajek"] = np.moveaxis(ht / hd[..., None], -1, 1)
            est["dm"] = np.moveaxis(dm / (cnt[..., None] * (1 - beta)), -1, 1)
        row = dict(eps=eps, l=l, b=b, t_mix=t, N=N, nblocks=nb, T_epochs=nb * l, R=R,
                   k_cell_mean=float(cnt.sum(0)[safe_mask].mean() / N) if True else 0)
        for name, e in est.items():
            ok = safe_mask[None, None] & np.isfinite(e)
            err = np.where(ok, e - mu[None], np.nan)
            for k, kn in ((0, "quality"), (1, "exceed")):
                ek = err[:, k]
                row[f"{name}_bias_{kn}_inc"] = float(np.nanmean(ek[:, inc_mask]))
                row[f"{name}_rmse_{kn}_inc"] = float(np.sqrt(np.nanmean(ek[:, inc_mask] ** 2)))
                non = safe_mask & ~inc_mask
                row[f"{name}_bias_{kn}_non"] = float(np.nanmean(ek[:, non]))
                row[f"{name}_rmse_{kn}_non"] = float(np.sqrt(np.nanmean(ek[:, non] ** 2)))
            row[f"{name}_missing"] = float(1 - ok.mean() / safe_mask.mean() * 1.0) if False else float(np.mean(~np.isfinite(e)[:, :, safe_mask]))
        Zp = Zlog[:, :, :nb]
        dm_bias_val = dm_bias(t, l, b)
        rad = {}
        for kind in ("hoeffding", "freedman"):
            rr = np.zeros_like(est["ht"])
            for k in range(NOUT):
                mh = np.clip(est["ht"][:, k], 0, 1)
                rr[:, k] = radii(q, qlow, mh, kx, nb, N, t, l, b, dcell, kind)
            rad[("ht", kind)] = rr
            mdm = np.nan_to_num(np.clip(est["dm"], 0, 1), nan=0.5)
            rad[("dm", kind)] = dm_radius(Zp, l, b, t, X, K, G, dcell, mdm, kind)
        rad[("dm", "best")] = dm_radius(Zp, l, b, t, X, K, G, dcell, mdm, "best")
        with np.errstate(divide="ignore", invalid="ignore"):
            f2 = dm2 / (cnt[..., None] * (1 - beta) ** 2)
            mean = dm / (cnt[..., None] * (1 - beta))
            var = np.maximum(f2 - mean ** 2, 0) * cnt[..., None] / (cnt[..., None] - 1)
            n_ = cnt[..., None]
            emp = dm_bias_val + np.sqrt(2 * var * log(2.0 / dcell) / n_) + 7 * log(2.0 / dcell) / (3 * (n_ - 1))
            emp = np.where(n_ >= 2, emp, np.inf)
        rad[("dm", "empbern")] = np.moveaxis(emp, -1, 1)
        for (en, kind), rr in rad.items():
            e = est[en]
            with np.errstate(invalid="ignore"):
                hit = (np.abs(e - mu[None]) <= rr)
            hit = np.where(np.isfinite(e), hit, True)
            nm = f"{en}_{kind}"
            row[f"cover_{nm}"] = float(hit[:, :, safe_mask].mean())
            row[f"cover_{nm}_exceed_inc"] = float(hit[:, 1][:, inc_mask].mean())
            w = np.minimum(rr, 1.0)
            row[f"width_{nm}_exceed_inc"] = float(2 * w[:, 1][:, inc_mask].mean())
            row[f"width_{nm}_exceed_non"] = float(2 * w[:, 1][:, safe_mask & ~inc_mask].mean())
            row[f"radius_{nm}_exceed_inc_med"] = float(np.median(rr[:, 1][:, inc_mask]))
            row[f"radius_{nm}_exceed_non_med"] = float(np.median(rr[:, 1][:, safe_mask & ~inc_mask]))
        if do_cert:
            nrep = R if max_cert_reps is None else min(R, max_cert_reps)
            for kind in ("ht_hoeffding", "ht_freedman", "dm_hoeffding", "dm_freedman", "dm_best", "dm_empbern", "plugin"):
                valid = ab = 0
                util = []
                cert_valid = []
                for rep in range(nrep):
                    if kind == "plugin":
                        mh = np.nan_to_num(np.clip(est["dm"][rep], 0, 1), nan=0.5)
                        Ulo = mh[0]
                        Cup = [mh[1], mh[2], mh[3]]
                        sl = 0.0
                    else:
                        en = kind.split("_")[0]
                        e = est[en][rep]
                        rr = rad[(en, kind.split("_")[1])][rep]
                        mh = np.where(np.isfinite(e), np.clip(e, 0, 1), 0.5)
                        rr = np.where(np.isfinite(rr), rr, 1.0)
                        Ulo = np.clip(mh[0] - rr[0], 0, 1)
                        Cup = [np.clip(mh[k] + rr[k], 0, 1) for k in (1, 2, 3)]
                        sl = eta
                    pi = solve_policy(sys_, nu, Ulo, Cup, caps, slack=sl)
                    if pi is None:
                        ab += 1
                        pi = inc_pi
                        abst = True
                    else:
                        abst = False
                    V, J = cycle_values(sys_, l_dep, pi, G0, ops=ops_d)
                    feas = bool(np.all(J[1:] <= np.array(caps) + 1e-9))
                    util.append(J[0])
                    if not abst:
                        cert_valid.append(feas)
                    valid += feas
                row[f"{kind}_abstain"] = ab / nrep
                row[f"{kind}_valid_all"] = valid / nrep
                row[f"{kind}_valid_given_cert"] = float(np.mean(cert_valid)) if cert_valid else float("nan")
                row[f"{kind}_utility"] = float(np.mean(util))
            row["utility_incumbent"] = float(JQ_inc)
            row["utility_oracle"] = float(JQ_or)
            row["l_dep"] = l_dep
            row["eta_dep"] = eta
        rows.append(row)
    return rows
