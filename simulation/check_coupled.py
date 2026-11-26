import os, sys, itertools
import numpy as np
from math import comb, exp, log
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ciara_core import dm_bias, NOUT
from ciara_radius import lam_star, psi_e
from ciara_round_radius import c1_intervals
from ciara_coupled import CoupledSystem, cloud_matrix, count_vectors, multinom_w, C_BACK
from ciara_scenarios import design

rng = np.random.default_rng(20261021)


def check_one_step(trials=300):
    worst_P, worst_R = 0.0, 0.0
    for _ in range(trials):
        N = int(rng.integers(2, 5))
        q = float(rng.uniform(0.05, 0.9))
        levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        zs = list(itertools.product([0, 1], repeat=N))
        grid = list(itertools.product(range(len(levels)), repeat=N))
        pm = {z: rng.dirichlet(np.ones(len(grid)) * rng.uniform(0.1, 1.0)) for z in zs}
        lam = float(rng.uniform(0.01, 0.99))
        xhat = float(rng.uniform(0, 1))
        psi = -np.log1p(-lam) - lam
        Exi, vals = 0.0, []
        for z in zs:
            pz = np.prod([q if zi == 0 else 1 - q for zi in z])
            sel = np.array([1.0 if zi == 0 else 0.0 for zi in z])
            for gi, g in enumerate(grid):
                f = levels[list(g)]
                xi = (np.sum(sel * f) - xhat * sel.sum()) / N
                w = pz * pm[z][gi]
                Exi += w * xi
                vals.append((w, xi))
        tot = sum(w * np.exp(lam * (xi - Exi) - psi * xi ** 2) for w, xi in vals)
        worst_P = max(worst_P, tot)
        P1 = 1 - (1 - q) ** N
        Em, vals = 0.0, []
        for z in zs:
            if sum(1 for zi in z if zi == 0) == 0:
                continue
            pz = np.prod([q if zi == 0 else 1 - q for zi in z]) / P1
            sel = np.array([1.0 if zi == 0 else 0.0 for zi in z])
            for gi, g in enumerate(grid):
                f = levels[list(g)]
                yb = np.sum(sel * f) / sel.sum()
                w = pz * pm[z][gi]
                Em += w * yb
                vals.append((w, yb))
        tot = sum(w * np.exp(lam * (yb - Em) - psi * (yb - xhat) ** 2) for w, yb in vals)
        worst_R = max(worst_R, tot)
    return worst_P, worst_R


def c1_scalar(Z, F, N, l, b, t, L, look_k, k7, q, mode, K=3):
    nb = Z.shape[1]
    bias = dm_bias(t, l, b)
    out_c = np.zeros((len(look_k), NOUT, K))
    out_r = np.zeros((len(look_k), NOUT, K))
    lamlog = np.zeros((nb, NOUT, K))
    xlog = np.zeros((nb, NOUT, K))
    for a in range(K):
        for k in range(NOUT):
            cnt = sumf = sumx = sv = A = Lam = Vp = 0.0
            nplan = float(k7) if mode == "P" else float(k7) * (1 - (1 - q[a]) ** N)
            li = 0
            for i in range(nb):
                sel = (Z[:, i] == a).astype(float)
                X = sel.sum()
                Ys = float((sel * F[:, i, k]).sum())
                if mode == "P":
                    xhat = (0.5 + sumf) / (1 + sumx)
                    s2 = (0.25 + sv) / (1 + cnt)
                    lam = float(lam_star(np.array(L / (nplan * s2))))
                    ps = float(psi_e(np.array(lam)))
                    xi = (Ys - xhat * X) / N
                    A += lam * (xhat + xi / q[a]); Lam += lam; Vp += ps * xi ** 2; sv += xi ** 2
                    sumf += Ys; sumx += X; cnt += 1
                else:
                    ne = 1.0 if X > 0 else 0.0
                    xhat = (0.5 + sumf) / (1 + cnt)
                    s2 = (0.25 + sv) / (1 + cnt)
                    lam = float(lam_star(np.array(L / (nplan * s2))))
                    ps = float(psi_e(np.array(lam)))
                    if X > 0:
                        yb = Ys / X
                        A += lam * yb; Lam += lam; Vp += ps * (yb - xhat) ** 2; sv += (yb - xhat) ** 2
                        sumf += yb; cnt += 1
                lamlog[i, k, a] = lam
                xlog[i, k, a] = xhat
                if (i + 1) == look_k[li] if li < len(look_k) else False:
                    if mode == "P":
                        out_c[li, k, a] = A / Lam
                        out_r[li, k, a] = bias + (L + Vp) / (q[a] * Lam)
                    else:
                        out_c[li, k, a] = A / Lam if Lam > 0 else 0.5
                        out_r[li, k, a] = bias + (L + Vp) / Lam if Lam > 0 else np.inf
                    li += 1
    return out_c, out_r, lamlog, xlog


def check_vectorized():
    t = 5
    l, b = design(t)
    N, nb, R = 5, 120, 3
    q = np.array([2 / 3, 1 / 6, 1 / 6])
    Z = rng.choice(3, size=(R, N, nb), p=q).astype(np.int8)
    F = np.clip(0.3 + 0.1 * rng.standard_normal((R, N, nb, NOUT)) + 0.05 * (Z[..., None] == 1), 0, 1).astype(np.float32)
    look_k = np.array([20, 60, 120])
    L = log(2 * 24 / 0.05)
    worst = 0.0
    for mode in ("P", "R"):
        cen, rad, _ = c1_intervals(Z, F, N, l, b, t, L, look_k, 100, q, mode)
        for r in range(R):
            c, rr, lamlog, xlog = c1_scalar(Z[r], F[r].astype(np.float64), N, l, b, t, L, look_k, 100, q, mode)
            ci = np.transpose(c, (0, 1, 2))
            worst = max(worst, np.abs(cen[r] - ci).max(), np.nanmax(np.abs(np.where(np.isfinite(rr), rad[r] - rr, 0))))
            h = 70
            F2 = F[r].astype(np.float64).copy()
            Z2 = Z[r].copy()
            F2[:, h, :] = 1.0 - F2[:, h, :]
            Z2[:, h] = (Z2[:, h] + 1) % 3
            _, _, lam2, x2 = c1_scalar(Z2, F2, N, l, b, t, L, look_k, 100, q, mode)
            assert np.array_equal(lam2[:h + 1], lamlog[:h + 1]) and np.array_equal(x2[:h + 1], xlog[:h + 1]), "not predictable"
    return worst


def check_band(t, N, gam):
    cs = CoupledSystem("K3X1", t, N, gam)
    l, b = design(t)
    sys_ = cs.sys
    worst = np.zeros(NOUT)
    for ci, cnt in enumerate(cs.counts):
        S = cs.Sc[ci]
        Pc = cloud_matrix(S, cs.lam, cs.t_cloud)
        for a in range(3):
            if cnt[a] == 0:
                continue
            P = sys_.P[0, 0, a]
            Mx_e = np.eye(P.shape[0])
            Mx_c = np.eye(Pc.shape[0])
            acc_e = np.zeros((NOUT, P.shape[0]))
            acc_c = np.zeros(Pc.shape[0])
            for u in range(l):
                if u >= b:
                    acc_e += sys_.m[:, 0, 0, a] @ Mx_e.T
                    acc_c += cs.chat @ Mx_c.T
                Mx_e = Mx_e @ P
                Mx_c = Mx_c @ Pc
            acc_e /= (l - b)
            acc_c /= (l - b)
            sus_e = cs.mu_edge[:, a]
            sus_c = cs.hS[ci]
            for k in range(NOUT):
                dev_e = np.abs(acc_e[k] - sus_e[k]).max()
                dev_c = np.abs(cs.g[k] * cs.cl[a] * (acc_c - sus_c)).max()
                worst[k] = max(worst[k], dev_e + dev_c)
    return worst, dm_bias(t, l, b)


def pooled_order_counterexample():
    s_eta, s_e = 4e-4, 4e-4
    rho = s_eta / (s_eta + s_e)
    n = 400000
    eta = np.sqrt(s_eta) * rng.standard_normal(n)
    f1 = 0.3 + eta + np.sqrt(s_e) * rng.standard_normal(n)
    f2 = 0.3 + eta + np.sqrt(s_e) * rng.standard_normal(n)
    slope = np.polyfit(f1 - 0.3, f2 - 0.3, 1)[0]
    dev = np.abs((f2 - 0.3)[np.abs(f1 - 0.3) > 0.03]).mean()
    return rho, slope, np.abs(np.polyval([rho, 0], f1 - 0.3)).max()


if __name__ == "__main__":
    wp, wr = check_one_step()
    print(f"1. one-step inequality, largest conditional expectation over 300 random laws: pair-weighted {wp:.12f}, round-mean {wr:.12f}  (must be <= 1)")
    assert wp <= 1 + 1e-9 and wr <= 1 + 1e-9
    print(f"2. vectorized against scalar implementation (P and R): max abs difference {check_vectorized():.2e}; predictability assertion passed")
    for t in (3, 5, 8):
        for N in (5, 6):
            w, bd = check_band(t, N, 0.0936)
            print(f"3. carryover band, t={t} N={N} gamma=0.0936: worst exact deviation per outcome {np.round(w, 6)}  <=  omega*beta_DM = {bd:.6f}  {'OK' if (w <= bd + 1e-12).all() else 'VIOLATED'}")
    rho, slope, mx = pooled_order_counterexample()
    print(f"4. pooled order, common shock: regression slope of f_2 on f_1 = {slope:.3f} (theory {rho:.3f}); a window 0.03 above its mean shifts the conditional mean of the next node by {rho*0.03:.4f}, against a band of {dm_bias(5, *design(5)):.5f}")
    cs = CoupledSystem("K3X1", 5, 5, 0.0936)
    q = cs.q
    mt, mr = cs.level_pair(q), cs.level_round(q)
    N = cs.N
    for a in (1, 2):
        tot, nrm, tot_r, nrm_r = 0.0, 0.0, 0.0, 0.0
        for x in range(1, N + 1):
            px = comb(N, x) * q[a] ** x * (1 - q[a]) ** (N - x)
            others = [b_ for b_ in range(3) if b_ != a]
            m = N - x
            hx = 0.0
            for j in range(m + 1):
                pb, pc = q[others[0]] / (1 - q[a]), q[others[1]] / (1 - q[a])
                full = np.zeros(3); full[a], full[others[0]], full[others[1]] = x, j, m - j
                hx += comb(m, j) * pb ** j * pc ** (m - j) * cs._h_of(float(full @ cs.cl))
            tot += x * px * hx; nrm += x * px
            tot_r += px * hx; nrm_r += px
        lvl_p = cs.mu_edge[1, a] + cs.g[1] * cs.cl[a] * tot / nrm
        lvl_r = cs.mu_edge[1, a] + cs.g[1] * cs.cl[a] * tot_r / nrm_r
        print(f"5. arm {a}: mu_tilde exceed {mt[1, a]:.6f} vs size-biased {lvl_p:.6f}; mu_rw {mr[1, a]:.6f} vs {lvl_r:.6f}")
        assert abs(lvl_p - mt[1, a]) < 1e-9 and abs(lvl_r - mr[1, a]) < 1e-9
    print("all checks done")
