import numpy as np
from math import log
from ciara_core import NOUT, dm_bias
from ciara_radius import lam_star, psi_e
from ciara_round_radius import c1_intervals
from ciara_coupled import (cloud_matrix, stationary_vec, C_BACK, NAC, NSV, KAPPA, GSHAPE)
from ciara_scenarios import stationary_levels
from math import exp


def one_sided_level(Mc, delta):
    return log(Mc / delta)


def two_sided_level(Mc, delta):
    return log(2.0 * Mc / delta)


class BalSys:
    def __init__(self, sys_, N, gamma, t_cloud):
        self.sys, self.N, self.gamma = sys_, N, float(gamma)
        self.K = sys_.K
        self.cl = sys_.arms[:, 4].astype(float)
        self.lam = float(sys_.loads[0])
        self.g = self.gamma * GSHAPE
        self.t_cloud = float(t_cloud)
        self.rc = 1.0 - exp(-1.0 / self.t_cloud)
        self.mu_edge = stationary_levels(sys_)

    def h_of(self, S):
        P = cloud_matrix(S, self.lam, self.t_cloud)
        f = stationary_vec(P)
        return float(f @ (np.arange(C_BACK + 1) / C_BACK))

    def level_bal(self, comp):
        S = float(np.asarray(comp, float) @ self.cl)
        h = self.h_of(S)
        return self.mu_edge + self.g[:, None] * self.cl[None, :] * h

    def check_range(self, comp):
        lv = self.level_bal(comp)
        assert (lv >= -1e-12).all() and (lv <= 1 + 1e-12).all()


def draw_balanced(rng, R, comp, nb):
    base = np.repeat(np.arange(len(comp)), comp)
    N = len(base)
    key = rng.random((R, nb, N))
    perm = np.argsort(key, axis=-1)
    return np.transpose(base[perm], (0, 2, 1)).astype(np.int8)


def simulate_balanced(bs, comp, l, b, nblk, R, gammas, seed, common_shock=0.0):
    sys_, N, K = bs.sys, bs.N, bs.K
    S, M = sys_.S, sys_.M
    rng = np.random.default_rng([seed, 1])
    rngs = [np.random.default_rng([seed, 2, gi]) for gi in range(len(gammas))]
    Z = draw_balanced(np.random.default_rng([seed, 3]), R, comp, nblk)
    s = np.zeros((R, N), dtype=np.int64)
    c = np.zeros(R, dtype=np.int64)
    Fs = [np.zeros((R, N, nblk, NOUT), dtype=np.float32) for _ in gammas]
    m = sys_.m[:, 0, 0]
    cumP = sys_.cumP[0, 0]
    cl = bs.cl
    gam = [np.asarray(g_ * GSHAPE) for g_ in gammas]
    chat = np.arange(C_BACK + 1) / C_BACK
    pv_srv = KAPPA / NSV
    for i in range(nblk):
        z = Z[:, :, i].astype(np.int64)
        clz = cl[z]
        Ssum = clz.sum(1)
        pa = np.minimum(bs.lam * Ssum / NAC, 1.0)
        accs = [np.zeros((NOUT, R, N)) for _ in gammas]
        for u in range(l):
            cc = chat[c][:, None] * clz
            if u >= b:
                base = m[:, z, s]
                for gi in range(len(gammas)):
                    p = base + gam[gi][:, None, None] * cc[None]
                    accs[gi] += rngs[gi].binomial(M, np.clip(p, 0.0, 1.0)) / M
            cp = cumP[z, s]
            s = np.minimum((rng.random((R, N))[..., None] > cp).sum(-1), S - 1)
            A = rng.binomial(NAC, pa)
            V = rng.binomial(NSV, pv_srv, size=R)
            c = np.clip(c + A - V, 0, C_BACK)
            c = np.where(rng.random(R) < bs.rc, 0, c)
        for gi in range(len(gammas)):
            Fs[gi][:, :, i, :] = np.transpose(accs[gi] / (l - b), (1, 2, 0))
    return Z, Fs


def side_mask(K):
    up = np.ones((NOUT, K), dtype=bool)
    up[0, 1:] = False
    return up


def c1r_balanced(Z, F, N, l, b, t, L, look_k, k7, K):
    return c1_intervals(Z, F, N, l, b, t, L, look_k, k7, np.ones(K), "R", K=K)


def one_sided_violation(cen, rad, tgt, up):
    radf = np.where(np.isfinite(rad), rad, 1.0)
    dev = np.where(up[None, None], tgt[None, None] - cen, cen - tgt[None, None])
    return (dev / radf).max((2, 3))


def c1r_planned_radius(n, s2, L, bias):
    n = np.asarray(n, float)
    s2 = np.maximum(np.asarray(s2, float), 1e-12)
    lam = lam_star(L / (n * s2))
    return bias + (L + psi_e(lam) * n * s2) / (lam * n)
