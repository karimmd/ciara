import numpy as np
from math import comb, lgamma, exp, log
from ciara_core import NOUT, block_ops
from ciara_scenarios import scenario_system, design, stationary_levels, policy_values

C_BACK = 12
NAC = 40
NSV = 8
KAPPA = 5.5
GSHAPE = np.array([-0.5, 1.0, 0.5, 0.0])


def _binom_pmf(n, p):
    return np.array([comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)])


def cloud_matrix(S, lam, t_cloud):
    r = 1.0 - exp(-1.0 / t_cloud)
    nS = C_BACK + 1
    pa = _binom_pmf(NAC, min(lam * S / NAC, 1.0))
    pv = _binom_pmf(NSV, KAPPA / NSV)
    Q = np.zeros((nS, nS))
    for c in range(nS):
        for A in range(NAC + 1):
            if pa[A] < 1e-16:
                continue
            for V in range(NSV + 1):
                c2 = min(C_BACK, max(0, c + A - V))
                Q[c, c2] += pa[A] * pv[V]
    P = (1 - r) * Q
    P[:, 0] += r
    return P


def stationary_vec(P):
    n = P.shape[0]
    A = (P - np.eye(n)).T.copy()
    A[-1, :] = 1.0
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    return np.linalg.solve(A, rhs)


def count_vectors(N):
    return np.array([(a, b, N - a - b) for a in range(N + 1) for b in range(N + 1 - a)], dtype=np.int64)


def multinom_w(p, counts):
    N = int(counts[0].sum())
    coef = np.array([np.exp(lgamma(N + 1) - sum(lgamma(k + 1) for k in row)) for row in counts])
    p = np.asarray(p, float)
    w = np.ones(p.shape[:-1] + (len(counts),))
    for j in range(3):
        w = w * np.power(p[..., j, None], counts[:, j][None] if p.ndim > 1 else counts[:, j])
    return w * coef


class CoupledSystem:
    def __init__(self, scen, t_mix, N, gamma, t_cloud=None, l_dep=400):
        self.scen, self.t, self.N, self.gamma = scen, float(t_mix), N, float(gamma)
        self.t_cloud = float(t_cloud if t_cloud is not None else t_mix)
        self.sys = scenario_system(scen, t_mix)
        self.K = 3
        self.cl = self.sys.arms[:, 4].astype(float)
        self.lam = float(self.sys.loads[0])
        self.g = self.gamma * GSHAPE
        self.l_dep = l_dep
        self.rc = 1.0 - exp(-1.0 / self.t_cloud)
        self.mu_edge = stationary_levels(self.sys)
        self.ops = block_ops(self.sys, l_dep)
        self.counts = count_vectors(N)
        self.Sc = self.counts @ self.cl
        self._build_cloud()
        self.q = self.sys.propensities(0.5)[0]

    def _build_cloud(self):
        nS = C_BACK + 1
        chat = np.arange(nS) / C_BACK
        self.chat = chat
        nc = len(self.counts)
        self.hS = np.zeros(nc)
        self.Tl = np.zeros((nc, nS, nS))
        self.Occ = np.zeros((nc, nS))
        cache = {}
        for i, S in enumerate(self.Sc):
            key = round(float(S), 9)
            if key not in cache:
                P = cloud_matrix(S, self.lam, self.t_cloud)
                f = stationary_vec(P)
                v = chat.copy()
                occ = np.zeros(nS)
                for u in range(self.l_dep):
                    occ += v
                    v = P @ v
                cache[key] = (float(f @ chat), np.linalg.matrix_power(P, self.l_dep), occ / self.l_dep)
            self.hS[i], self.Tl[i], self.Occ[i] = cache[key]

    def level_pair(self, p):
        N, out = self.N, np.zeros((NOUT, self.K))
        sib = count_vectors(N - 1)
        w = multinom_w(np.asarray(p, float), sib)
        for a in range(self.K):
            full = sib.copy()
            full[:, a] += 1
            S = full @ self.cl
            h = np.array([self._h_of(s) for s in S])
            out[:, a] = self.mu_edge[:, a] + self.g * self.cl[a] * float(w @ h)
        return out

    def level_round(self, p):
        N, out = self.N, np.zeros((NOUT, self.K))
        p = np.asarray(p, float)
        for a in range(self.K):
            if p[a] <= 0:
                out[:, a] = self.mu_edge[:, a]
                continue
            others = [b for b in range(self.K) if b != a]
            tot, norm = 0.0, 0.0
            for x in range(1, N + 1):
                px = comb(N, x) * p[a] ** x * (1 - p[a]) ** (N - x)
                m = N - x
                hx = 0.0
                if 1 - p[a] <= 1e-15:
                    full = np.zeros(3)
                    full[a] = N
                    hx = self._h_of(float(full @ self.cl))
                else:
                    for j in range(m + 1):
                        pb, pc = p[others[0]] / (1 - p[a]), p[others[1]] / (1 - p[a])
                        pj = comb(m, j) * pb ** j * pc ** (m - j)
                        full = np.zeros(3)
                        full[a], full[others[0]], full[others[1]] = x, j, m - j
                        hx += pj * self._h_of(float(full @ self.cl))
                tot += px * hx
                norm += px
            out[:, a] = self.mu_edge[:, a] + self.g * self.cl[a] * tot / norm
        return out

    def _h_of(self, S):
        i = int(np.argmin(np.abs(self.Sc - S)))
        return self.hS[i]

    def zeta(self, step=0.02):
        mt, mr = self.level_pair(self.q), self.level_round(self.q)
        zp, zr = np.zeros(NOUT), np.zeros(NOUT)
        n = int(round(1 / step))
        for i in range(n + 1):
            for j in range(n + 1 - i):
                pi = np.array([i, j, n - i - j]) / n
                lv = self.level_pair(pi)
                zp = np.maximum(zp, np.abs(lv - mt).max(1))
                zr = np.maximum(zr, np.abs(lv - mr).max(1))
        return zp, zr

    def policy_values(self, pi):
        pi = np.asarray(pi, float)
        Jedge = policy_values(self.ops, pi)
        w = multinom_w(pi, self.counts)
        M = np.einsum("bn,nst->bst", w, self.Tl)
        S = M.shape[-1]
        A = np.transpose(M - np.eye(S)[None], (0, 2, 1)).copy()
        A[:, -1, :] = 1.0
        rhs = np.zeros((pi.shape[0], S))
        rhs[:, -1] = 1.0
        f = np.linalg.solve(A, rhs[..., None])[..., 0]
        share = self.Sc / self.N
        occ = np.einsum("bs,ns->bn", f, self.Occ)
        cloud = np.einsum("bn,bn,n->b", w, occ, share)
        return Jedge + cloud[:, None] * self.g[None, :]

    def check_range(self):
        hi = self.sys.m.max(axis=(1, 2, 3, 4))
        lo = self.sys.m.min(axis=(1, 2, 3, 4))
        top = hi + np.maximum(self.g, 0) * self.cl.max()
        bot = lo + np.minimum(self.g, 0) * self.cl.max()
        assert (top <= 1 + 1e-12).all() and (bot >= -1e-12).all(), (top, bot)
        return True


def simulate_coupled(cs, eps_law, l, b, nblk, R, gammas, seed):
    sys_, N, K = cs.sys, cs.N, cs.K
    S, M = sys_.S, sys_.M
    rng = np.random.default_rng([seed, 1])
    rngs = [np.random.default_rng([seed, 2, gi]) for gi in range(len(gammas))]
    cum = np.cumsum(eps_law)
    cum[-1] = 1.0
    s = np.zeros((R, N), dtype=np.int64)
    c = np.zeros(R, dtype=np.int64)
    Z = np.zeros((R, N, nblk), dtype=np.int8)
    Fs = [np.zeros((R, N, nblk, NOUT), dtype=np.float32) for _ in gammas]
    m = sys_.m[:, 0, 0]
    cumP = sys_.cumP[0, 0]
    cl = cs.cl
    gam = [np.asarray(g_ * GSHAPE) for g_ in gammas]
    chat = np.arange(C_BACK + 1) / C_BACK
    pv_srv = KAPPA / NSV
    for i in range(nblk):
        z = np.minimum((rng.random((R, N))[..., None] > cum[None, None, :]).sum(-1), K - 1)
        Z[:, :, i] = z
        clz = cl[z]
        Ssum = clz.sum(1)
        pa = np.minimum(cs.lam * Ssum / NAC, 1.0)
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
            c = np.where(rng.random(R) < cs.rc, 0, c)
        for gi in range(len(gammas)):
            Fs[gi][:, :, i, :] = np.transpose(accs[gi] / (l - b), (1, 2, 0))
    return Z, Fs


def estimate_zeta(Z0, F0, cl, N):
    R = Z0.shape[0]
    clz = cl[Z0.astype(int)]
    tot = clz.sum(1, keepdims=True)
    e = (tot - clz) / (N - 1)
    out = np.zeros((R, NOUT))
    for a in np.where(cl > 0)[0]:
        sel = (Z0 == a).astype(float)
        n = sel.sum((1, 2))
        em = (sel * e).sum((1, 2)) / np.maximum(n, 1)
        ed = (e - em[:, None, None]) * sel
        den = (ed ** 2).sum((1, 2))
        for k in range(NOUT):
            fm = (sel * F0[..., k]).sum((1, 2)) / np.maximum(n, 1)
            fd = (F0[..., k] - fm[:, None, None]) * sel
            slope = np.where(den > 1e-12, (ed * fd).sum((1, 2)) / np.maximum(den, 1e-12), 0.0)
            out[:, k] = np.maximum(out[:, k], np.abs(slope))
    return out
