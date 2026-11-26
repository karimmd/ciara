import numpy as np
from math import ceil, exp, log, sqrt, gcd

OUT_NAMES = ["quality", "exceed", "deadline", "cloud"]
NOUT = 4

ARMS = np.array([
    [0.55, 3.6, 0.02, 0.01, 0.0],
    [0.55, 4.6, 0.04, 0.02, 0.0],
    [0.70, 2.8, 0.05, 0.03, 0.0],
    [0.70, 3.8, 0.08, 0.05, 0.0],
    [0.82, 2.0, 0.10, 0.08, 0.0],
    [0.80, 5.0, 0.06, 0.03, 0.5],
    [0.88, 4.2, 0.09, 0.05, 0.6],
    [0.92, 3.2, 0.13, 0.09, 0.8],
])
LOADS = np.array([2.4, 3.4, 4.4])
SAFE = [list(range(8)), list(range(8)), [0, 1, 2, 3, 5, 6]]
INCUMBENT = [1, 1, 1]
CAPS = [0.15, 0.10, 0.30]


class System:
    def __init__(self, t_mix=5.0, B=12, M=10, drift=0.0, D=1, arms=ARMS, loads=LOADS,
                 safe=SAFE, incumbent=INCUMBENT):
        self.t_mix = float(t_mix)
        self.r = 1.0 - exp(-1.0 / t_mix)
        self.B, self.S, self.M, self.D = B, B + 1, M, D
        self.arms, self.loads = arms, loads
        self.K, self.X = len(arms), len(loads)
        self.safe, self.inc = safe, incumbent
        S, K, X = self.S, self.K, self.X
        self.P = np.zeros((D, X, K, S, S))
        self.m = np.zeros((NOUT, D, X, K, S))
        nA, nS = 8, 8
        for ph in range(D):
            for x in range(X):
                lam = loads[x] * (1.0 + drift * np.sin(2 * np.pi * ph / D))
                lam = min(max(lam, 0.05), nA - 0.05)
                pa = _binom_pmf(nA, lam / nA)
                for a in range(K):
                    qual, cap, e0, d0, cl = arms[a]
                    ps = _binom_pmf(nS, cap / nS)
                    Q = np.zeros((S, S))
                    for s in range(S):
                        for A_ in range(nA + 1):
                            for V in range(nS + 1):
                                s2 = min(B, max(0, s + A_ - V))
                                Q[s, s2] += pa[A_] * ps[V]
                    Pm = (1 - self.r) * Q
                    Pm[:, 0] += self.r
                    self.P[ph, x, a] = Pm
                    u = np.arange(S) / B
                    self.m[0, ph, x, a] = qual * (1 - 0.35 * u)
                    self.m[1, ph, x, a] = np.clip(e0 + 0.60 * u, 0, 1)
                    self.m[2, ph, x, a] = np.clip(d0 + 0.80 * u ** 2, 0, 1)
                    self.m[3, ph, x, a] = cl
        self.cumP = np.cumsum(self.P, axis=-1)
        self.cumP[..., -1] = 1.0

    def propensities(self, eps):
        q = np.zeros((self.X, self.K))
        for x in range(self.X):
            sf = self.safe[x]
            for a in sf:
                q[x, a] = eps / len(sf)
            q[x, self.inc[x]] += 1 - eps
        return q


def _binom_pmf(n, p):
    from math import comb
    return np.array([comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)])


def seg_len(l, D, G0=400):
    base = l * D // gcd(l, D)
    return base * ceil(G0 / base)


def block_ops(sys_, l):
    S, X, K, D = sys_.S, sys_.X, sys_.K, sys_.D
    assert l % D == 0
    T = np.zeros((X, K, S, S))
    O = np.zeros((X, K, NOUT, S))
    for x in range(X):
        for a in range(K):
            Mx = np.eye(S)
            occ = np.zeros((NOUT, S))
            for u in range(l):
                ph = u % D
                occ += sys_.m[:, ph, x, a] @ Mx.T
                Mx = Mx @ sys_.P[ph, x, a]
            T[x, a] = Mx
            O[x, a] = occ / l
    return T, O


def cycle_values(sys_, l, pi, G0=400, tol=1e-13, maxcyc=400, ops=None):
    T, O = ops if ops is not None else block_ops(sys_, l)
    X, K, S = sys_.X, sys_.K, sys_.S
    G = seg_len(l, sys_.D, G0)
    nb = G // l
    Tm = np.einsum("xa,xast->xst", pi, T)
    Om = np.einsum("xa,xaks->xks", pi, O)
    f = np.zeros(S)
    f[0] = 1.0
    for cyc in range(maxcyc):
        f0 = f.copy()
        for x in range(X):
            for _ in range(nb):
                f = f @ Tm[x]
        if np.abs(f - f0).sum() < tol:
            break
    V = np.zeros((NOUT, X))
    for x in range(X):
        acc = np.zeros(NOUT)
        for _ in range(nb):
            acc += Om[x] @ f
            f = f @ Tm[x]
        V[:, x] = acc / nb
    return V, V.mean(axis=1)


def arm_levels(sys_, l, G0=400):
    ops = block_ops(sys_, l)
    mu = np.zeros((NOUT, sys_.X, sys_.K))
    for a in range(sys_.K):
        pi = np.zeros((sys_.X, sys_.K))
        pi[:, a] = 1.0
        V, _ = cycle_values(sys_, l, pi, G0, ops=ops)
        mu[:, :, a] = V
    return mu


def lp_max(c, A_ub, b_ub, A_eq, b_eq, tol=1e-9):
    c = np.asarray(c, float)
    n = len(c)
    mu_, me = len(b_ub), len(b_eq)
    A = np.zeros((mu_ + me, n + mu_))
    b = np.zeros(mu_ + me)
    if mu_:
        A[:mu_, :n] = A_ub
        A[:mu_, n:] = np.eye(mu_)
        b[:mu_] = b_ub
    if me:
        A[mu_:, :n] = A_eq
        b[mu_:] = b_eq
    neg = b < 0
    A[neg] *= -1
    b[neg] *= -1
    m_, nt = A.shape
    Tab = np.zeros((m_ + 1, nt + m_ + 1))
    Tab[:m_, :nt] = A
    Tab[:m_, nt:nt + m_] = np.eye(m_)
    Tab[:m_, -1] = b
    basis = list(range(nt, nt + m_))
    Tab[m_, :nt] = -A.sum(axis=0)
    Tab[m_, -1] = -b.sum()

    def run(Tab, basis, ncols):
        for _ in range(5000):
            row = Tab[-1, :ncols]
            js = np.where(row < -tol)[0]
            if len(js) == 0:
                return True
            j = js[0]
            col = Tab[:-1, j]
            pos = col > tol
            if not pos.any():
                return False
            ratios = np.full(len(col), np.inf)
            ratios[pos] = Tab[:-1, -1][pos] / col[pos]
            mn = ratios.min()
            cand = [i for i in range(len(col)) if ratios[i] <= mn + 1e-12]
            i = min(cand, key=lambda r: basis[r])
            Tab[i] /= Tab[i, j]
            for r_ in range(Tab.shape[0]):
                if r_ != i and Tab[r_, j] != 0:
                    Tab[r_] -= Tab[r_, j] * Tab[i]
            basis[i] = j
        return True

    run(Tab, basis, nt + m_)
    if Tab[m_, -1] < -1e-7:
        return None
    for i in range(m_):
        if basis[i] >= nt:
            js = np.where(np.abs(Tab[i, :nt]) > 1e-9)[0]
            if len(js):
                j = js[0]
                Tab[i] /= Tab[i, j]
                for r_ in range(Tab.shape[0]):
                    if r_ != i and Tab[r_, j] != 0:
                        Tab[r_] -= Tab[r_, j] * Tab[i]
                basis[i] = j
    T2 = np.zeros((m_ + 1, nt + 1))
    T2[:m_, :nt] = Tab[:m_, :nt]
    T2[:m_, -1] = Tab[:m_, -1]
    cc = np.zeros(nt)
    cc[:n] = c
    T2[m_, :nt] = -cc
    for i in range(m_):
        if basis[i] < nt and T2[m_, basis[i]] != 0:
            T2[m_] -= T2[m_, basis[i]] * T2[i]
    keep = [i for i in range(m_) if basis[i] < nt]
    T2 = T2[keep + [m_]]
    basis = [basis[i] for i in keep]
    ok = run(T2, basis, nt)
    if not ok:
        return None
    x = np.zeros(nt)
    for i, bj in enumerate(basis):
        x[bj] = T2[i, -1]
    return x[:n]


def solve_policy(sys_, nu, U, C, cap, slack=0.0):
    X, K = sys_.X, sys_.K
    idx = [(x, a) for x in range(X) for a in sys_.safe[x]]
    n = len(idx)
    c = np.array([nu[x] * U[x, a] for x, a in idx])
    A_ub = np.array([[nu[x] * Ck[x, a] for x, a in idx] for Ck in C])
    b_ub = np.array(cap, float) - slack
    A_eq = np.zeros((X, n))
    for j, (x, a) in enumerate(idx):
        A_eq[x, j] = 1.0
    sol = lp_max(c, A_ub, b_ub, A_eq, np.ones(X))
    if sol is None:
        return None
    pi = np.zeros((X, K))
    for j, (x, a) in enumerate(idx):
        pi[x, a] = max(sol[j], 0.0)
    pi /= pi.sum(axis=1, keepdims=True)
    return pi


def carry_bias(t, l, b):
    return exp(-b / t) * (1 - exp(-l / t)) / ((1 - exp(-1 / t)) * l)


def tail_tau(t, l):
    return exp(-1 / t) / ((1 - exp(-1 / t)) * l)


def persistence_slack(t, l_dep):
    return (1 - exp(-l_dep / t)) / ((1 - exp(-1 / t)) * l_dep)


def radii(q, qlow, mhat, kx, ktot, N, t, l, b, delta_cell, kind="hoeffding", kappa=4.0):
    beta = b / l
    tau = tail_tau(t, l)
    L = log(2.0 / delta_cell)
    X, K = q.shape
    qq = np.where(q > 0, q, np.nan)
    ql = np.where(qlow > 0, qlow, np.nan)[None, :]
    Wbar = (1 - beta) / qq + beta / (qq * ql)
    rho = Wbar + beta / ql + tau
    kxx = np.asarray(kx, float)[:, None]
    bb = carry_bias(t, l, b) + beta / kxx
    if kind == "hoeffding":
        S2 = 3.0 * (kxx * (Wbar ** 2 + (beta / ql) ** 2) + ktot * tau ** 2)
        dev = np.sqrt(N * S2 * L / 2.0) / (N * kxx)
        r = bb + dev
        return np.broadcast_to(r, mhat.shape).copy()
    V1 = (1 - beta) ** 2 / qq + beta ** 2 / (qq * ql)
    r = np.ones(mhat.shape)
    for _ in range(40):
        mp = np.clip(mhat + r, 0.0, 1.0)
        v = N * (kappa * kxx * mp * V1 + ktot * tau ** 2)
        Rl = rho * L / 3.0
        eps = Rl + np.sqrt(Rl ** 2 + 2 * v * L)
        r = bb + eps / (N * kxx)
    return r


def arm_levels_stationary(sys_):
    S = sys_.S
    mu = np.zeros((NOUT, sys_.X, sys_.K))
    for x in range(sys_.X):
        for a in range(sys_.K):
            P = sys_.P[0, x, a]
            w, v = np.linalg.eig(P.T)
            f = np.real(v[:, np.argmin(np.abs(w - 1))])
            f = np.abs(f) / np.abs(f).sum()
            mu[:, x, a] = sys_.m[:, 0, x, a] @ f
    return mu


def dm_bias(t, l, b):
    return exp(-b / t) * (1 - exp(-(l - b) / t)) / ((1 - exp(-1 / t)) * (l - b))


def dm_radius(Zlog, l, b, t, X, K, G, delta_cell, mhat=None, kind="hoeffding"):
    if kind == "best":
        a = dm_radius(Zlog, l, b, t, X, K, G, delta_cell / 2.0, mhat, "hoeffding")
        f = dm_radius(Zlog, l, b, t, X, K, G, delta_cell / 2.0, mhat, "freedman")
        return np.minimum(a, f)
    R, N, nb = Zlog.shape
    L = log(2.0 / delta_cell)
    ctx = (np.arange(nb) * l // G) % X
    nterm = int(min(nb, ceil(40 * t / l) + 2))
    m_idx = np.arange(1, nterm + 1)
    cvec = np.zeros(nterm + 1)
    ssum = exp(-(b + 1) / t) * (1 - exp(-(l - b) / t)) / (1 - exp(-1 / t))
    cvec[1:] = np.exp(-(m_idx - 1) * l / t) * ssum / (l - b)
    n_t = np.zeros((R, X, K))
    S_rho2 = np.zeros((R, X, K))
    S_c2 = np.zeros((R, X, K))
    S_cross = np.zeros((R, X, K))
    Rmax = np.zeros((R, X, K))
    for r in range(R):
        for c in range(N):
            z = Zlog[r, c]
            for x in range(X):
                for a in range(K):
                    sel = ((ctx == x) & (z == a)).astype(float)
                    ns = sel.sum()
                    if ns == 0:
                        continue
                    tail = np.convolve(sel[::-1], cvec)[:nb][::-1]
                    rho = sel + tail
                    n_t[r, x, a] += ns
                    S_rho2[r, x, a] += (rho ** 2).sum()
                    S_c2[r, x, a] += (tail ** 2).sum()
                    S_cross[r, x, a] += (sel * tail).sum()
                    Rmax[r, x, a] = max(Rmax[r, x, a], rho.max())
    bias = dm_bias(t, l, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "hoeffding":
            rad = bias + np.sqrt(S_rho2 * L / 2.0) / n_t
            rad = np.where(n_t > 0, rad, np.inf)
            return np.broadcast_to(rad[:, None], (R, NOUT, X, K)).copy()
        out = np.zeros((R, NOUT, X, K))
        for k in range(NOUT):
            rk = np.ones((R, X, K))
            for _ in range(40):
                mp = np.clip(mhat[:, k] + rk, 0.0, 1.0)
                psi = np.where(mp <= 0.5, mp * (1 - mp), 0.25)
                v = psi * n_t + 2 * np.sqrt(psi) * S_cross + S_c2
                Rl = Rmax * L / 3.0
                eps_sum = Rl + np.sqrt(Rl ** 2 + 2 * v * L)
                rk = bias + eps_sum / n_t
            out[:, k] = np.where(n_t > 0, rk, np.inf)
        return out
