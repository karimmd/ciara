import numpy as np
from math import exp

NOUT = 4
LAM_MAX = 0.99


def dm_bias(t, l, b):
    return exp(-b / t) * (1 - exp(-(l - b) / t)) / ((1 - exp(-1 / t)) * (l - b))


def psi_e(lam):
    return -np.log1p(-lam) - lam


_LG = np.linspace(1e-6, LAM_MAX, 20000)
_GG = _LG ** 2 / (1 - _LG) + np.log1p(-_LG) + _LG


def lam_star(ratio):
    return np.interp(ratio, _GG, _LG, left=_LG[0], right=LAM_MAX)


def c1_intervals(Z, F, N, l, b, t, L, look_k, k7, q, K=3):
    R, _, nb = Z.shape
    Zn = Z[:, :N].astype(np.int64)
    Fn = F[:, :N].astype(np.float64)
    bias = dm_bias(t, l, b)
    q = np.asarray(q, float)
    cnt = np.zeros((R, 1, K))
    sumf = np.zeros((R, NOUT, K))
    sv = np.zeros((R, NOUT, K))
    A = np.zeros((R, NOUT, K))
    Lam = np.zeros((R, NOUT, K))
    Vp = np.zeros((R, NOUT, K))
    looks = {int(k): i for i, k in enumerate(look_k)}
    nl = len(look_k)
    cen = np.zeros((R, nl, NOUT, K))
    rad = np.zeros((R, nl, NOUT, K))
    lamsum = np.zeros((R, nl, NOUT, K))
    nplan = (float(k7) * (1.0 - (1.0 - q) ** N))[None, None, :]
    eye = np.eye(K)
    for i in range(nb):
        sel = eye[Zn[:, :, i]]
        X = sel.sum(1)[:, None, :]
        Ys = np.einsum("rnk,rno->rok", sel, Fn[:, :, i, :])
        ne = (X > 0).astype(float)
        Yb = np.where(X > 0, Ys / np.maximum(X, 1.0), 0.0)
        xhat = (0.5 + sumf) / (1.0 + cnt)
        s2 = (0.25 + sv) / (1.0 + cnt)
        lam = lam_star(L / (nplan * s2))
        ps = psi_e(lam)
        res2 = ne * (Yb - xhat) ** 2
        A += lam * ne * Yb
        Lam += lam * ne
        Vp += ps * res2
        sv += res2
        sumf += ne * Yb
        cnt += ne
        if (i + 1) in looks:
            j = looks[i + 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                cen[:, j] = np.where(Lam > 0, A / Lam, 0.5)
                rad[:, j] = np.where(Lam > 0, bias + (L + Vp) / Lam, np.inf)
            lamsum[:, j] = Lam
    return cen, rad, lamsum
