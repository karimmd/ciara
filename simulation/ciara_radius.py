import numpy as np
from math import log
from ciara_core import dm_bias, NOUT

LAM_MAX = 0.99
PRIOR_MEAN = 0.5
PRIOR_VAR = 0.25


def psi_e(lam):
    return -np.log1p(-lam) - lam


_LG = np.linspace(1e-6, LAM_MAX, 20000)
_GG = _LG ** 2 / (1 - _LG) + np.log1p(-_LG) + _LG


def lam_star(ratio):
    return np.interp(ratio, _GG, _LG, left=_LG[0], right=LAM_MAX)


def eb_intervals(Z, F, nodes, l, b, t, L, look_k, n_plan, K=3):
    R, _, nb = Z.shape
    Zn = Z[:, :nodes]
    Fn = F[:, :nodes].astype(np.float64)
    bias = dm_bias(t, l, b)
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
    lam_sum = np.zeros((R, nl, NOUT, K))
    npl = np.asarray(n_plan, float)[None, None, :]
    eye = np.eye(K)
    for i in range(nb):
        xhat = (PRIOR_MEAN + sumf) / (1.0 + cnt)
        s2 = (PRIOR_VAR + sv) / (1.0 + cnt)
        lam = lam_star(L / (npl * s2))
        ps = psi_e(lam)
        sel = eye[Zn[:, :, i]]
        f = Fn[:, :, i, :]
        Sf = np.einsum("rnk,rno->rok", sel, f)
        res2 = (f[:, :, :, None] - xhat[:, None]) ** 2
        Sr = np.einsum("rnk,rnok->rok", sel, res2)
        cn = sel.sum(1)[:, None, :]
        A += lam * Sf
        Lam += lam * cn
        Vp += ps * Sr
        sv += Sr
        sumf += Sf
        cnt += cn
        if (i + 1) in looks:
            j = looks[i + 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                cen[:, j] = np.where(Lam > 0, A / Lam, 0.5)
                rad[:, j] = np.where(Lam > 0, bias + (L + Vp) / Lam, np.inf)
            lam_sum[:, j] = Lam
    return cen, rad, lam_sum


def eb_level(Mc, delta, looks_split=1):
    return log(2.0 * Mc * looks_split / delta)
