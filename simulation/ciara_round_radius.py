import numpy as np
from ciara_core import dm_bias, NOUT
from ciara_radius import lam_star, psi_e


def c1_intervals(Z, F, N, l, b, t, L, look_k, k7, q, mode, K=3):
    R, _, nb = Z.shape
    Zn = Z[:, :N].astype(np.int64)
    Fn = F[:, :N].astype(np.float64)
    bias = dm_bias(t, l, b)
    q = np.asarray(q, float)
    cnt = np.zeros((R, 1, K))
    sumf = np.zeros((R, NOUT, K))
    sumx = np.zeros((R, 1, K))
    sv = np.zeros((R, NOUT, K))
    A = np.zeros((R, NOUT, K))
    Lam = np.zeros((R, NOUT, K))
    Vp = np.zeros((R, NOUT, K))
    looks = {int(k): i for i, k in enumerate(look_k)}
    nl = len(look_k)
    cen = np.zeros((R, nl, NOUT, K))
    rad = np.zeros((R, nl, NOUT, K))
    lamsum = np.zeros((R, nl, NOUT, K))
    if mode == "P":
        nplan = np.full((1, 1, K), float(k7))
    else:
        nplan = (float(k7) * (1.0 - (1.0 - q) ** N))[None, None, :]
    eye = np.eye(K)
    for i in range(nb):
        sel = eye[Zn[:, :, i]]
        X = sel.sum(1)[:, None, :]
        Ys = np.einsum("rnk,rno->rok", sel, Fn[:, :, i, :])
        if mode == "P":
            xhat = (0.5 + sumf) / (1.0 + sumx)
            s2 = (0.25 + sv) / (1.0 + cnt)
            lam = lam_star(L / (nplan * s2))
            ps = psi_e(lam)
            xi = (Ys - xhat * X) / N
            A += lam * (xhat + xi / q[None, None, :])
            Lam += lam
            Vp += ps * xi ** 2
            sv += xi ** 2
            sumf += Ys
            sumx += X
            cnt += 1.0
        else:
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
                if mode == "P":
                    cen[:, j] = A / Lam
                    rad[:, j] = bias + (L + Vp) / (q[None, None, :] * Lam)
                else:
                    cen[:, j] = np.where(Lam > 0, A / Lam, 0.5)
                    rad[:, j] = np.where(Lam > 0, bias + (L + Vp) / Lam, np.inf)
            lamsum[:, j] = Lam
    return cen, rad, lamsum
