from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .band import beta_dm, rho_of
from .constants import DELTA, K_OUT

LAM_MAX = 0.99
PRIOR_MEAN = 0.5
PRIOR_VAR = 0.25


def eta_dep(t_bar: float, l_dep: float, tau: float = 0.0) -> float:
    rho = rho_of(t_bar)
    return (tau + (1 - rho ** (l_dep - tau)) / (1 - rho)) / l_dep


def level(n_arms: int, n_outcomes: int = K_OUT, delta: float = DELTA, one_sided: bool = False) -> float:
    return math.log((1.0 if one_sided else 2.0) * n_arms * n_outcomes / delta)


def planned_rounds(q: float, n_nodes: int, k_blocks: int) -> float:
    return k_blocks * (1.0 - (1.0 - q) ** n_nodes)


def planned_rounds_balanced(k_blocks: int) -> float:
    return float(k_blocks)


def closed_radius(s2: float, u: float, L: float, band: float) -> float:
    if u <= 0:
        return float("inf")
    return band + math.sqrt(2.0 * max(s2, 0.0) * L / u) + L / u


def psi_e(lam):
    lam = np.asarray(lam, float)
    return -np.log1p(-lam) - lam


_LG = np.linspace(1e-6, LAM_MAX, 20000)
_GG = _LG ** 2 / (1 - _LG) + np.log1p(-_LG) + _LG


def lam_star(ratio):
    return np.interp(ratio, _GG, _LG, left=_LG[0], right=LAM_MAX)


def oracle_radius(s2: float, u: float, L: float, band: float) -> float:
    if u <= 0:
        return float("inf")
    s2 = max(s2, 1e-18)
    lam = float(lam_star(L / (u * s2)))
    return band + (L + float(psi_e(lam)) * u * s2) / (lam * u)


def c1r_intervals(Z: np.ndarray, F: np.ndarray, n_nodes: int, L: float, look_k: Sequence[int], nplan: Sequence[float],
                  band: Sequence[float], n_arms: Optional[int] = None) -> Dict[str, np.ndarray]:
    Z = np.asarray(Z, dtype=np.int64)[:, :n_nodes]
    F = np.asarray(F, dtype=np.float64)[:, :n_nodes]
    nb, _, nout = F.shape
    K = int(n_arms if n_arms is not None else (Z.max() + 1 if Z.size else 1))
    nplan = np.asarray(nplan, float)[None, :]
    band = np.asarray(band, float)
    band = band[None, :] if band.ndim == 1 else band
    cnt = np.zeros((nout, K))
    sumf = np.zeros((nout, K))
    sv = np.zeros((nout, K))
    A = np.zeros((nout, K))
    Lam = np.zeros((nout, K))
    Vp = np.zeros((nout, K))
    looks = {int(k): i for i, k in enumerate(look_k)}
    nl = len(look_k)
    cen = np.full((nl, nout, K), 0.5)
    rad = np.full((nl, nout, K), np.inf)
    lamsum = np.zeros((nl, nout, K))
    nonempty = np.zeros((nl, nout, K))
    for i in range(nb):
        sel = (Z[i][:, None] == np.arange(K)[None, :]).astype(float)
        valid = ~np.isnan(F[i])
        w = sel[:, None, :] * valid[:, :, None]
        X = w.sum(0)
        Ys = np.einsum("nok,no->ok", w, np.nan_to_num(F[i]))
        ne = (X > 0).astype(float)
        Yb = np.where(X > 0, Ys / np.maximum(X, 1.0), 0.0)
        xhat = (PRIOR_MEAN + sumf) / (1.0 + cnt)
        s2 = (PRIOR_VAR + sv) / (1.0 + cnt)
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
                cen[j] = np.where(Lam > 0, A / Lam, 0.5)
                rad[j] = np.where(Lam > 0, band + (L + Vp) / Lam, np.inf)
            lamsum[j] = Lam
            nonempty[j] = cnt
    return {"centre": cen, "radius": rad, "lam_sum": lamsum, "nonempty": nonempty}
