import numpy as np
from math import ceil, log
from ciara_core import System, ARMS, LOADS, NOUT

T_REF = 20160


def scenario_system(name, t_mix):
    arms = ARMS[[1, 5, 6]].copy()
    if name == "K3X1":
        pass
    elif name == "small":
        arms[1, 0], arms[2, 0] = 0.60, 0.62
    elif name == "null":
        arms[1, 0], arms[2, 0] = 0.545, 0.50
    elif name == "trap":
        arms[1, 2], arms[2, 2] = 0.16, 0.21
    else:
        raise ValueError(name)
    return System(t_mix=t_mix, arms=arms, loads=LOADS[[1]], safe=[[0, 1, 2]], incumbent=[0])


def design(t_mix):
    b = int(ceil(t_mix / 2.0 * log(T_REF)))
    return 2 * b, b


def stationary_levels(sys_):
    S = sys_.S
    mu = np.zeros((NOUT, sys_.K))
    for a in range(sys_.K):
        P = sys_.P[0, 0, a]
        w, v = np.linalg.eig(P.T)
        f = np.real(v[:, np.argmin(np.abs(w - 1))])
        f = np.abs(f) / np.abs(f).sum()
        mu[:, a] = sys_.m[:, 0, 0, a] @ f
    return mu


def policy_values(ops, pi):
    T, O = ops
    T0, O0 = T[0], O[0]
    S = T0.shape[-1]
    Tm = np.einsum("ba,ast->bst", pi, T0)
    A = np.transpose(Tm - np.eye(S)[None], (0, 2, 1)).copy()
    A[:, -1, :] = 1.0
    rhs = np.zeros((pi.shape[0], S))
    rhs[:, -1] = 1.0
    f = np.linalg.solve(A, rhs[..., None])[..., 0]
    Om = np.einsum("ba,aks->bks", pi, O0)
    return np.einsum("bks,bs->bk", Om, f)
