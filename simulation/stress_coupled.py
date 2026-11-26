import os, sys
import numpy as np
from math import log
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ciara_radius import eb_intervals
from ciara_round_radius import c1_intervals
from ciara_scenarios import design
from ciara_core import NOUT

def run(R, N, H, q, mode_dep, seed, M_c=1, t=5):
    rng = np.random.default_rng(seed)
    l, b = design(t)
    L = log(2 * M_c / 0.05)
    qv = np.array([q, (1 - q) / 2, (1 - q) / 2])
    Z = rng.choice(3, size=(R, N, H), p=qv).astype(np.int8)
    if mode_dep == "common":
        base = rng.integers(0, 2, size=(R, 1, H)).astype(np.float32)
        F = np.broadcast_to(base[..., None], (R, N, H, NOUT)).copy()
    elif mode_dep == "indep":
        F = rng.integers(0, 2, size=(R, N, H, NOUT)).astype(np.float32)
    elif mode_dep == "half":
        sh = rng.integers(0, 2, size=(R, 1, H, 1)).astype(np.float32)
        own = rng.integers(0, 2, size=(R, N, H, NOUT)).astype(np.float32)
        pick = rng.random((R, N, H, 1)) < 0.5
        F = np.where(pick, sh, own)
    looks = np.arange(10, H + 1, 10)
    k7 = H
    cen, rad, _ = eb_intervals(Z, F, N, l, b, t, L, looks, N * k7 * qv)
    res = {}
    res["NAIVE"] = float((np.abs(cen - 0.5) > rad)[:, :, 0, :].any((1, 2)).mean())
    cenP, radP, _ = c1_intervals(Z, F, N, l, b, t, L, looks, k7, qv, "P")
    res["C1P"] = float((np.abs(cenP - 0.5) > radP)[:, :, 0, :].any((1, 2)).mean())
    cenR, radR, _ = c1_intervals(Z, F, N, l, b, t, L, looks, k7, qv, "R")
    res["C1R"] = float((np.abs(cenR - 0.5) > np.where(np.isfinite(radR), radR, 9))[:, :, 0, :].any((1, 2)).mean())
    res["radius_ratio_med_last_arm0"] = (float(np.median(radP[:, -1, 0, 0] / rad[:, -1, 0, 0])), float(np.median(radR[:, -1, 0, 0] / rad[:, -1, 0, 0])))
    return res

if __name__ == "__main__":
    for dep in ("indep", "half", "common"):
        for N, q in ((5, 2 / 3), (5, 1 / 6)):
            r = run(4000, N, 400, q, dep, 7)
            print(f"{dep:7s} N={N} q={q:.3f}: miss rate over all looks (nominal 0.05, M_c = 1, arms pooled): NAIVE {r['NAIVE']:.4f}  C1P {r['C1P']:.4f}  C1R {r['C1R']:.4f}   radius ratio at the last look, arm 0: C1P/NAIVE {r['radius_ratio_med_last_arm0'][0]:.2f}, C1R/NAIVE {r['radius_ratio_med_last_arm0'][1]:.2f}")
