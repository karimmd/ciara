import os, sys
import numpy as np
from math import comb, log
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ciara_radius import eb_intervals
from ciara_round_radius import c1_intervals
from ciara_core import NOUT
N, q1, H, R, t, l, b = 5, 1 / 6, 6000, 200, 5, 50, 25
qv = np.array([2 / 3, q1, 1 / 6])
p = np.array([comb(N, x) * q1 ** x * (1 - q1) ** (N - x) for x in range(N + 1)])
g = lambda x: 0.20 + 0.10 * (x - 1)
mu_t = sum(x * p[x] * g(x) for x in range(1, N + 1)) / sum(x * p[x] for x in range(1, N + 1))
mu_r = sum(p[x] * g(x) for x in range(1, N + 1)) / (1 - p[0])
rng = np.random.default_rng(3)
Z = rng.choice(3, size=(R, N, H), p=qv).astype(np.int8)
X1 = (Z == 1).sum(1)
lvl = np.where(Z == 1, g(X1)[:, None, :], 0.5)
F = np.clip(lvl[..., None] + 0.01 * rng.standard_normal((R, N, H, NOUT)), 0, 1).astype(np.float32)
L = log(2 * 24 / 0.05)
looks = np.array([H])
print(f"mu_tilde = {mu_t:.4f}, mu_rw = {mu_r:.4f}")
cen, rad, _ = eb_intervals(Z, F, N, 50, 25, 5, L, looks, N * 403 * qv)
cP, rP, _ = c1_intervals(Z, F, N, 50, 25, 5, L, looks, 403, qv, "P")
cR, rR, _ = c1_intervals(Z, F, N, 50, 25, 5, L, looks, 403, qv, "R")
for name, c, r in (("NAIVE (ratio of pairs)", cen, rad), ("C1P", cP, rP), ("C1R", cR, rR)):
    c1, r1 = c[:, 0, 1, 1], r[:, 0, 1, 1]
    print(f"{name:24s} centre mean {c1.mean():.4f}  radius {np.median(r1):.4f}  covers mu_tilde {np.mean(np.abs(c1 - mu_t) <= r1):.2f}  covers mu_rw {np.mean(np.abs(c1 - mu_r) <= r1):.2f}")
