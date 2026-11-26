import csv, os, sys
import numpy as np
from math import exp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ciara_core import System, arm_levels_stationary, cycle_values, block_ops, persistence_slack

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def two_state(t, l, p=0.5):
    r = 1 - exp(-1 / t)
    rho1 = 1 - r
    A1 = (1 - rho1 ** l) / (l * r)
    A0 = 1.0 / l
    p0 = p * (1 - rho1 ** l) / (1 - p * rho1 ** l)
    J = p + p * (p0 - 1) * A1 + (1 - p) * p0 * A0
    return J - p


rows = []
for t in (5.0, 10.0, 20.0):
    for l in (1, 5, 10, 25, 50, 100, 200, 400):
        gap = abs(two_state(t, l))
        bound = persistence_slack(t, l)
        rows.append(dict(kind="two_state", t_mix=t, l=l, gap=gap, bound=bound, ratio=gap / bound))
rng = np.random.default_rng(0)
for t in (5.0, 10.0):
    s = System(t_mix=t)
    mu = arm_levels_stationary(s)
    for l in (10, 25, 50, 100, 200, 400):
        ops = block_ops(s, l)
        worst = np.zeros(4)
        for _ in range(40):
            pi = np.zeros((s.X, s.K))
            for x in range(s.X):
                pi[x, s.safe[x]] = rng.dirichlet(np.ones(len(s.safe[x])) * 0.3)
            V, J = cycle_values(s, l, pi, ops=ops)
            lin = np.einsum("kxa,xa->k", mu, pi) / s.X
            worst = np.maximum(worst, np.abs(J - lin))
        rows.append(dict(kind="sim_system", t_mix=t, l=l, gap=float(worst.max()),
                         bound=persistence_slack(t, l), ratio=float(worst.max() / persistence_slack(t, l))))
with open(os.path.join(RES, "lemma4_check.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print("%-10s t=%4.0f l=%4d gap=%.4f bound=%.4f ratio=%.3f" % (r["kind"], r["t_mix"], r["l"], r["gap"], r["bound"], r["ratio"]))
