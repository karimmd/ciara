import numpy as np
from ciara_radius import psi_e, lam_star, LAM_MAX, PRIOR_MEAN, PRIOR_VAR


def eb_path(f, L, beta, n_plan):
    n = len(f)
    cnt = sumf = sv = A = Lam = Vp = 0.0
    cen = np.zeros(n); rad = np.zeros(n)
    for j in range(n):
        xh = (PRIOR_MEAN + sumf) / (1 + cnt)
        s2 = (PRIOR_VAR + sv) / (1 + cnt)
        lam = float(lam_star(L / (n_plan * s2)))
        A += lam * f[j]; Lam += lam; Vp += psi_e(lam) * (f[j] - xh) ** 2
        sv += (f[j] - xh) ** 2; sumf += f[j]; cnt += 1
        cen[j] = A / Lam; rad[j] = beta + (L + Vp) / Lam
    return cen, rad


def main():
    lam = np.linspace(0, 0.999, 300)[:, None]
    x = np.linspace(-1, 30, 5000)[None, :]
    gap = np.log1p(lam * x) - (lam * x - psi_e(lam) * x ** 2)
    print("1. min gap (must be >= -1e-12):", gap.min())
    rng = np.random.default_rng(7)
    R, n, L = 4000, 400, np.log(2 / 0.2)
    for name in ("bern", "beta", "adv"):
        miss = 0
        for r in range(R):
            mu, beta = 0.3, 0.0
            if name == "bern":
                mu = 0.5; f = rng.binomial(1, 0.5, n).astype(float)
            elif name == "beta":
                a_ = mu * (mu * (1 - mu) / 0.03 ** 2 - 1); b_ = (1 - mu) * (mu * (1 - mu) / 0.03 ** 2 - 1)
                f = rng.beta(a_, b_, n)
            else:
                beta = 0.02; f = np.zeros(n); s = 0.0
                for j in range(n):
                    m = mu + beta if (j == 0 or s / j >= mu) else mu - beta
                    f[j] = np.clip(rng.normal(m, 0.03), 0, 1)
                    s += f[j]
            cen, rad = eb_path(f, L, beta, n)
            miss += bool((np.abs(cen - mu) > rad).any())
        print(f"2{name}: miss over all prefixes {miss / R:.4f} (bound {2 * np.exp(-L):.3f})")
    from ciara_radius import eb_intervals
    from ciara_core import NOUT
    Z = rng.integers(0, 3, (2, 1, 300)).astype(np.int8)
    F = rng.random((2, 1, 300, NOUT)).astype(np.float32) * 0.2 + 0.4
    looks = [100, 200, 300]
    cen, rad, _ = eb_intervals(Z, F, 1, 50, 25, 5.0, 6.0, looks, [120.0, 60.0, 60.0])
    from ciara_core import dm_bias
    err = 0.0
    for r in range(2):
        for a in range(3):
            for o in range(NOUT):
                for j, k in enumerate(looks):
                    f = F[r, 0, :k, o][Z[r, 0, :k] == a].astype(float)
                    c2, r2 = eb_path(f, 6.0, dm_bias(5.0, 50, 25), [120.0, 60.0, 60.0][a])
                    err = max(err, abs(c2[-1] - cen[r, j, o, a]), abs(r2[-1] - rad[r, j, o, a]))
    print("3. max |vectorized - scalar| (must be < 1e-9):", err)
    print("LAM_MAX", LAM_MAX)


if __name__ == "__main__":
    main()
