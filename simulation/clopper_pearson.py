from math import lgamma, log, exp


def cp_upper(x, n, alpha=0.05):
    if x >= n:
        return 1.0
    def cdf(p):
        if p <= 0: return 1.0
        s = 0.0
        for k in range(0, int(x) + 1):
            s += exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1 - p))
        return s
    lo, hi = x / n, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cdf(mid) > alpha: lo = mid
        else: hi = mid
    return hi
