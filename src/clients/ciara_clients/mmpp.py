from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .rng import expo, pick


@dataclass
class MMPPSpec:
    states: List[str]
    rel_rate: List[float]
    mean_dwell_s: List[float]
    jump: Optional[List[List[float]]] = None

    def __post_init__(self) -> None:
        k = len(self.states)
        if not (len(self.rel_rate) == len(self.mean_dwell_s) == k and k >= 1):
            raise ValueError("states, rel_rate and mean_dwell_s must have the same length")
        if any(d <= 0 for d in self.mean_dwell_s) or any(r < 0 for r in self.rel_rate):
            raise ValueError("dwell times must be positive and relative rates non-negative")
        if self.jump is None:
            self.jump = [[0.0 if i == j else (1.0 / (k - 1) if k > 1 else 0.0) for j in range(k)] for i in range(k)]
        for i, row in enumerate(self.jump):
            if k > 1 and (abs(sum(row) - 1.0) > 1e-9 or row[i] != 0.0):
                raise ValueError("jump rows must sum to 1 with a zero diagonal")

    @property
    def k(self) -> int:
        return len(self.states)


def embedded_stationary(jump: Sequence[Sequence[float]], iters: int = 5000) -> List[float]:
    k = len(jump)
    if k == 1:
        return [1.0]
    v = [1.0 / k] * k
    for _ in range(iters):
        w = [sum(v[i] * jump[i][j] for i in range(k)) for j in range(k)]
        s = sum(w)
        w = [x / s for x in w]
        w = [0.5 * a + 0.5 * b for a, b in zip(v, w)]
        if max(abs(a - b) for a, b in zip(v, w)) < 1e-15:
            v = w
            break
        v = w
    return v


def stationary(spec: MMPPSpec) -> List[float]:
    nu = embedded_stationary(spec.jump)
    w = [n * d for n, d in zip(nu, spec.mean_dwell_s)]
    s = sum(w)
    return [x / s for x in w]


def state_rates(spec: MMPPSpec, base_rate: float) -> List[float]:
    pi = stationary(spec)
    norm = sum(p * r for p, r in zip(pi, spec.rel_rate))
    return [base_rate * r / norm for r in spec.rel_rate]


def realized_rates(spec: MMPPSpec, base_rate: float, path: Sequence[Tuple[float, int]], horizon_s: float) -> List[float]:
    occ = [0.0] * spec.k
    for idx, (a, s) in enumerate(path):
        b = min(path[idx + 1][0] if idx + 1 < len(path) else horizon_s, horizon_s)
        occ[s] += max(0.0, b - a) / horizon_s
    norm = sum(o * r for o, r in zip(occ, spec.rel_rate))
    if norm <= 0.0:
        return state_rates(spec, base_rate)
    return [base_rate * r / norm for r in spec.rel_rate]


def simulate_path(rng: random.Random, spec: MMPPSpec, horizon_s: float) -> List[Tuple[float, int]]:
    pi = stationary(spec)
    s = pick(rng, pi)
    t = 0.0
    path: List[Tuple[float, int]] = []
    while t < horizon_s:
        path.append((t, s))
        t += expo(rng, 1.0 / spec.mean_dwell_s[s])
        if spec.k > 1:
            s = pick(rng, spec.jump[s])
    return path


def _pieces(a: float, b: float, bounds: Optional[Sequence[float]], scales: Optional[Sequence[float]]):
    if not bounds or not scales:
        yield a, b, 1.0
        return
    import bisect

    k = max(0, bisect.bisect_right(bounds, a) - 1)
    t = a
    while t < b:
        kk = min(k, len(scales) - 1)
        end = min(b, bounds[k + 1]) if k + 1 < len(bounds) else b
        if end <= t:
            k += 1
            continue
        yield t, end, scales[kk]
        t = end
        k += 1


def arrivals_from_path(rng: random.Random, path: Sequence[Tuple[float, int]], rates: Sequence[float],
                       horizon_s: float, mult: float = 1.0, bounds: Optional[Sequence[float]] = None,
                       scales: Optional[Sequence[float]] = None) -> List[float]:
    out: List[float] = []
    for idx, (a, s) in enumerate(path):
        b = path[idx + 1][0] if idx + 1 < len(path) else horizon_s
        b = min(b, horizon_s)
        if b <= a:
            continue
        for pa, pb, sc in _pieces(a, b, bounds, scales):
            r = mult * rates[s] * sc
            if r <= 0.0 or pb <= pa:
                continue
            t = pa
            while True:
                t += expo(rng, r)
                if t >= pb:
                    break
                out.append(t)
    return out


def poisson_arrivals(rng: random.Random, rate: float, horizon_s: float, bounds: Optional[Sequence[float]] = None,
                     scales: Optional[Sequence[float]] = None) -> List[float]:
    out: List[float] = []
    if rate <= 0.0:
        return out
    for pa, pb, sc in _pieces(0.0, horizon_s, bounds, scales):
        r = rate * sc
        if r <= 0.0 or pb <= pa:
            continue
        t = pa
        while True:
            t += expo(rng, r)
            if t >= pb:
                break
            out.append(t)
    return out


def count_cv_two_state(spec: MMPPSpec, base_rate: float, window_s: float) -> Tuple[float, float]:
    if spec.k != 2:
        raise ValueError("closed form is for two states")
    pi = stationary(spec)
    lam = state_rates(spec, base_rate)
    mean_rate = sum(p * x for p, x in zip(pi, lam))
    r1, r2 = 1.0 / spec.mean_dwell_s[0], 1.0 / spec.mean_dwell_s[1]
    q = r1 + r2
    var = mean_rate * window_s + 2.0 * pi[0] * pi[1] * (lam[0] - lam[1]) ** 2 * (window_s / q - (1.0 - math.exp(-q * window_s)) / (q * q))
    m = mean_rate * window_s
    return m, math.sqrt(var) / m if m > 0 else float("nan")


def _poisson_logpmf(k: int, lam: float) -> float:
    if lam <= 0.0:
        return 0.0 if k == 0 else -1e30
    return k * math.log(lam) - lam - math.lgamma(k + 1)


@dataclass
class FitResult:
    k: int
    bin_s: float
    rate_per_bin: List[float]
    trans: List[List[float]]
    log_lik: float
    iterations: int
    rel_rate: List[float] = field(default_factory=list)
    mean_dwell_s: List[float] = field(default_factory=list)
    jump: List[List[float]] = field(default_factory=list)
    stationary: List[float] = field(default_factory=list)


def fit_poisson_hmm(counts: Sequence[int], bin_s: float, k: int = 2, max_iter: int = 200, tol: float = 1e-6) -> FitResult:
    n = len(counts)
    if n < 10 * k:
        raise ValueError("too few bins for the requested number of states")
    srt = sorted(counts)
    mean = sum(counts) / n
    lam = [max(1e-3, srt[min(n - 1, int((i + 0.5) / k * n))] + 0.1 * (i + 1) * (mean + 0.1)) for i in range(k)]
    lam.sort()
    A = [[(0.9 if i == j else 0.1 / (k - 1)) if k > 1 else 1.0 for j in range(k)] for i in range(k)]
    pi0 = [1.0 / k] * k
    prev = -1e300
    it = 0
    ll = prev
    for it in range(1, max_iter + 1):
        B = [[math.exp(_poisson_logpmf(counts[t], lam[i])) for i in range(k)] for t in range(n)]
        alpha = [[0.0] * k for _ in range(n)]
        c = [0.0] * n
        for i in range(k):
            alpha[0][i] = pi0[i] * B[0][i]
        c[0] = sum(alpha[0]) or 1e-300
        alpha[0] = [x / c[0] for x in alpha[0]]
        for t in range(1, n):
            for j in range(k):
                alpha[t][j] = sum(alpha[t - 1][i] * A[i][j] for i in range(k)) * B[t][j]
            c[t] = sum(alpha[t]) or 1e-300
            alpha[t] = [x / c[t] for x in alpha[t]]
        ll = sum(math.log(x) for x in c)
        beta = [[1.0] * k for _ in range(n)]
        for t in range(n - 2, -1, -1):
            for i in range(k):
                beta[t][i] = sum(A[i][j] * B[t + 1][j] * beta[t + 1][j] for j in range(k)) / c[t + 1]
        gamma = [[alpha[t][i] * beta[t][i] for i in range(k)] for t in range(n)]
        for t in range(n):
            s = sum(gamma[t]) or 1e-300
            gamma[t] = [x / s for x in gamma[t]]
        xi_sum = [[0.0] * k for _ in range(k)]
        for t in range(n - 1):
            for i in range(k):
                for j in range(k):
                    xi_sum[i][j] += alpha[t][i] * A[i][j] * B[t + 1][j] * beta[t + 1][j] / c[t + 1]
        for i in range(k):
            denom = sum(gamma[t][i] for t in range(n))
            lam[i] = max(1e-6, sum(gamma[t][i] * counts[t] for t in range(n)) / max(denom, 1e-300))
            row = sum(xi_sum[i]) or 1e-300
            A[i] = [xi_sum[i][j] / row for j in range(k)]
        pi0 = gamma[0]
        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            break
        prev = ll
    order = sorted(range(k), key=lambda i: lam[i])
    lam = [lam[i] for i in order]
    A = [[A[i][j] for j in order] for i in order]
    res = FitResult(k=k, bin_s=bin_s, rate_per_bin=lam, trans=A, log_lik=ll, iterations=it)
    dwell, jump = [], []
    for i in range(k):
        stay = min(max(A[i][i], 0.0), 1.0 - 1e-9)
        dwell.append(bin_s / (1.0 - stay))
        off = [A[i][j] for j in range(k) if j != i]
        s = sum(off) or 1.0
        jump.append([0.0 if j == i else A[i][j] / s for j in range(k)])
    res.mean_dwell_s = dwell
    res.jump = jump
    res.rel_rate = [x / lam[0] if lam[0] > 0 else float("nan") for x in lam]
    spec = MMPPSpec([str(i) for i in range(k)], res.rel_rate, dwell, jump)
    res.stationary = stationary(spec)
    return res


def bin_counts(times_s: Sequence[float], bin_s: float) -> List[int]:
    if not times_s:
        return []
    t0 = min(times_s)
    nb = int((max(times_s) - t0) // bin_s) + 1
    counts = [0] * nb
    for t in times_s:
        counts[int((t - t0) // bin_s)] += 1
    return counts


def read_times(path: str, time_col: str, max_rows: Optional[int] = None) -> List[float]:
    out: List[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if time_col not in (rd.fieldnames or []):
            raise SystemExit(f"column {time_col!r} not in {rd.fieldnames}")
        for i, row in enumerate(rd):
            if max_rows and i >= max_rows:
                break
            out.append(float(row[time_col]))
    return out


def _fmt(xs: Sequence[float], nd: int = 4) -> str:
    return "[" + ", ".join(f"{x:.{nd}g}" for x in xs) + "]"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MMPP tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit a K-state MMPP (Poisson HMM) to a CSV of request timestamps")
    f.add_argument("--csv", required=True)
    f.add_argument("--time-col", required=True, help="column with arrival times in seconds")
    f.add_argument("--bin-s", type=float, default=1.0)
    f.add_argument("--states", type=int, default=2)
    f.add_argument("--time-scale", type=float, default=1.0, help="run time = trace time / time-scale (compression)")
    f.add_argument("--max-rows", type=int, default=None)
    t = sub.add_parser("selftest", help="simulate a known MMPP, fit it back, print both")
    t.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        times = read_times(a.csv, a.time_col, a.max_rows)
        counts = bin_counts(times, a.bin_s)
        res = fit_poisson_hmm(counts, a.bin_s, a.states)
        sc = a.time_scale
        print(f"bins={len(counts)} mean_count_per_bin={sum(counts) / len(counts):.4g} log_lik={res.log_lik:.1f} iters={res.iterations}")
        print(f"rate_per_s_in_trace_time = {_fmt([x / a.bin_s for x in res.rate_per_bin])}")
        print(f"stationary = {_fmt(res.stationary)}")
        print("# paste into [mmpp] of configs/clients/default.toml")
        print(f"rel_rate = {_fmt(res.rel_rate)}")
        print(f"mean_dwell_s = {_fmt([d / sc for d in res.mean_dwell_s])}   # trace dwell divided by time_scale={sc}")
        if a.states > 2:
            print(f"jump = {res.jump}")
        return 0
    if a.cmd == "selftest":
        from .rng import stream

        spec = MMPPSpec(["calm", "burst"], [1.0, 10.0], [150.0, 20.0])
        rng = stream(a.seed, "selftest")
        horizon = 30000.0
        rates = state_rates(spec, 2.0)
        path = simulate_path(rng, spec, horizon)
        times = arrivals_from_path(rng, path, rates, horizon)
        counts = bin_counts(times, 1.0)
        res = fit_poisson_hmm(counts, 1.0, 2)
        print("true  rate/s:", _fmt(rates), "dwell:", spec.mean_dwell_s, "rel:", spec.rel_rate)
        print("fit   rate/s:", _fmt(res.rate_per_bin), "dwell:", _fmt(res.mean_dwell_s), "rel:", _fmt(res.rel_rate))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
