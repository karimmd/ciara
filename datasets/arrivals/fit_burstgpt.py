import argparse
import csv
import json
import math
import random
import statistics

from ciara_clients.mmpp import fit_poisson_hmm

BIN = 10
H = 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="BurstGPT_1.csv (columns Timestamp, Log Type, ...)")
    ap.add_argument("--band", type=float, default=0.25, help="keep hours within +-band of the median of the busiest half of the hours")
    a = ap.parse_args()
    ts = sorted(int(float(r["Timestamp"])) for r in csv.DictReader(open(a.trace)) if r["Log Type"] == "API log")
    nh = ts[-1] // H + 1
    hour = [0] * nh
    for t in ts:
        hour[t // H] += 1
    busy = sorted(hour)[nh // 2:]
    med = statistics.median(busy)
    keep = [h for h in range(nh) if (1 - a.band) * med <= hour[h] <= (1 + a.band) * med]
    runs, cur = [], []
    for h in keep:
        if cur and h == cur[-1] + 1:
            cur.append(h)
        else:
            if len(cur) >= 3:
                runs.append(cur)
            cur = [h]
    if len(cur) >= 3:
        runs.append(cur)
    counts_all = []
    segs = []
    idx = 0
    for run in runs:
        lo, hi = run[0] * H, (run[-1] + 1) * H
        c = [0] * ((hi - lo) // BIN)
        while idx < len(ts) and ts[idx] < lo:
            idx += 1
        j = idx
        while j < len(ts) and ts[j] < hi:
            c[(ts[j] - lo) // BIN] += 1
            j += 1
        segs.append(c)
        counts_all += c
    print("api arrivals", len(ts), "hours", nh, "busy-half median per hour", med, "kept hours", len(keep), "runs", len(runs), "bins", len(counts_all))
    fit = fit_poisson_hmm(counts_all, BIN, k=2, max_iter=150)
    print(json.dumps({"rel_rate": fit.rel_rate, "mean_dwell_s": fit.mean_dwell_s, "stationary": fit.stationary, "log_lik": fit.log_lik, "iterations": fit.iterations}, indent=1))

    def cv60(c10):
        c = [sum(c10[i:i + 6]) for i in range(0, len(c10) - 5, 6)]
        m = statistics.mean(c)
        return statistics.pstdev(c) / m, m

    cvd = [cv60(s) for s in segs]
    w = sum(m for _, m in cvd)
    print("data CV of 60 s counts (rate-weighted mean over runs):", round(sum(cv * m for cv, m in cvd) / w, 3))
    random.seed(1)
    lam_mean = statistics.mean(counts_all) / BIN
    rr, dw, pi = fit.rel_rate, fit.mean_dwell_s, fit.stationary
    norm = sum(p * r for p, r in zip(pi, rr))
    rates = [lam_mean * r / norm for r in rr]
    T = 30 * 24 * 3600
    t = 0.0
    s = 0
    c60 = [0] * (T // 60)
    while t < T:
        d = random.expovariate(1 / dw[s])
        end = min(T, t + d)
        while True:
            t += random.expovariate(rates[s])
            if t >= end:
                break
            c60[int(t // 60)] += 1
        t = end
        s = 1 - s
    print("fitted MMPP CV of 60 s counts:", round(statistics.pstdev(c60) / statistics.mean(c60), 3), "Poisson CV:", round(1 / math.sqrt(statistics.mean(c60)), 3))


if __name__ == "__main__":
    main()
