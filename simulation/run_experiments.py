import sys, os, csv, time, argparse
import numpy as np
from math import ceil, log
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ciara_core import System, ARMS, LOADS, SAFE, INCUMBENT, arm_levels, arm_levels_stationary
from ciara_sim import simulate, evaluate

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_NODES = 5
DELTA = 0.05
T_REF = 20160


def make_system(t_mix, arm_idx=None, ctx_idx=None, drift=0.0, D=1):
    arm_idx = list(range(len(ARMS))) if arm_idx is None else list(arm_idx)
    ctx_idx = list(range(len(LOADS))) if ctx_idx is None else list(ctx_idx)
    arms = ARMS[arm_idx]
    loads = LOADS[ctx_idx]
    safe, inc = [], []
    for c in ctx_idx:
        safe.append([i for i, a in enumerate(arm_idx) if a in SAFE[c]])
        inc.append(arm_idx.index(INCUMBENT[c]))
    return System(t_mix=t_mix, drift=drift, D=D, arms=arms, loads=loads, safe=safe, incumbent=inc)


def burn_in(t_mix, T=T_REF):
    return int(ceil(t_mix / 2.0 * log(T)))


def run_one(cfg):
    sys_ = make_system(cfg["t_mix"], cfg.get("arm_idx"), cfg.get("ctx_idx"), cfg.get("drift", 0.0), cfg.get("D", 1))
    l, b = cfg["l"], cfg["b"]
    ck = sorted(set(max(1, T // l) for T in cfg["T_list"]))
    Tmax = max(ck) * l
    t0 = time.time()
    sn, Z = simulate(sys_, cfg["eps"], l, b, Tmax, cfg["R"], N_NODES, set(ck), cfg["seed"])
    mu = arm_levels_stationary(sys_) if cfg.get("truth", "stationary") == "stationary" else arm_levels(sys_, l)
    rows = evaluate(sys_, cfg["eps"], l, b, N_NODES, sn, Z, mu, DELTA, cfg.get("l_dep", 400),
                    do_cert=cfg.get("cert", True), max_cert_reps=cfg.get("cert_reps"))
    for r in rows:
        r["tag"] = cfg.get("tag", "")
        r["design"] = cfg.get("design", "")
        r["K"] = sys_.K
        r["X"] = sys_.X
        r["drift"] = cfg.get("drift", 0.0)
        r["secs"] = time.time() - t0
    return rows


def write_csv(name, rows):
    os.makedirs(RES, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", name, len(rows), "rows", flush=True)


def go(cfgs, name, procs):
    t0 = time.time()
    with Pool(procs) as p:
        out = p.map(run_one, cfgs, chunksize=1)
    rows = [r for o in out for r in o]
    write_csv(name, rows)
    print(name, "done in %.0f s" % (time.time() - t0), flush=True)


def exp_estimators(R, procs, quick):
    cfgs = []
    for t in (2, 5, 10, 20):
        b = burn_in(t)
        for design, l, bb in (("per-epoch", 1, 0), ("switchback-no-burnin", 2 * b, 0), ("safe-block", 2 * b, b)):
            cfgs.append(dict(t_mix=t, eps=0.4, l=l, b=bb, T_list=[20160], R=R, seed=100 + t,
                             cert=False, design=design, tag="estimators"))
    go(cfgs, "estimators.csv", procs)


def exp_shortblocks(R, procs, quick):
    cfgs = []
    for l in (1, 2, 5, 10, 25, 50, 100):
        cfgs.append(dict(t_mix=5, eps=0.4, l=l, b=0, T_list=[20160], R=R, seed=44, cert=False,
                         design="switchback-no-burnin", tag="shortblocks"))
    go(cfgs, "shortblocks.csv", procs)


EXPS = dict(shortblocks=exp_shortblocks, estimators=exp_estimators)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=min(20, os.cpu_count() or 1))
    ap.add_argument("--R", type=int, default=None)
    ap.add_argument("--out", default=None, help="output folder (default: results/ next to this script)")
    a = ap.parse_args()
    if a.out:
        RES = a.out
    names = list(EXPS) if "all" in a.exps else a.exps
    for n in names:
        R = a.R or (20 if a.quick else 100)
        EXPS[n](R, a.procs, a.quick)
