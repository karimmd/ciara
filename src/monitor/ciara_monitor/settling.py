from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ciara.band import beta_dm
from ciara.contraction import SETTLE_SE_MULT

from .fileio import now_cst, write_json, write_text

EPOCH_NS = 30 * 10 ** 9
BLOCK_EPOCHS = 50
PLATEAU = (30, 50)
SHOW = 30
MIN_PLATEAU = 10
MIN_BLOCKS = 5
BURN_IN, WINDOW_END = 25, 50
SERIES = ("cloud_queue", "mean_latency_s", "tail_f1", "tail_f2", "deadline_miss", "cloud_share")
LABEL = {"cloud_queue": "cloud queue", "mean_latency_s": "mean latency", "tail_f1": "family 1 tail", "tail_f2": "family 2 tail", "deadline_miss": "deadline misses", "cloud_share": "cloud share"}
T_BAR_PLANNED = 5.0
T_GRID = np.exp(np.linspace(math.log(0.25), math.log(60.0), 600))
T_MAX = float(T_GRID[-1])
_E = np.exp(-np.arange(30, dtype=float)[None, :] / T_GRID[:, None])
NULL_REPS = 200
MIN_CLUSTERS = 5


def iter_jsonl(path: str, needle: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    with open(path) as fh:
        for line in fh:
            if needle is not None and needle not in line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


class Grid:
    def __init__(self, epoch_rows: Iterable[Dict[str, Any]]):
        per: Dict[str, List[Tuple[int, int, int]]] = {}
        for e in epoch_rows:
            if e.get("rec", "epoch") != "epoch" or e.get("t_start_mono_ns") is None:
                continue
            per.setdefault(e["node"], []).append((int(e["t_start_mono_ns"]), int(e["epoch"]), int(e["block_id"]) * BLOCK_EPOCHS + int(e["epoch_in_block"])))
        self.start: Dict[str, List[int]] = {}
        self.epoch: Dict[str, List[int]] = {}
        self.slot: Dict[str, List[int]] = {}
        for n, v in per.items():
            v.sort()
            self.start[n] = [x[0] for x in v]
            self.epoch[n] = [x[1] for x in v]
            self.slot[n] = [x[2] for x in v]

    def locate(self, node: str, t_ns: float) -> Optional[Tuple[int, int]]:
        s = self.start.get(node)
        if not s:
            return None
        i = bisect.bisect_right(s, t_ns) - 1
        if i < 0 or t_ns >= s[i] + EPOCH_NS * 1.02:
            return None
        return self.epoch[node][i], self.slot[node][i]


def accumulate(requests: Iterable[Dict[str, Any]], cloud_arrivals: Iterable[Dict[str, Any]], grid: Grid) -> Dict[Tuple[str, int], List[float]]:
    acc: Dict[Tuple[str, int], List[float]] = {}
    for r in requests:
        t = r.get("t_arrival")
        node = r.get("node")
        if t is None or node is None:
            continue
        loc = grid.locate(node, float(t))
        if loc is None:
            continue
        a = acc.setdefault((node, loc[0]), [0.0] * 12)
        a[0] += 1
        lat = r.get("latency_s")
        if r.get("ok") and lat is not None and math.isfinite(float(lat)):
            a[1] += float(lat)
            a[2] += 1
        ex = r.get("exceed")
        if ex is not None:
            k = 3 if int(r.get("family") or 0) == 1 else 5 if int(r.get("family") or 0) == 2 else None
            if k is not None:
                a[k] += 1
                a[k + 1] += float(ex)
        dv = r.get("deadline_violation")
        if dv is not None:
            a[7] += 1
            a[8] += float(dv)
        a[9] += 1.0 if r.get("cloud_used") else 0.0
    for c in cloud_arrivals:
        t = c.get("t_arrive")
        node = c.get("node")
        if t is None or node is None:
            continue
        loc = grid.locate(node, float(t))
        if loc is None:
            continue
        a = acc.setdefault((node, loc[0]), [0.0] * 12)
        a[10] += 1
        a[11] += float(c.get("queue_len_at_arrival") or 0) + float(c.get("in_flight_at_arrival") or 0)
    return acc


def epoch_values(a: Optional[List[float]]) -> Dict[str, Optional[float]]:
    if a is None:
        return {s: None for s in SERIES}
    return {"cloud_queue": a[11] / a[10] if a[10] > 0 else None, "mean_latency_s": a[1] / a[2] if a[2] > 0 else None, "tail_f1": a[4] / a[3] if a[3] > 0 else None,
            "tail_f2": a[6] / a[5] if a[5] > 0 else None, "deadline_miss": a[8] / a[7] if a[7] > 0 else None, "cloud_share": a[9] / a[0] if a[0] > 0 else None}


def switch_blocks(run_dir: str, grid: Grid, acc: Dict[Tuple[str, int], List[float]], k_rounds: Optional[int] = None) -> List[Dict[str, Any]]:
    arm: Dict[Tuple[str, int], str] = {}
    psw: Dict[Tuple[str, int], bool] = {}
    done: Dict[Tuple[str, int], bool] = {}
    for r in iter_jsonl(os.path.join(run_dir, "blocks.jsonl")):
        key = (r.get("node"), int(r.get("block_id", -1)))
        if r.get("rec") == "block":
            arm[key] = r.get("arm_id")
            psw[key] = bool(r.get("profile_switched"))
        elif r.get("rec") == "block_end":
            done[key] = r.get("status") == "complete" and int(r.get("epochs_logged") or 0) >= BLOCK_EPOCHS
    return build_blocks(arm, psw, done, grid, acc, k_rounds)


def build_blocks(arm: Dict[Tuple[str, int], str], psw: Dict[Tuple[str, int], bool], done: Dict[Tuple[str, int], bool], grid: Grid, acc: Dict[Tuple[str, int], List[float]],
                 k_rounds: Optional[int] = None) -> List[Dict[str, Any]]:
    slot_epoch: Dict[str, Dict[int, int]] = {n: dict(zip(grid.slot[n], grid.epoch[n])) for n in grid.start}
    out: List[Dict[str, Any]] = []
    for (node, b), a in sorted(arm.items(), key=lambda kv: (kv[0][1], str(kv[0][0]))):
        prev = arm.get((node, b - 1))
        if prev is None or prev == a or not done.get((node, b)) or (k_rounds is not None and b >= k_rounds):
            continue
        curves: Dict[str, List[Optional[float]]] = {s: [] for s in SERIES}
        for k in range(BLOCK_EPOCHS):
            e = slot_epoch.get(node, {}).get(b * BLOCK_EPOCHS + k)
            v = epoch_values(acc.get((node, e))) if e is not None else epoch_values(None)
            for s in SERIES:
                curves[s].append(v[s])
        out.append({"node": node, "round": b, "arm": a, "prev": prev, "profile_switched": psw.get((node, b), False), "curves": curves})
    return out


def deviations(blocks: Sequence[Dict[str, Any]], series: str) -> np.ndarray:
    D = np.full((len(blocks), BLOCK_EPOCHS), np.nan)
    for i, b in enumerate(blocks):
        x = np.array([np.nan if v is None else float(v) for v in b["curves"][series]])
        pl = x[PLATEAU[0]:PLATEAU[1]]
        if np.isfinite(pl).sum() < MIN_PLATEAU:
            continue
        D[i] = x - np.nanmean(pl)
    return D


def cluster_starts(rounds: Sequence[int]) -> np.ndarray:
    r = np.asarray(rounds)
    return np.flatnonzero(np.r_[True, r[1:] != r[:-1]]) if len(r) else np.array([], dtype=int)


def mean_se(D: np.ndarray, starts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fin = np.isfinite(D)
    n = fin.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(fin, D, 0.0).sum(axis=0) / n
        Z = np.where(fin, D - m[None, :], 0.0)
        T = np.add.reduceat(Z, starts, axis=0)
        G = (np.add.reduceat(fin.astype(float), starts, axis=0) > 0).sum(axis=0).astype(float)
        se = np.sqrt(G / (G - 1.0) * (T ** 2).sum(axis=0)) / n
    ok = (n >= MIN_BLOCKS) & (G >= MIN_CLUSTERS)
    return np.where(ok, m, np.nan), np.where(ok, se, np.nan), n


def settled_epoch(m: np.ndarray, se: np.ndarray, mult: float, horizon: int = SHOW) -> Optional[int]:
    last = -1
    for j in range(horizon):
        if np.isfinite(m[j]) and np.isfinite(se[j]) and se[j] > 0 and abs(m[j]) > mult * se[j]:
            last = j
    return None if last == horizon - 1 else last + 1


def _profile(M: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    chi = np.zeros(len(T_GRID))
    amp = np.zeros((len(T_GRID), M.shape[0]))
    null = 0.0
    for i in range(M.shape[0]):
        ok = np.isfinite(M[i]) & np.isfinite(S[i]) & (S[i] > 0)
        if ok.sum() < 3:
            continue
        w = 1.0 / S[i][ok] ** 2
        m = M[i][ok]
        e = _E[:, ok]
        num = (e * (w * m)[None, :]).sum(axis=1)
        den = (e * e * w[None, :]).sum(axis=1)
        a = num / den
        c0 = float((w * m * m).sum())
        chi += c0 - a * num
        amp[:, i] = a
        null += c0
    return chi, amp, null


def fit_decay(M: np.ndarray, S: np.ndarray, crit: float = 0.0) -> Dict[str, Any]:
    chi, amp, null = _profile(M, S)
    j = int(np.argmin(chi))
    gain = null - float(chi[j])
    return {"t_bar": float(T_GRID[j]), "amplitude": [float(x) for x in amp[j]], "chi2": float(chi[j]), "chi2_null": float(null), "chi2_gain": float(gain), "gain_critical_5pct": float(crit),
            "identified": bool(gain >= crit)}


def null_gain_critical(D: np.ndarray, starts: np.ndarray, reps: int = NULL_REPS, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    sizes = np.diff(np.r_[starts, D.shape[0]])
    g = []
    for _ in range(reps):
        sgn = np.repeat(rng.choice([-1.0, 1.0], size=len(starts)), sizes)
        m, se, _ = mean_se(D * sgn[:, None], starts)
        chi, _, null = _profile(m[None, :SHOW], se[None, :SHOW])
        g.append(null - float(chi.min()))
    return float(np.quantile(g, 0.95))


def joint_fit(M: np.ndarray, S: np.ndarray, crits: Sequence[float]) -> Dict[str, Any]:
    sel = []
    for i in range(M.shape[0]):
        f = fit_decay(M[i:i + 1], S[i:i + 1], crits[i])
        if f["identified"]:
            sel.append(i)
    if not sel:
        return {"selected": [], "t_bar": None, "amplitude": [], "identified": False}
    f = fit_decay(M[sel], S[sel])
    return {"selected": sel, "t_bar": f["t_bar"], "amplitude": f["amplitude"], "chi2_gain": f["chi2_gain"], "identified": True}


def bootstrap_t_bar(Ds: Sequence[np.ndarray], rounds: Sequence[int], crits: Sequence[float], reps: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    r = np.asarray(rounds)
    ur = np.unique(r)
    idx_of = {int(x): np.flatnonzero(r == x) for x in ur}
    joint: List[Optional[float]] = []
    single: List[List[Optional[float]]] = [[] for _ in Ds]
    for _ in range(reps):
        pick = rng.choice(ur, size=len(ur), replace=True)
        parts = [idx_of[int(x)] for x in pick]
        idx = np.concatenate(parts)
        starts = np.r_[0, np.cumsum([len(p) for p in parts])[:-1]]
        M = np.zeros((len(Ds), SHOW))
        S = np.zeros((len(Ds), SHOW))
        for i, D in enumerate(Ds):
            m, se, _ = mean_se(D[idx], starts)
            M[i], S[i] = m[:SHOW], se[:SHOW]
            f = fit_decay(M[i:i + 1], S[i:i + 1], crits[i])
            single[i].append(f["t_bar"] if f["identified"] else None)
        jf = joint_fit(M, S, crits)
        joint.append(jf["t_bar"] if jf["identified"] else None)
    return {"joint": _interval(joint, reps), "single": [_interval(v, reps) for v in single]}


def _interval(vals: Sequence[Optional[float]], reps: int) -> Dict[str, Any]:
    got = np.array([v for v in vals if v is not None], dtype=float)
    cons = np.array([T_MAX if v is None else v for v in vals], dtype=float)
    return {"reps": reps, "identified_fraction": len(got) / reps, "lo": float(np.quantile(got, 0.025)) if len(got) >= 20 else None, "hi": float(np.quantile(got, 0.975)) if len(got) >= 20 else None,
            "median": float(np.median(got)) if len(got) else None, "hi_conservative": float(np.quantile(cons, 0.975))}


def window_bias(D: np.ndarray, starts: np.ndarray) -> Dict[str, Any]:
    lo = BURN_IN
    b = np.array([np.nanmean(D[i, lo:PLATEAU[0]]) * (PLATEAU[0] - lo) / (WINDOW_END - lo) if np.isfinite(D[i, lo:PLATEAU[0]]).sum() >= 3 else np.nan for i in range(D.shape[0])])
    m, se, _ = mean_se(b[:, None], starts)
    return {"mean": float(m[0]) if np.isfinite(m[0]) else None, "se": float(se[0]) if np.isfinite(se[0]) else None}


def group_result(blocks: Sequence[Dict[str, Any]], reps: int, seed: int, with_ci: bool = True) -> Dict[str, Any]:
    blocks = sorted(blocks, key=lambda b: b["round"])
    rounds = [b["round"] for b in blocks]
    starts = cluster_starts(rounds)
    res: Dict[str, Any] = {"n_blocks": len(blocks), "n_rounds": len(starts), "series": {}}
    Ds, Ms, Ss, crits = [], [], [], []
    for i, s in enumerate(SERIES):
        D = deviations(blocks, s)
        Ds.append(D)
        m, se, n = mean_se(D, starts)
        Ms.append(m[:SHOW])
        Ss.append(se[:SHOW])
        crit = null_gain_critical(D, starts, seed=seed + i) if len(starts) >= MIN_CLUSTERS else float("inf")
        crits.append(crit)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = m / se
        res["series"][s] = {"mean": _lst(m[:SHOW]), "se": _lst(se[:SHOW]), "n": [int(x) for x in n[:SHOW]], "t": _lst(t[:SHOW]),
                            "settled_epoch_b": settled_epoch(m, se, SETTLE_SE_MULT), "settled_epoch_t3": settled_epoch(m, se, 3.0),
                            "max_abs_t_epochs_1_to_29": _fmax(np.abs(t[1:SHOW])), "fit": fit_decay(m[None, :SHOW], se[None, :SHOW], crit), "window_bias": window_bias(D, starts)}
    M, S = np.array(Ms), np.array(Ss)
    joint = joint_fit(M, S, crits)
    joint["selected"] = [SERIES[i] for i in joint["selected"]]
    if with_ci and len(blocks) >= 2 * MIN_BLOCKS and len(starts) >= MIN_CLUSTERS:
        bs = bootstrap_t_bar(Ds, rounds, crits, reps, seed)
        joint["ci"] = bs["joint"]
        for i, s in enumerate(SERIES):
            res["series"][s]["fit"]["ci"] = bs["single"][i]
    res["joint"] = joint
    return res


def _lst(a: np.ndarray) -> List[Optional[float]]:
    return [float(x) if np.isfinite(x) else None for x in a]


def _fmax(a: np.ndarray) -> Optional[float]:
    a = a[np.isfinite(a)]
    return float(a.max()) if len(a) else None


def analyze(blocks: Sequence[Dict[str, Any]], reps: int = 400, seed: int = 20261029) -> Dict[str, Any]:
    arms = sorted({b["arm"] for b in blocks} | {b["prev"] for b in blocks})
    groups: Dict[str, List[Dict[str, Any]]] = {"pooled": list(blocks), "pooled, engine switch": [b for b in blocks if b.get("profile_switched")],
                                               "pooled, admission only": [b for b in blocks if not b.get("profile_switched")]}
    for a in arms:
        groups[f"into {a}"] = [b for b in blocks if b["arm"] == a]
    for a in arms:
        groups[f"out of {a}"] = [b for b in blocks if b["prev"] == a]
    res: Dict[str, Any] = {"written": now_cst(), "t_bar_planned": T_BAR_PLANNED, "settle_threshold_se": SETTLE_SE_MULT, "n_switch_blocks": len(blocks),
                           "n_profile_switch": sum(1 for b in blocks if b.get("profile_switched")), "reps": reps, "groups": {},
                           "band_planned": {"t_bar_5": beta_dm(5.0, BURN_IN, WINDOW_END), "t_bar_8": beta_dm(8.0, BURN_IN, WINDOW_END)}}
    for i, (name, bl) in enumerate(groups.items()):
        if len({b["round"] for b in bl}) >= MIN_CLUSTERS:
            res["groups"][name] = group_result(bl, reps, seed + 10 * i, with_ci=True)
    pj = (res["groups"].get("pooled") or {}).get("joint") or {}
    ci = pj.get("ci") or {}
    res["t_bar"] = pj.get("t_bar")
    res["t_bar_ci"] = [ci.get("lo"), ci.get("hi")] if (ci.get("hi") is not None and pj.get("t_bar") is not None) else None
    res["t_bar_hi_conservative"] = ci.get("hi_conservative")
    res["identified_fraction"] = ci.get("identified_fraction")
    res["selected_series"] = pj.get("selected")
    res["identified"] = bool(pj.get("identified"))
    res["decision_needed"] = bool(res["t_bar_ci"] and res["t_bar_ci"][1] > T_BAR_PLANNED)
    pooled = (res["groups"].get("pooled") or {}).get("series") or {}
    res["settled_epoch_pooled"] = {s: pooled[s]["settled_epoch_b"] for s in pooled}
    res["settled_epoch_max_over_groups"] = {s: _worst([g["series"][s]["settled_epoch_b"] for g in res["groups"].values()]) for s in SERIES} if res["groups"] else {}
    return res


def _worst(v: Sequence[Optional[int]]) -> Optional[int]:
    return None if any(x is None for x in v) else max(v)


def run(run_dir: str, requests_path: str, out: str, k_rounds: Optional[int] = None, reps: int = 400, seed: int = 20261029, mock: bool = False) -> Dict[str, Any]:
    grid = Grid(iter_jsonl(os.path.join(run_dir, "epochs.jsonl"), '"rec":"epoch"'))
    acc = accumulate(iter_jsonl(requests_path), iter_jsonl(os.path.join(run_dir, "cloud-gw.jsonl"), '"rec":"request"'), grid)
    blocks = switch_blocks(run_dir, grid, acc, k_rounds)
    res = analyze(blocks, reps, seed)
    res["rounds"] = k_rounds
    res["run_dir"] = run_dir
    if mock:
        res["_mock"] = True
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "SETTLING.json"), res)
    write_json(os.path.join(out, "settling-blocks.json"), [{**{k: v for k, v in b.items() if k != "curves"}, "curves": {s: [None if x is None else round(x, 6) for x in c] for s, c in b["curves"].items()}} for b in blocks])
    write_text(os.path.join(out, "SETTLING.md"), render_md(res))
    return res


def _f(x: Any, d: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def _k(x: Optional[int]) -> str:
    return "not within 30" if x is None else str(x)


def render_md(res: Dict[str, Any]) -> str:
    L: List[str] = []
    mock = " (MOCK: rounds so far, not the final data)" if res.get("_mock") else ""
    L.append(f"# Settling after a switch{mock}")
    L.append("")
    L.append(f"Written {res.get('written')}. Rounds {res.get('rounds')}. {res['n_switch_blocks']} node-blocks begin with a switch of the arm ({res['n_profile_switch']} with a switch of the engine profile, "
             f"{res['n_switch_blocks'] - res['n_profile_switch']} with a change of the admission order only). Deviation of each epoch from the plateau (epochs 30 to 49) of the same block; mean over "
             f"blocks, standard error clustered by round (the four node-blocks of a round share the cloud and the clock).")
    L.append("")
    pj = (res.get("groups", {}).get("pooled") or {}).get("joint") or {}
    ci = pj.get("ci") or {}
    sel = ", ".join(LABEL[s] for s in (pj.get("selected") or [])) or "none"
    head = f"**Fitted t_bar (pooled, one decay time over the series with a detected transient: {sel}):** "
    if res.get("t_bar") is None:
        head += "no series shows a detected transient; t_bar is not identified, and the deviations are within the noise from epoch 0."
    else:
        head += f"{_f(res['t_bar'], 2)} epochs"
        if res.get("t_bar_ci"):
            head += (f", 95 percent interval {_f(res['t_bar_ci'][0], 2)} to {_f(res['t_bar_ci'][1], 2)} (bootstrap over rounds, {res['reps']} replicates, series selected again in each; a transient is detected in "
                     f"{_f(ci['identified_fraction'] * 100, 1)} percent of them; counting the others at the grid bound of {_f(T_MAX, 0)} epochs the upper limit is {_f(res.get('t_bar_hi_conservative'), 2)})")
        head += f"; planned value {_f(res['t_bar_planned'], 0)}. " + ("Upper limit above the planned value: Decision needed." if res.get("decision_needed") else "Upper limit not above the planned value.")
    L.append(head)
    L.append("")
    L.append(f"**Epochs to settle (first epoch after which every deviation of epochs 0 to 29 stays within {_f(res['settle_threshold_se'], 2)} standard errors, the threshold of criterion (b)):**")
    L.append("")
    L.append("| group | blocks | " + " | ".join(LABEL[s] for s in SERIES) + " | joint t_bar (95 percent) |")
    L.append("|---|---|" + "---|" * (len(SERIES) + 1))
    for g, r in res["groups"].items():
        c = r["joint"].get("ci") or {}
        tb = "no transient detected" if r["joint"].get("t_bar") is None else f"{_f(r['joint']['t_bar'], 2)}" + (f" ({_f(c['lo'], 2)} to {_f(c['hi'], 2)})" if c.get("hi") is not None else "")
        L.append(f"| {g.replace('|', chr(92) + '|')} | {r['n_blocks']} | " + " | ".join(_k(r["series"][s]["settled_epoch_b"]) for s in SERIES) + f" | {tb} |")
    L.append("")
    L.append(f"With three standard errors instead (the threshold of criterion (a)), pooled: " + ", ".join(f"{LABEL[s]} {_k(((res['groups'].get('pooled') or {}).get('series') or {}).get(s, {}).get('settled_epoch_t3'))}" for s in SERIES) + ".")
    L.append("")
    L.append("**Pooled mean deviation by epoch (t = mean over standard error):**")
    L.append("")
    pooled = res["groups"].get("pooled")
    if pooled:
        L.append("| epoch | " + " | ".join(f"{LABEL[s]} mean (t)" for s in SERIES) + " |")
        L.append("|---|" + "---|" * len(SERIES))
        for k in range(0, 9):
            L.append(f"| {k} | " + " | ".join(f"{_f(pooled['series'][s]['mean'][k], 4)} ({_f(pooled['series'][s]['t'][k], 1)})" for s in SERIES) + " |")
        L.append("| largest abs t, epochs 9 to 29 | " + " | ".join(_f(_fmax(np.abs(np.array([x if x is not None else np.nan for x in pooled["series"][s]["t"][9:SHOW]]))), 1) for s in SERIES) + " |")
        L.append("")
        L.append("**Per series decay fit (pooled; amplitude, t_bar with the 95 percent interval over the replicates that detect a transient):** " + "; ".join(
            f"{LABEL[s]} {_f(pooled['series'][s]['fit']['amplitude'][0], 4)}, " + (f"{_f(pooled['series'][s]['fit']['t_bar'], 2)}" + (f" ({_f(pooled['series'][s]['fit']['ci']['lo'], 2)} to {_f(pooled['series'][s]['fit']['ci']['hi'], 2)})"
                                                                                                                                    if (pooled['series'][s]['fit'].get('ci') or {}).get('hi') is not None else "")
                                                                                       if pooled["series"][s]["fit"]["identified"] else "transient not detected") for s in SERIES) + ".")
        L.append("")
        bt = res.get("band_planned") or {}
        L.append(f"**Carryover left in the measurement window (mean deviation of epochs 25 to 29 times 5/25, pooled; mean, standard error):** " + "; ".join(
            f"{LABEL[s]} {_f(pooled['series'][s]['window_bias']['mean'], 4)}, {_f(pooled['series'][s]['window_bias']['se'], 4)}" for s in SERIES)
            + f". The planned carryover band for an outcome in [0, 1] is {_f(bt.get('t_bar_5'), 4)} at t_bar = 5 and {_f(bt.get('t_bar_8'), 4)} at t_bar = 8 (worst case oscillation one).")
    L.append("")
    L.append("Reading: t_bar is the e-folding time of the exponential carryover band (rho = exp(-1 / t_bar)); a value below the planned 5 epochs supports the planning value, and the "
             "t_bar = 8 robustness reported at the looks is conservative relative to it. A series without a detected transient has no identified t_bar: its slow component, if any, is below the noise. "
             "Reported, not deciding; the look estimator and the rule are unchanged.")
    return "\n".join(L) + "\n"


def summary_line(res: Optional[Dict[str, Any]]) -> str:
    if not res or res.get("error"):
        return f"Settling after a switch: not computed ({(res or {}).get('error', 'no result in this run')})."
    ci = res.get("t_bar_ci")
    se = res.get("settled_epoch_pooled") or {}
    worst = res.get("settled_epoch_max_over_groups") or {}
    tb = "no transient detected" if res.get("t_bar") is None else f"fitted t_bar {_f(res.get('t_bar'), 2)} epochs" + (f" (95 percent {_f(ci[0], 2)} to {_f(ci[1], 2)})" if ci else "")
    wv = [x for x in worst.values() if x is not None]
    return (f"Settling after a switch ({res['n_switch_blocks']} node-blocks): {tb} against the planned {_f(res['t_bar_planned'], 0)}; epochs to settle, pooled "
            + ", ".join(f"{LABEL[s]} {_k(se.get(s))}" for s in SERIES) + f"; worst direction {max(wv) if wv else 'n/a'}" + (" (not within 30 for some series)" if len(wv) < len(worst) else "")
            + ("; upper limit above 5 (Decision needed)." if res.get("decision_needed") else "."))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Settling after a switch, measured on the campaign data.', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--requests", required=True, help="joined request table (analysis directory of a look: requests.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--reps", type=int, default=400)
    a = ap.parse_args(argv)
    res = run(a.run_dir, a.requests, a.out, a.rounds, a.reps)
    print(summary_line(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
