from __future__ import annotations

import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .constants import finite, mean

HOLD = 40
CYCLES = 6
SE_MULT = 3.0
SETTLE_SE_MULT = 4.08
SETTLE_WINDOW = (5, 10)
PLATEAU_WINDOW = (30, 40)
SETTLE_ALPHA = 0.05
T_BAR_MAX = 5
REL_TOL = 1e-3
SERIES = ("cloud_queue", "mean_latency_s", "tail_exceedance")


def hold_blocks(cycles: int = CYCLES) -> Dict[str, List[int]]:
    return {"H": [1 + 2 * c for c in range(cycles)], "L": [0] + [2 + 2 * c for c in range(cycles)]}


def context_schedule(cycles: int = CYCLES, hold: int = HOLD) -> Dict[str, Any]:
    segs = []
    n_blocks = 1 + 2 * cycles
    for b in range(n_blocks):
        segs.append({"start_epoch": b * hold, "end_epoch": (b + 1) * hold, "context_id": "H" if b in hold_blocks(cycles)["H"] else "L"})
    return {"default_context": "L", "segments": segs}


def pick_vectors(selection: Dict[str, Any]) -> Dict[str, Any]:
    S = selection["S"]
    sm = selection["screened_means"]

    def key_heavy(a: str):
        m = sm[a]
        return (m.get("Y_C") or 0.0, (m.get("Y_L_f2") or 0.0) + (m.get("Y_L_f1") or 0.0), -S.index(a))
    heavy = max(S, key=key_heavy)
    light = min(S, key=lambda a: (sm[a].get("Y_C") or 0.0, (sm[a].get("Y_L_f2") or 0.0) + (sm[a].get("Y_L_f1") or 0.0), S.index(a)))
    return {"heavy": heavy, "light": light, "heavy_cloud_share": sm[heavy].get("Y_C"), "light_cloud_share": sm[light].get("Y_C")}


def epoch_series(requests: Sequence[Dict[str, Any]], poll: Sequence[Dict[str, Any]], t0_mono_ns: int, epoch_ns: int, n_epochs: int) -> Dict[str, List[Optional[float]]]:
    lat: List[List[float]] = [[] for _ in range(n_epochs)]
    exc: List[List[float]] = [[] for _ in range(n_epochs)]
    for r in requests:
        e = r.get("epoch")
        if e is None or not (0 <= e < n_epochs):
            continue
        if r.get("ok") and finite(r.get("latency_s")):
            lat[e].append(float(r["latency_s"]))
        if r.get("exceed") is not None:
            exc[e].append(float(r["exceed"]))
    cq: List[List[float]] = [[] for _ in range(n_epochs)]
    for p in poll:
        e = (p["mono_ns"] - t0_mono_ns) // epoch_ns
        c = p.get("cloud_gw")
        if 0 <= e < n_epochs and c and c.get("queue") is not None:
            cq[int(e)].append(float(c["queue"]) + float(c.get("inflight") or 0))
    return {"cloud_queue": [mean(x) for x in cq], "mean_latency_s": [mean(x) for x in lat], "tail_exceedance": [mean(x) for x in exc]}


def _window(series: Sequence[Optional[float]], lo: int, hi: int) -> Optional[float]:
    vals = [series[i] for i in range(lo, hi) if i < len(series) and series[i] is not None]
    return (sum(vals) / len(vals)) if len(vals) >= max(3, (hi - lo) // 2) else None


def plateau_test(series: Sequence[Optional[float]], cycles: int = CYCLES, hold: int = HOLD) -> Dict[str, Any]:
    d: List[float] = []
    level: List[float] = []
    for b in hold_blocks(cycles)["H"]:
        s = b * hold
        a, c = _window(series, s + 20, s + 30), _window(series, s + 30, s + 40)
        if a is not None and c is not None:
            d.append(c - a)
            level.append(abs(a))
    if len(d) < 2:
        return {"pass": False, "reason": "fewer than two complete heavy holds", "d": d}
    md = sum(d) / len(d)
    sd = statistics.stdev(d)
    se = sd / math.sqrt(len(d))
    tol = REL_TOL * max(max(level), 1e-9)
    ok = abs(md) <= SE_MULT * se + tol
    t_stat = (md / se) if se > 0 else (0.0 if abs(md) <= tol else float("inf"))
    return {"pass": bool(ok), "d": d, "mean_d": md, "se": se, "two_se": SE_MULT * se, "t": t_stat, "tolerance": tol}


def tbar_after_switch(series: Sequence[Optional[float]], into: str, cycles: int = CYCLES, hold: int = HOLD) -> Dict[str, Any]:
    blocks = hold_blocks(cycles)[into] if into == "H" else [b + 1 for b in hold_blocks(cycles)["H"]]
    rows = []
    for b in blocks:
        seg = series[b * hold:(b + 1) * hold]
        if len(seg) == hold and sum(1 for x in seg if x is not None) >= hold - 4:
            fill = []
            last = None
            for x in seg:
                last = x if x is not None else last
                fill.append(last)
            first = next((x for x in fill if x is not None), None)
            rows.append([first if x is None else x for x in fill])
    if len(rows) < 2:
        return {"t_bar": None, "pass": False, "reason": "fewer than two complete holds", "holds": len(rows)}
    u = [sum(r[k] for r in rows) / len(rows) for k in range(hold)]
    mu = sum(u[hold - 10:]) / 10.0
    sd = statistics.stdev(u[hold - 20:])
    D = [abs(x - mu) for x in u]
    thr = max(0.1 * (D[0] + D[1]) / 2.0, 2.0 * sd, 1e-12)
    t = next((k for k in range(hold) if all(D[j] <= thr for j in range(k, hold))), None)
    return {"t_bar": t, "pass": t is not None and t <= T_BAR_MAX, "plateau": mu, "noise_sd": sd, "threshold": thr, "holds": len(rows), "u": [round(x, 4) for x in u[:12]]}


def _switch_blocks(into: str, cycles: int) -> List[int]:
    return hold_blocks(cycles)["H"] if into == "H" else [b + 1 for b in hold_blocks(cycles)["H"]]


def settle_test(series: Sequence[Optional[float]], into: str, cycles: int = CYCLES, hold: int = HOLD, mult: Optional[float] = None) -> Dict[str, Any]:
    mult = SETTLE_SE_MULT if mult is None else mult
    e: List[float] = []
    level: List[float] = []
    for b in _switch_blocks(into, cycles):
        s0 = b * hold
        a, c = _window(series, s0 + SETTLE_WINDOW[0], s0 + SETTLE_WINDOW[1]), _window(series, s0 + PLATEAU_WINDOW[0], s0 + PLATEAU_WINDOW[1])
        if a is not None and c is not None:
            e.append(a - c)
            level.append(abs(c))
    if len(e) < 2:
        return {"pass": False, "reason": "fewer than two complete holds", "e": e}
    me = sum(e) / len(e)
    se = statistics.stdev(e) / math.sqrt(len(e))
    tol = REL_TOL * max(max(level), 1e-9)
    t_stat = (me / se) if se > 0 else (0.0 if abs(me) <= tol else math.copysign(float("inf"), me))
    ok = abs(me) <= mult * se + tol
    return {"pass": bool(ok), "e": e, "mean_e": me, "se": se, "t": t_stat, "threshold_se": mult, "tolerance": tol, "holds": len(e)}


def settle_stats(series: Dict[str, Sequence[Optional[float]]], cycles: int = CYCLES, hold: int = HOLD) -> List[float]:
    out = []
    for k in SERIES:
        for w in ("H", "L"):
            r = settle_test(series[k], w, cycles, hold, mult=1e9)
            out.append(abs(r["t"]) if "t" in r else float("inf"))
    return out


def shuffled_within_holds(series: Dict[str, Sequence[Optional[float]]], rng, cycles: int = CYCLES, hold: int = HOLD) -> Dict[str, List[Optional[float]]]:
    out = {k: list(v) for k, v in series.items()}
    for b in sorted(set(_switch_blocks("H", cycles) + _switch_blocks("L", cycles))):
        lo, hi = b * hold, (b + 1) * hold
        if hi > len(next(iter(out.values()))):
            continue
        idx = list(range(lo, hi))
        perm = list(idx)
        rng.shuffle(perm)
        for v in out.values():
            seg = [v[i] for i in perm]
            v[lo:hi] = seg
    return out


def calibrate_settle(series: Dict[str, Sequence[Optional[float]]], n_shuffles: int = 2000, seed: int = 20261026, alpha: float = SETTLE_ALPHA,
                     cycles: int = CYCLES, hold: int = HOLD, base_mult: float = SE_MULT) -> Dict[str, Any]:
    import random
    rng = random.Random(seed)
    worst: List[float] = []
    cell_fail = [0] * 6
    for _ in range(n_shuffles):
        t = settle_stats(shuffled_within_holds(series, rng, cycles, hold), cycles, hold)
        worst.append(max(t))
        for i, x in enumerate(t):
            cell_fail[i] += x > base_mult
    worst.sort()
    n_ok = int(math.floor((1.0 - alpha) * n_shuffles))
    thr = worst[n_ok - 1]
    thr = math.ceil(thr * 100.0 - 1e-9) / 100.0
    rate_at = lambda m: sum(1 for x in worst if x > m) / n_shuffles
    rate_base = rate_at(base_mult)
    used = base_mult if rate_base <= alpha else max(base_mult, thr)
    return {"n_shuffles": n_shuffles, "seed": seed, "alpha": alpha, "base_mult": base_mult, "false_failure_joint_at_base": rate_base,
            "false_failure_per_cell_at_base": {f"{k}/{w}": cell_fail[i] / n_shuffles for i, (k, w) in enumerate((k, w) for k in SERIES for w in ("into_heavy", "out_of_heavy"))},
            "smallest_threshold": thr, "threshold_used": used, "false_failure_joint_at_used": rate_at(used), "raised": used > base_mult}


def verdict(series: Dict[str, Sequence[Optional[float]]], cycles: int = CYCLES, hold: int = HOLD, settle_mult: Optional[float] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"series": {}, "pass": True, "failed": [], "settle_threshold_se": SETTLE_SE_MULT if settle_mult is None else settle_mult}
    for k in SERIES:
        s = series[k]
        a = plateau_test(s, cycles, hold)
        sb_in = settle_test(s, "H", cycles, hold, settle_mult)
        sb_out = settle_test(s, "L", cycles, hold, settle_mult)
        b_in = tbar_after_switch(s, "H", cycles, hold)
        b_out = tbar_after_switch(s, "L", cycles, hold)
        out["series"][k] = {"plateau": a, "settle_into_heavy": sb_in, "settle_out_of_heavy": sb_out, "t_bar_into_heavy": b_in, "t_bar_out_of_heavy": b_out,
                            "pass": bool(a["pass"] and sb_in["pass"] and sb_out["pass"]), "pass_t_bar_reading": bool(a["pass"] and b_in["pass"] and b_out["pass"])}
        if not out["series"][k]["pass"]:
            out["pass"] = False
            out["failed"].append(k)
    t_all = [out["series"][k][w]["t_bar"] for k in SERIES for w in ("t_bar_into_heavy", "t_bar_out_of_heavy") if out["series"][k][w]["t_bar"] is not None]
    out["t_bar_joint_max"] = max(t_all) if t_all else None
    out["pass_t_bar_reading"] = all(out["series"][k]["pass_t_bar_reading"] for k in SERIES)
    return out


def reevaluate(result_json: str, n_shuffles: int = 10000, seed: int = 20261026) -> Dict[str, Any]:
    old = json.load(open(result_json))
    series = old["series_values"]
    cycles, hold = int(old.get("cycles", CYCLES)), int(old.get("hold", HOLD))
    cal = calibrate_settle(series, n_shuffles, seed, cycles=cycles, hold=hold)
    res = verdict(series, cycles, hold, settle_mult=cal["threshold_used"])
    for k in ("attempt", "heavy", "light", "heavy_cloud_share", "light_cloud_share", "series_values", "hold", "cycles", "run"):
        res[k] = old.get(k)
    res.update({"reevaluation": True, "original_result": os.path.basename(result_json), "original_pass": old.get("pass"), "original_failed": old.get("failed"),
                "settle_calibration": cal})
    return res
