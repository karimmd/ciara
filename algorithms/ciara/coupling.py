from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import screening as S
from .constants import OUTCOMES, finite

QUEUE_KEYS = ("backlog", "wait_s")


def fe_slope(x: Sequence[float], y: Sequence[float], groups: Sequence[Any], clusters: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    gl = sorted(set(groups), key=str)
    gi = np.array([gl.index(g) for g in groups]) if n else np.array([], int)
    out: Dict[str, Any] = {"n": int(n), "n_groups": len(gl), "slope": None, "se": None, "se_cluster": None, "t": None, "n_clusters": None}
    if n < len(gl) + 2:
        return out
    xt, yt = x.copy(), y.copy()
    for g in range(len(gl)):
        m = gi == g
        xt[m] -= x[m].mean()
        yt[m] -= y[m].mean()
    sxx = float(np.sum(xt ** 2))
    if sxx <= 1e-12:
        return out
    beta = float(np.sum(xt * yt) / sxx)
    e = yt - beta * xt
    dof = n - len(gl) - 1
    se = math.sqrt(float(np.sum(e ** 2)) / dof / sxx) if dof > 0 else None
    sec = None
    if clusters is not None:
        cl = sorted(set(clusters), key=str)
        out["n_clusters"] = len(cl)
        if len(cl) > 1 and dof > 0:
            ci = np.array([cl.index(c) for c in clusters])
            meat = sum(float(np.sum(xt[ci == c] * e[ci == c])) ** 2 for c in range(len(cl)))
            sec = math.sqrt(meat * len(cl) / (len(cl) - 1) * (n - 1) / dof) / sxx
    out.update({"slope": beta, "se": se, "se_cluster": sec})
    ref = sec if sec else se
    out["t"] = (beta / ref) if (ref and ref > 0) else None
    return out


def screening_slopes(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rs = S.usable_rows(rows)
    ids = sorted({r["arm_id"] for r in rs})
    est = S.arm_estimates(rs, ids)
    cl = {a: (est[a]["outcomes"]["Y_C"]["mean"] or 0.0) for a in ids}
    e = S.sibling_shares(rs, cl)
    out: Dict[str, Any] = {"n_rows": len(rs), "n_arms": len(ids), "rounds": len({int(r["block_id"]) for r in rs}), "outcomes": {}}
    for k in OUTCOMES:
        pts = [(e[(r["node"], int(r["block_id"]))], float(r[k]), r["arm_id"], int(r["block_id"])) for r in rs if finite(r.get(k)) and (r["node"], int(r["block_id"])) in e]
        out["outcomes"][k] = fe_slope([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts])
    return out


def round_queue(cloud_gw_path: str, rows: Sequence[Dict[str, Any]], k_rounds: Optional[int] = None) -> Dict[int, Dict[str, float]]:
    win: Dict[int, List[float]] = {}
    for r in rows:
        if r.get("block_status") != "complete" or r.get("t_lo_mono_ns") is None or r.get("t_hi_mono_ns") is None:
            continue
        b = int(r["block_id"])
        if k_rounds is not None and b >= k_rounds:
            continue
        w = win.setdefault(b, [float(r["t_lo_mono_ns"]), float(r["t_hi_mono_ns"])])
        w[0], w[1] = min(w[0], float(r["t_lo_mono_ns"])), max(w[1], float(r["t_hi_mono_ns"]))
    spans = sorted((lo, hi, b) for b, (lo, hi) in win.items())
    starts = [s[0] for s in spans]
    acc: Dict[int, List[float]] = {b: [0.0, 0.0, 0.0] for b in win}
    import bisect
    with open(cloud_gw_path) as fh:
        for line in fh:
            if '"rec":"request"' not in line and '"rec": "request"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t, ta = d.get("t_arrive"), d.get("t_admit")
            if t is None or ta is None:
                continue
            i = bisect.bisect_right(starts, t) - 1
            if i < 0 or t > spans[i][1]:
                continue
            a = acc[spans[i][2]]
            a[0] += float(d.get("queue_len_at_arrival") or 0) + float(d.get("in_flight_at_arrival") or 0)
            a[1] += (float(ta) - float(t)) / 1e9
            a[2] += 1.0
    return {b: {"backlog": v[0] / v[2], "wait_s": v[1] / v[2], "n": int(v[2])} for b, v in acc.items() if v[2] > 0}


def cloud_service_by_arm(cloud_gw_path: str, members: Sequence[str]) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, List[float]] = {m: [0.0, 0.0] for m in members}
    with open(cloud_gw_path) as fh:
        for line in fh:
            if '"rec":"request"' not in line and '"rec": "request"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            a = d.get("arm_id")
            if a in acc and d.get("t_admit") is not None and d.get("t_done") is not None:
                acc[a][0] += (float(d["t_done"]) - float(d["t_admit"])) / 1e9
                acc[a][1] += 1.0
    return {m: {"service_s": v[0] / v[1], "n": int(v[1])} for m, v in acc.items() if v[1] > 0}


def work_check(dec: Dict[str, Any], service: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    tab, pi = dec.get("arms_table") or {}, dec.get("pi")
    if not pi or any(a not in service for a in tab):
        return {"available": False, "reason": "no policy or no service times"}
    lhs = sum(pi[a] * tab[a]["outcomes"]["Y_C"]["upper"] * service[a]["service_s"] for a in tab)
    rhs = sum(tab[a]["propensity"] * tab[a]["outcomes"]["Y_C"]["centre"] * service[a]["service_s"] for a in tab)
    return {"available": True, "policy_work_upper": lhs, "design_work": rhs, "margin": rhs - lhs, "holds": lhs <= rhs + 1e-12, "service_s": {a: service[a]["service_s"] for a in tab}}


def look_slopes(rows: Sequence[Dict[str, Any]], queue: Dict[int, Dict[str, float]], members: Sequence[str]) -> Dict[str, Any]:
    rs = [r for r in S.usable_rows(rows) if r["arm_id"] in members and int(r["block_id"]) in queue]
    out: Dict[str, Any] = {"rounds_with_queue": len({int(r["block_id"]) for r in rs}), "n_rows": len(rs),
                           "queue_mean": {m: float(np.mean([queue[b][m] for b in queue])) if queue else None for m in QUEUE_KEYS}, "by_measure": {}}
    for qk in QUEUE_KEYS:
        res: Dict[str, Any] = {}
        for k in OUTCOMES:
            pts = [(queue[int(r["block_id"])][qk], float(r[k]), r["arm_id"], int(r["block_id"])) for r in rs if finite(r.get(k))]
            ent: Dict[str, Any] = {"pooled": fe_slope([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts]), "members": {}}
            for m in members:
                pm = [p for p in pts if p[2] == m]
                ent["members"][m] = fe_slope([p[0] for p in pm], [p[1] for p in pm], [m] * len(pm), [p[3] for p in pm])
            res[k] = ent
        out["by_measure"][qk] = res
    return out


def _f(x: Optional[float], d: int = 4) -> str:
    return "n/a" if x is None or not finite(x) else f"{x:.{d}f}"


def render_slopes_md(title: str, res: Dict[str, Any], measure: Optional[str] = None) -> str:
    L = [f"**{title}**\n", "| outcome | slope | se (classical) | se (clustered by round) | t | n |", "|---|---|---|---|---|---|"]
    items = res["outcomes"] if measure is None else {k: v["pooled"] for k, v in res["by_measure"][measure].items()}
    for k in OUTCOMES:
        s = items[k]
        L.append(f"| {k} | {_f(s['slope'], 5)} | {_f(s['se'], 5)} | {_f(s['se_cluster'], 5)} | {_f(s['t'], 2)} | {s['n']} |")
    return "\n".join(L) + "\n"
