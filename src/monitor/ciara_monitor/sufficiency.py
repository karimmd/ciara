from __future__ import annotations

import bisect
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ciara.constants import OUTCOMES, finite

HEAD_S = 120.0
MIN_REQUESTS = 5
BURN_IN_EPOCHS = 25
EPOCH_S = 30.0
MIN_CLUSTERS = 12
ALPHA = 0.05


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_pvalue(t: float, df: float) -> float:
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def f_pvalue(f: float, d1: float, d2: float) -> float:
    if f <= 0:
        return 1.0
    return betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def holm(pvals: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    items = sorted(((p, k) for k, p in pvals.items() if p is not None))
    m = len(items)
    out: Dict[str, Optional[float]] = {k: None for k in pvals}
    run = 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def ols_cluster(X: np.ndarray, y: np.ndarray, clusters: Sequence[Any]) -> Dict[str, Any]:
    n, p = X.shape
    cl = sorted(set(clusters), key=str)
    G = len(cl)
    out: Dict[str, Any] = {"n": int(n), "p": int(p), "G": G, "beta": None, "cov": None}
    if G < 3 or n <= p or np.linalg.matrix_rank(X) < p:
        return out
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    ci = np.array([cl.index(c) for c in clusters])
    meat = np.zeros((p, p))
    for g in range(G):
        s = X[ci == g].T @ e[ci == g]
        meat += np.outer(s, s)
    cov = xtx_inv @ meat @ xtx_inv * (G / (G - 1.0)) * ((n - 1.0) / (n - p))
    out.update({"beta": beta, "cov": cov})
    return out


def coefficient_tests(X: np.ndarray, y: np.ndarray, clusters: Sequence[Any], q_prev: int) -> Dict[str, Any]:
    fit = ols_cluster(X, y, clusters)
    res: Dict[str, Any] = {"n": fit["n"], "G": fit["G"], "composition": {"p": None, "stat": None}, "backlog": {"coef": None, "se": None, "t": None, "p": None}}
    if fit["cov"] is None:
        return res
    b, V, G = fit["beta"], fit["cov"], fit["G"]
    if q_prev > 0:
        bi, Vi = b[1:1 + q_prev], V[1:1 + q_prev, 1:1 + q_prev]
        try:
            w = float(bi @ np.linalg.solve(Vi, bi))
            res["composition"] = {"p": f_pvalue(w / q_prev, q_prev, G - 1.0), "stat": w / q_prev, "coefs": [float(x) for x in bi]}
        except np.linalg.LinAlgError:
            pass
    se = math.sqrt(V[-1, -1]) if V[-1, -1] > 0 else None
    if se:
        t = float(b[-1]) / se
        res["backlog"] = {"coef": float(b[-1]), "se": se, "t": t, "p": t_pvalue(t, G - 1.0)}
    return res


def backlog_at_block_start(cloud_gw_path: str, rows: Sequence[Dict[str, Any]], k_rounds: Optional[int] = None, head_s: float = HEAD_S, burn_in_epochs: int = BURN_IN_EPOCHS) -> Dict[int, float]:
    start: Dict[int, float] = {}
    for r in rows:
        if r.get("block_status") != "complete" or r.get("t_lo_mono_ns") is None or r.get("window", "measure") != "measure":
            continue
        b = int(r["block_id"])
        if k_rounds is not None and b >= k_rounds:
            continue
        t0 = float(r["t_lo_mono_ns"]) - burn_in_epochs * EPOCH_S * 1e9
        start[b] = min(start.get(b, t0), t0)
    spans = sorted((lo, lo + head_s * 1e9, b) for b, lo in start.items())
    starts = [s[0] for s in spans]
    acc: Dict[int, List[float]] = {b: [0.0, 0.0] for b in start}
    with open(cloud_gw_path) as fh:
        for line in fh:
            if '"rec":"request"' not in line and '"rec": "request"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = d.get("t_arrive")
            if t is None:
                continue
            i = bisect.bisect_right(starts, t) - 1
            if i < 0 or t >= spans[i][1]:
                continue
            a = acc[spans[i][2]]
            a[0] += float(d.get("queue_len_at_arrival") or 0) + float(d.get("in_flight_at_arrival") or 0)
            a[1] += 1.0
    return {b: v[0] / v[1] for b, v in acc.items() if v[1] >= MIN_REQUESTS}


def previous_arm(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int], str]:
    out: Dict[Tuple[str, int], str] = {}
    for r in rows:
        if r.get("arm_id") and r.get("window", "measure") == "measure":
            out[(r["node"], int(r["block_id"]))] = r["arm_id"]
    return out


def state_sufficiency(rows: Sequence[Dict[str, Any]], members: Sequence[str], backlog: Dict[int, float], k_rounds: Optional[int] = None, alpha: float = ALPHA,
                      outcomes: Sequence[str] = OUTCOMES, min_clusters: int = MIN_CLUSTERS) -> Dict[str, Any]:
    prev = previous_arm(rows)
    use = [r for r in rows if r.get("window", "measure") == "measure" and r.get("block_status") == "complete" and r.get("arm_id") in members
           and (k_rounds is None or int(r["block_id"]) < k_rounds)]
    ref, others = members[0], list(members[1:])
    res: Dict[str, Any] = {"alpha": alpha, "members": list(members), "reference_previous_arm": ref, "previous_arm_columns": others, "head_s": HEAD_S, "arms": {}, "rounds_with_backlog": len(backlog)}
    for a in members:
        pts = []
        for r in use:
            b = int(r["block_id"])
            if r["arm_id"] != a or b < 1 or b not in backlog:
                continue
            pa = prev.get((r["node"], b - 1))
            if pa is None or pa not in members:
                continue
            pts.append((r, b, pa))
        ent: Dict[str, Any] = {"n_rows": len(pts), "rounds": len({p[1] for p in pts}), "testable": False, "outcomes": {}}
        res["arms"][a] = ent
        if ent["rounds"] < min_clusters:
            continue
        ent["testable"] = True
        pc, pb = {}, {}
        for k in outcomes:
            sub = [p for p in pts if finite(p[0].get(k))]
            if len(sub) < min_clusters:
                ent["outcomes"][k] = {"n": len(sub), "composition": {"p": None}, "backlog": {"p": None}}
                continue
            cols = [o for o in others if 0 < sum(1 for p in sub if p[2] == o) < len(sub)]
            X = np.array([[1.0] + [1.0 if p[2] == o else 0.0 for o in cols] + [backlog[p[1]]] for p in sub])
            y = np.array([float(p[0][k]) for p in sub])
            t = coefficient_tests(X, y, [p[1] for p in sub], len(cols))
            t["columns"] = cols
            ent["outcomes"][k] = t
            pc[k], pb[k] = t["composition"]["p"], t["backlog"]["p"]
        hc, hb = holm(pc), holm(pb)
        for k, t in ent["outcomes"].items():
            t["composition"]["p_holm"], t["backlog"]["p_holm"] = hc.get(k), hb.get(k)
            ps = [x for x in (hc.get(k), hb.get(k)) if x is not None]
            t["sufficient"] = (all(x >= alpha for x in ps) if ps else None)
    verdict: Dict[str, Optional[bool]] = {}
    for k in outcomes:
        v = [res["arms"][a]["outcomes"][k]["sufficient"] for a in members if res["arms"][a]["testable"] and k in res["arms"][a]["outcomes"]
             and res["arms"][a]["outcomes"][k].get("sufficient") is not None]
        verdict[k] = (all(v) if v else None)
    res["sufficient"] = verdict
    res["not_sufficient"] = [k for k, v in verdict.items() if v is False]
    res["n_arms_tested"] = sum(1 for a in members if res["arms"][a]["testable"])
    return res


def check_run(run_dir: str, rows: Sequence[Dict[str, Any]], members: Sequence[str], k_rounds: Optional[int] = None, cloud_gw_path: Optional[str] = None) -> Dict[str, Any]:
    import os
    gw = cloud_gw_path or os.path.join(run_dir, "cloud-gw.jsonl")
    bl = backlog_at_block_start(gw, rows, k_rounds)
    return state_sufficiency(rows, members, bl, k_rounds)


def render_md(res: Dict[str, Any]) -> str:
    def f(x: Any) -> str:
        return "n/a" if x is None else f"{x:.3f}"
    L = [f"## State sufficiency of the arm levels ({res['rounds_with_backlog']} rounds with a block-start backlog)\n",
         f"Regression of the node-block window mean, within each arm, on the arm that the node played in the previous round (reference {res['reference_previous_arm']}) and on the cloud backlog at "
         f"block start, round-clustered standard errors; Holm over outcomes; alpha {res['alpha']}. Arms tested: {res['n_arms_tested']} of {len(res['members'])}.\n",
         "| outcome | sufficient | " + " | ".join(f"{a.split('|')[-2] + '|' + a.split('|')[-1] if '|' in a else a}: p composition, p backlog (Holm)" for a in res["members"]) + " |",
         "|---|---|" + "---|" * len(res["members"])]
    for k, v in res["sufficient"].items():
        cells = []
        for a in res["members"]:
            e = res["arms"][a]
            o = e["outcomes"].get(k) if e["testable"] else None
            cells.append("not testable" if o is None else f"{f(o['composition'].get('p_holm'))}, {f(o['backlog'].get('p_holm'))}")
        L.append(f"| {k} | {'n/a' if v is None else ('yes' if v else 'NO')} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"Not sufficient for: {', '.join(res['not_sufficient']) or 'none'}.")
    return "\n".join(L) + "\n"
