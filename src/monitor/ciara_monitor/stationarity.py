from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from ciara.constants import OUTCOMES, finite


def round_means(rows: Sequence[Dict[str, Any]], arm: str, outcome: str, k: int) -> List[Optional[float]]:
    by: Dict[int, List[float]] = {}
    for r in rows:
        if r.get("window", "measure") == "measure" and r.get("block_status") == "complete" and r.get("arm_id") == arm and int(r["block_id"]) < k and finite(r.get(outcome)):
            by.setdefault(int(r["block_id"]), []).append(float(r[outcome]))
    return [(sum(by[b]) / len(by[b]) if b in by else None) for b in range(k)]


def slope_nw(y: Sequence[Optional[float]], lag: int = 5) -> Dict[str, Any]:
    pts = [(i, v) for i, v in enumerate(y) if v is not None]
    n = len(pts)
    if n < 8:
        return {"n": n, "slope_per_100": None, "se_per_100": None, "t": None}
    x = [p[0] for p in pts]
    v = [p[1] for p in pts]
    mx, my = sum(x) / n, sum(v) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx <= 0:
        return {"n": n, "slope_per_100": None, "se_per_100": None, "t": None}
    b = sum((a - mx) * (c - my) for a, c in zip(x, v)) / sxx
    e = [c - my - b * (a - mx) for a, c in zip(x, v)]
    u = [(a - mx) * ei for a, ei in zip(x, e)]
    s = sum(t * t for t in u)
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * sum(u[i] * u[i - l] for i in range(l, n))
    se = math.sqrt(max(s, 0.0)) / sxx
    return {"n": n, "slope_per_100": 100.0 * b, "se_per_100": 100.0 * se, "t": (b / se) if se > 0 else None}


def half_difference(y: Sequence[Optional[float]]) -> Dict[str, Any]:
    v = [t for t in y if t is not None]
    n = len(v)
    if n < 8:
        return {"n": n, "diff": None, "se": None, "t": None}
    a, b = v[: n // 2], v[n // 2:]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((t - ma) ** 2 for t in a) / max(len(a) - 1, 1)
    vb = sum((t - mb) ** 2 for t in b) / max(len(b) - 1, 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    return {"n": n, "diff": mb - ma, "se": se, "t": ((mb - ma) / se) if se > 0 else None}


def a7(rows: Sequence[Dict[str, Any]], members: Sequence[str], k: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rounds": k, "arms": {}, "findings": []}
    for a in members:
        out["arms"][a] = {}
        for o in OUTCOMES:
            y = round_means(rows, a, o, k)
            s, h = slope_nw(y), half_difference(y)
            out["arms"][a][o] = {"slope": s, "halves": h}
            for name, t in (("slope", s.get("t")), ("half difference", h.get("t"))):
                if t is not None and abs(t) > 3.0:
                    out["findings"].append({"arm": a, "outcome": o, "test": name, "t": t})
    out["n_tests"] = 2 * len(members) * len(OUTCOMES)
    out["expected_false_findings_at_3_se"] = out["n_tests"] * 0.0027
    return out


def render_md(res: Dict[str, Any]) -> str:
    L = [f"## A7 stationarity of the arm levels ({res['rounds']} rounds)\n",
         f"Tests: {res['n_tests']} (slope and half difference per member and outcome); findings beyond three standard errors: {len(res['findings'])} (about {res['expected_false_findings_at_3_se']:.2f} expected by chance).\n"]
    for f in res["findings"]:
        L.append(f"* {f['arm']}, {f['outcome']}, {f['test']}: t = {f['t']:.2f}")
    return "\n".join(L) + "\n"
