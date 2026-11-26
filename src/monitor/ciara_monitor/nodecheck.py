from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ciara.constants import BUDGETS, OUTCOMES, finite

from .sufficiency import f_pvalue, holm, ols_cluster, t_pvalue

BINDING = ("Y_C", "Y_L_f1")
MIN_PER_CELL = 1


def window_rows(rows: Sequence[Dict[str, Any]], members: Sequence[str], k: Optional[int]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("window", "measure") == "measure" and r.get("block_status") == "complete" and r.get("arm_id") in members and (k is None or int(r["block_id"]) < k)]


def _rss(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    return {"rss": float(e @ e), "rank": int(np.linalg.matrix_rank(X))}


def _dummies(labels: Sequence[str]) -> np.ndarray:
    levels = sorted(set(labels))
    return np.array([[1.0 if x == lv else 0.0 for lv in levels] for x in labels])


def nested_f(rows: Sequence[Dict[str, Any]], outcome: str) -> Dict[str, Any]:
    rr = [r for r in rows if finite(r.get(outcome))]
    out: Dict[str, Any] = {"n": len(rr), "node": {"F": None, "df": None, "p": None}, "interaction": {"F": None, "df": None, "p": None}}
    if len(rr) < 8:
        return out
    y = np.array([float(r[outcome]) for r in rr])
    A = _dummies([r["arm_id"] for r in rr])
    N = _dummies([r["node"] for r in rr])
    cells = _dummies([f"{r['arm_id']}@{r['node']}" for r in rr])
    m0, m1, m2 = _rss(A, y), _rss(np.hstack([A, N]), y), _rss(cells, y)
    n = len(y)
    dfw = n - m2["rank"]
    for key, small, big in (("node", m0, m1), ("interaction", m1, m2)):
        d1 = big["rank"] - small["rank"]
        d2 = n - big["rank"]
        if d1 <= 0 or d2 <= 0 or big["rss"] <= 1e-18:
            continue
        F = ((small["rss"] - big["rss"]) / d1) / (big["rss"] / d2)
        out[key] = {"F": F, "df": [d1, d2], "p": f_pvalue(F, d1, d2)}
    out["within_cell_df"] = int(dfw)
    return out


def node_means(rows: Sequence[Dict[str, Any]], members: Sequence[str], outcome: str) -> Dict[str, Dict[str, Any]]:
    acc: Dict[str, Dict[str, List[float]]] = {a: {} for a in members}
    for r in rows:
        if finite(r.get(outcome)):
            acc[r["arm_id"]].setdefault(r["node"], []).append(float(r[outcome]))
    return {a: {n: {"mean": sum(v) / len(v), "n": len(v)} for n, v in sorted(d.items())} for a, d in acc.items()}


def throttle_by_node(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    acc: Dict[str, List[float]] = {}
    for r in rows:
        if finite(r.get("gpu_throttled_frac")):
            acc.setdefault(r["node"], []).append(float(r["gpu_throttled_frac"]))
    return {n: {"mean_throttled_share": sum(v) / len(v), "blocks_with_throttle": int(sum(1 for x in v if x > 0)), "n": len(v)} for n, v in sorted(acc.items())}


def throttle_slope(rows: Sequence[Dict[str, Any]], outcome: str) -> Dict[str, Any]:
    rr = [r for r in rows if finite(r.get(outcome)) and finite(r.get("gpu_throttled_frac"))]
    out: Dict[str, Any] = {"slope": None, "se": None, "t": None, "p": None, "n": len(rr)}
    if len(rr) < 12:
        return out
    arm_mean: Dict[str, float] = {}
    for a in {r["arm_id"] for r in rr}:
        v = [float(r[outcome]) for r in rr if r["arm_id"] == a]
        arm_mean[a] = sum(v) / len(v)
    y = np.array([float(r[outcome]) - arm_mean[r["arm_id"]] for r in rr])
    N = _dummies([r["node"] for r in rr])
    x = np.array([float(r["gpu_throttled_frac"]) for r in rr])
    X = np.hstack([N, x[:, None]])
    fit = ols_cluster(X, y, [int(r["block_id"]) for r in rr])
    if fit["cov"] is None:
        return out
    b, se = float(fit["beta"][-1]), math.sqrt(max(float(fit["cov"][-1, -1]), 0.0))
    if se > 0:
        out.update({"slope": b, "se": se, "t": b / se, "p": t_pvalue(b / se, fit["G"] - 1)})
    return out


def _center(rows: Sequence[Dict[str, Any]], arm: str, outcome: str, drop: Optional[str] = None) -> Optional[Dict[str, float]]:
    v = [float(r[outcome]) for r in rows if r["arm_id"] == arm and finite(r.get(outcome)) and r["node"] != drop]
    if not v:
        return None
    m = sum(v) / len(v)
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1) if len(v) > 1 else float("nan")
    return {"mean": m, "var": var, "n": len(v)}


def leave_one_node_out(rows: Sequence[Dict[str, Any]], members: Sequence[str], a0: str, outcome: str) -> Dict[str, Any]:
    nodes = sorted({r["node"] for r in rows})
    full = {a: _center(rows, a, outcome) for a in members}
    out: Dict[str, Any] = {"budget": BUDGETS.get(outcome), "centers": {}, "contrasts": {}}
    for a in members:
        if full[a] is None:
            continue
        ch = {}
        for nd in nodes:
            c = _center(rows, a, outcome, nd)
            if c is not None:
                ch[nd] = c["mean"]
        out["centers"][a] = {"full": full[a]["mean"], "without": ch, "max_abs_change": max((abs(v - full[a]["mean"]) for v in ch.values()), default=None)}
    for a in members:
        if a == a0 or full[a] is None or full.get(a0) is None:
            continue
        d_full = full[a]["mean"] - full[a0]["mean"]
        se = math.sqrt(full[a]["var"] / full[a]["n"] + full[a0]["var"] / full[a0]["n"]) if full[a]["n"] > 1 and full[a0]["n"] > 1 else float("nan")
        ch = {}
        for nd in nodes:
            ca, c0 = _center(rows, a, outcome, nd), _center(rows, a0, outcome, nd)
            if ca is not None and c0 is not None:
                ch[nd] = (ca["mean"] - c0["mean"]) - d_full
        worst = max(ch.items(), key=lambda kv: abs(kv[1]), default=(None, None))
        out["contrasts"][a] = {"full": d_full, "se": se if finite(se) else None, "change_without": ch, "worst_node": worst[0], "max_abs_change": abs(worst[1]) if worst[1] is not None else None,
                               "max_change_in_se": (abs(worst[1]) / se) if (worst[1] is not None and finite(se) and se > 0) else None}
    return out


def node_check(rows: Sequence[Dict[str, Any]], members: Sequence[str], a0: str, k: Optional[int] = None) -> Dict[str, Any]:
    wr = window_rows(rows, members, k)
    res: Dict[str, Any] = {"rounds": k, "node_blocks": len(wr), "members": list(members), "a0": a0, "throttle_by_node": throttle_by_node(wr), "outcomes": {}}
    pn: Dict[str, Optional[float]] = {}
    pi: Dict[str, Optional[float]] = {}
    for o in OUTCOMES:
        t = nested_f(wr, o)
        res["outcomes"][o] = {"tests": t, "node_means": node_means(wr, members, o), "throttle_slope": throttle_slope(wr, o), "leave_one_node_out": leave_one_node_out(wr, members, a0, o)}
        pn[o], pi[o] = t["node"]["p"], t["interaction"]["p"]
    hn, hi = holm(pn), holm(pi)
    for o in OUTCOMES:
        res["outcomes"][o]["tests"]["node"]["p_holm"] = hn[o]
        res["outcomes"][o]["tests"]["interaction"]["p_holm"] = hi[o]
    res["flags"] = {"node_effect": [o for o in OUTCOMES if hn[o] is not None and hn[o] < 0.05], "interaction": [o for o in OUTCOMES if hi[o] is not None and hi[o] < 0.05],
                    "throttle_slope": [o for o in OUTCOMES if (res["outcomes"][o]["throttle_slope"].get("p") or 1.0) < 0.05 / len(OUTCOMES)],
                    "contrast_moves_over_one_se": {o: [a for a, c in res["outcomes"][o]["leave_one_node_out"]["contrasts"].items() if (c.get("max_change_in_se") or 0.0) > 1.0] for o in OUTCOMES
                                                   if any((c.get("max_change_in_se") or 0.0) > 1.0 for c in res["outcomes"][o]["leave_one_node_out"]["contrasts"].values())},
                    "binding_rows_move": {o: [a for a, c in res["outcomes"][o]["leave_one_node_out"]["contrasts"].items() if (c.get("max_change_in_se") or 0.0) > 1.0] for o in BINDING}}
    res["n_tests"] = 2 * len(OUTCOMES)
    return res


def _f(x: Any, d: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def render_md(res: Dict[str, Any]) -> str:
    L = [f"## Node check, assumption A8 ({res['rounds']} rounds, {res['node_blocks']} node-blocks)\n",
         "F tests of the node effect (arm + node against arm) and of the node-by-arm interaction (the planned A8 test); Holm over the eight outcomes within each test. Reported, not used to decide.\n",
         "| outcome | node F (df) | p | Holm p | interaction F (df) | p | Holm p | throttle slope (t) |\n|---|---|---|---|---|---|---|---|"]
    for o in OUTCOMES:
        v = res["outcomes"][o]
        t, ts = v["tests"], v["throttle_slope"]
        df = lambda d: "n/a" if not d else f"{d[0]}, {d[1]}"
        L.append(f"| {o} | {_f(t['node']['F'], 2)} ({df(t['node']['df'])}) | {_f(t['node']['p'], 3)} | {_f(t['node'].get('p_holm'), 3)} | {_f(t['interaction']['F'], 2)} ({df(t['interaction']['df'])}) | "
                 f"{_f(t['interaction']['p'], 3)} | {_f(t['interaction'].get('p_holm'), 3)} | {_f(ts['slope'], 4)} ({_f(ts['t'], 2)}) |")
    fl = res["flags"]
    L.append(f"\nFlags: node effect {fl['node_effect'] or 'none'}; interaction {fl['interaction'] or 'none'}; throttle slope {fl['throttle_slope'] or 'none'}; "
             f"binding rows whose arm contrast moves by more than one standard error when a node is left out: { {o: v for o, v in fl['binding_rows_move'].items() if v} or 'none'}.\n")
    L.append("### Throttled epochs by node (share of the window epochs with a throttle reason, mean over node-blocks)\n\n| node | mean share | node-blocks with a throttled epoch | node-blocks |\n|---|---|---|---|")
    for n, d in res["throttle_by_node"].items():
        L.append(f"| {n} | {_f(d['mean_throttled_share'], 3)} | {d['blocks_with_throttle']} | {d['n']} |")
    nodes = sorted(res["throttle_by_node"])
    for o in list(BINDING) + [x for x in OUTCOMES if x not in BINDING]:
        v = res["outcomes"][o]
        L.append(f"\n### {o}{' (binding row of the certified program)' if o in BINDING else ''}\n")
        L.append("| arm | " + " | ".join(nodes) + " | center | center without node (min to max) |\n|---|" + "---|" * (len(nodes) + 2))
        for a in res["members"]:
            nm = v["node_means"].get(a, {})
            c = v["leave_one_node_out"]["centers"].get(a)
            if c is None:
                continue
            w = list(c["without"].values())
            L.append(f"| {a.replace('|', chr(92) + '|')} | " + " | ".join(_f(nm[n]["mean"]) + f" ({nm[n]['n']})" if n in nm else "n/a" for n in nodes) + f" | {_f(c['full'])} | {_f(min(w))} to {_f(max(w))} |")
        L.append("\n| contrast to a0 | full | s.e. | change without node " + " / ".join(nodes) + " | largest change in s.e. |\n|---|---|---|---|---|")
        for a, c in v["leave_one_node_out"]["contrasts"].items():
            L.append(f"| {a.replace('|', chr(92) + '|')} | {_f(c['full'])} | {_f(c['se'])} | " + " / ".join(_f(c['change_without'].get(n)) for n in nodes) + f" | {_f(c['max_change_in_se'], 2)} |")
    return "\n".join(L) + "\n"
