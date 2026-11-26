from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import radius as R
from .constants import B_EPOCHS, BUDGETS, COUNT_KEY, EPS, K7, L_DEP, L_EPOCHS, N_NODES, OUTCOMES, T_BAR, finite, mean, variance
from .transient import TauTable

CLOUD_PER_CONFIGURATION, CLOUD_SET_LEVEL = "per_configuration", "set_level"
CLOUD_MODES = (CLOUD_PER_CONFIGURATION, CLOUD_SET_LEVEL)
TIE_Q = 0.005
MIN_BLOCKS = 3


def usable_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("window", "measure") == "measure" and r.get("block_status") == "complete" and r.get("arm_id")]


def arm_estimates(rows: Sequence[Dict[str, Any]], arm_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {a: {"n_blocks": 0, "outcomes": {}} for a in arm_ids}
    by: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arm_ids}
    for r in rows:
        if r["arm_id"] in by:
            by[r["arm_id"]].append(r)
    for a, rs in by.items():
        out[a]["n_blocks"] = len(rs)
        for k in OUTCOMES:
            ys = [float(r[k]) for r in rs if finite(r.get(k))]
            ns = [float(r.get(COUNT_KEY[k]) or 0) for r in rs if finite(r.get(k))]
            m = mean(ys)
            s2 = variance(ys)
            nw = mean(ns) if ns else None
            floor = (m * (1 - m) / nw) if (m is not None and nw) else None
            out[a]["outcomes"][k] = {"n": len(ys), "mean": m, "s2_sample": s2, "n_w": nw, "s2_floor": floor,
                                     "s2": (max(s2 or 0.0, floor or 0.0) if m is not None else None), "values": ys}
    return out


def mean_of(est: Dict[str, Dict[str, Any]], arm: str, k: str) -> Optional[float]:
    return est[arm]["outcomes"][k]["mean"]


def sibling_shares(rows: Sequence[Dict[str, Any]], cl: Dict[str, float]) -> Dict[Tuple[str, int], float]:
    rounds: Dict[int, Dict[str, str]] = {}
    for r in rows:
        rounds.setdefault(int(r["block_id"]), {})[r["node"]] = r["arm_id"]
    e: Dict[Tuple[str, int], float] = {}
    for b, nodes in rounds.items():
        for n in nodes:
            others = [cl.get(arm) for m, arm in nodes.items() if m != n and cl.get(arm) is not None]
            if others:
                e[(n, b)] = sum(others) / len(others)
    return e


def coupling_slopes(rows: Sequence[Dict[str, Any]], arm: str, cl: Dict[str, float]) -> Dict[str, Optional[float]]:
    e = sibling_shares(rows, cl)
    rs = [r for r in rows if r["arm_id"] == arm and (r["node"], int(r["block_id"])) in e]
    out: Dict[str, Optional[float]] = {}
    for k in OUTCOMES:
        pts = [(e[(r["node"], int(r["block_id"]))], float(r[k])) for r in rs if finite(r.get(k))]
        if len(pts) < 4:
            out[k] = None
            continue
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        den = sum((p[0] - mx) ** 2 for p in pts)
        out[k] = (sum((p[0] - mx) * (p[1] - my) for p in pts) / den) if den > 1e-12 else 0.0
    return out


def design_law(members: Sequence[str], a0: str, eps: float = EPS) -> Dict[str, float]:
    n = len(members)
    return {a: (1.0 - eps if a == a0 else 0.0) + eps / n for a in members}


def escalation_gap(members: Sequence[str], a0: str, cl: Dict[str, float], eps: float = EPS, q: Optional[Dict[str, float]] = None) -> float:
    q = q if q is not None else design_law(members, a0, eps)
    e_d = sum(q[a] * cl[a] for a in members)
    vals = [cl[a] for a in members]
    return max(max(vals) - e_d, e_d - min(vals), 0.0)


def zeta_hat(rows: Sequence[Dict[str, Any]], members: Sequence[str], a0: str, cl: Dict[str, float], eps: float = EPS, q: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    gap = escalation_gap(members, a0, cl, eps, q)
    slopes = {a: coupling_slopes(rows, a, cl) for a in members if cl.get(a, 0.0) > 0.0}
    z: Dict[str, float] = {}
    for k in OUTCOMES:
        vals = [abs(s[k]) for s in slopes.values() if s.get(k) is not None]
        z[k] = (max(vals) * gap) if vals else 0.0
    return {"zeta": z, "gap": gap, "slopes": slopes, "members_with_escalation": sorted(slopes)}


class ScreenConfig:
    def __init__(self, arms: Sequence[Dict[str, Any]], tau_table: TauTable, mem_gib, budgets: Optional[Dict[str, float]] = None,
                 n_nodes: int = N_NODES, eps: float = EPS, t_bar: float = T_BAR, b: int = B_EPOCHS, l: int = L_EPOCHS,
                 l_dep: int = L_DEP, k7: int = K7, n_set: int = 3, used_profiles: Optional[Dict[str, Sequence[str]]] = None,
                 cloud_share_mode: str = CLOUD_PER_CONFIGURATION):
        if cloud_share_mode not in CLOUD_MODES:
            raise ValueError(f"cloud_share_mode must be one of {CLOUD_MODES}")
        self.cloud_share_mode = cloud_share_mode
        self.arms = {a["id"]: a for a in arms}
        self.order = {a["id"]: i for i, a in enumerate(arms)}
        self.tau_table = tau_table
        self.mem_gib = mem_gib
        self.budgets = dict(BUDGETS if budgets is None else budgets)
        self.n_nodes, self.eps, self.t_bar, self.b, self.l, self.l_dep, self.k7, self.n_set = n_nodes, eps, t_bar, b, l, l_dep, k7, n_set
        self.profiles = {a["id"]: list(a.get("profiles") or ([a["profile"]] if a.get("profile") else [])) for a in arms}
        for k, v in (used_profiles or {}).items():
            if v:
                self.profiles[k] = list(v)
        self.configs = [a["id"] for a in arms if a["kind"] == "static"]
        self.regimes = [a["id"] for a in arms if a["kind"] == "incumbent"]


def violation_norm(est: Dict[str, Dict[str, Any]], arm: str, budgets: Dict[str, float]) -> Optional[float]:
    v = []
    for k, c in budgets.items():
        m = mean_of(est, arm, k)
        if m is None:
            return None
        v.append((m - c) / c)
    return max(v)


def pick_incumbent(est: Dict[str, Dict[str, Any]], cfg: ScreenConfig) -> Dict[str, Any]:
    rows = []
    for a in cfg.regimes:
        v = violation_norm(est, a, cfg.budgets)
        q = mean_of(est, a, "Y_Q")
        rows.append({"arm": a, "violation": v, "quality": q, "feasible": v is not None and v <= 0.0, "n_blocks": est[a]["n_blocks"]})
    have = [r for r in rows if r["violation"] is not None and r["quality"] is not None]
    feas = [r for r in have if r["feasible"]]
    if feas:
        best = max(r["quality"] for r in feas)
        tie = [r for r in feas if best - r["quality"] <= TIE_Q + 1e-12]
        rule = "highest quality among the incumbents whose means satisfy every budget"
    else:
        best = min(r["violation"] for r in have) if have else None
        tie = [r for r in have if best is not None and r["violation"] - best <= TIE_Q + 1e-12]
        rule = "none satisfies every budget: smallest largest normalized violation"
    if not tie:
        return {"a0": None, "rule": rule, "table": rows}
    i1 = [r for r in tie if r["arm"].startswith("I1")]
    if i1:
        pick = i1[0]
    else:
        pick = sorted(tie, key=(lambda r: -r["quality"]) if feas else (lambda r: r["violation"]))[0]
    return {"a0": pick["arm"], "rule": rule, "tie": [r["arm"] for r in tie], "table": rows}


def eval_member(arm: str, est: Dict[str, Dict[str, Any]], members: Sequence[str], zeta: Dict[str, float], cfg: ScreenConfig,
                with_oracle: bool = False) -> Dict[str, Any]:
    tau = cfg.tau_table.set_tau(cfg.profiles, members)
    band = R.beta_dm(cfg.t_bar, cfg.b, cfg.l, tau)
    eta = R.eta_dep(cfg.t_bar, cfg.l_dep, tau)
    q_a = cfg.eps / cfg.n_set
    u = R.planned_rounds(q_a, cfg.n_nodes, cfg.k7)
    L = R.level(cfg.n_set)
    rows: Dict[str, Any] = {}
    ok = True
    why: List[str] = []
    if est[arm]["n_blocks"] < MIN_BLOCKS:
        return {"arm": arm, "pass": False, "reason": f"fewer than {MIN_BLOCKS} node-blocks ({est[arm]['n_blocks']})", "tau": tau, "eta_dep": eta,
                "u": u, "L": L, "band": band, "budgets": {}}
    ok_oracle = True
    for k, c in cfg.budgets.items():
        o = est[arm]["outcomes"][k]
        if o["mean"] is None:
            ok = False
            why.append(f"{k}: no data")
            rows[k] = {"mean": None}
            continue
        r = R.closed_radius(o["s2"], u, L, band)
        ro = R.oracle_radius(o["s2"], u, L, band)
        z = zeta.get(k, 0.0)
        total = o["mean"] + r + eta + z
        margin = c - total
        rows[k] = {"mean": o["mean"], "s2": o["s2"], "radius": r, "radius_oracle": ro, "eta_dep": eta, "zeta": z, "bound": total,
                   "budget": c, "margin": margin, "pass": margin >= 0.0, "margin_oracle": c - (o["mean"] + ro + eta + z)}
        waived = k == "Y_C" and cfg.cloud_share_mode == CLOUD_SET_LEVEL
        rows[k]["waived"] = waived
        if margin < 0.0 and not waived:
            ok = False
            why.append(f"{k}: {total:.4f} > {c}")
        if c - (o["mean"] + ro + eta + z) < 0.0 and not waived:
            ok_oracle = False
    binding = min((k for k in rows if rows[k].get("margin") is not None and not rows[k].get("waived")), key=lambda k: rows[k]["margin"], default=None)
    return {"arm": arm, "pass": ok, "pass_oracle_radius": ok_oracle, "reason": "; ".join(why), "tau": tau, "eta_dep": eta, "u": u, "L": L,
            "band": band, "budgets": rows, "binding": binding}


def design_law_check(est: Dict[str, Dict[str, Any]], members: Sequence[str], a0: str, cfg: ScreenConfig) -> Dict[str, Any]:
    q = design_law(members, a0, cfg.eps)
    out: Dict[str, Any] = {"q": q, "mixture": {}, "pass": True, "incumbent_exceeds": []}
    for k, c in cfg.budgets.items():
        vals = [mean_of(est, a, k) for a in members]
        inc = mean_of(est, a0, k)
        if any(v is None for v in vals) or inc is None:
            out["mixture"][k] = None
            out["pass"] = False
            continue
        m = sum(q[a] * v for a, v in zip(members, vals))
        exceeds = inc > c
        limit = inc if exceeds else c
        if exceeds:
            out["incumbent_exceeds"].append(k)
        ok = m <= limit + 1e-12
        out["mixture"][k] = {"value": m, "budget": c, "incumbent": inc, "limit": limit, "incumbent_exceeds": exceeds, "pass": ok}
        if not ok:
            out["pass"] = False
    return out


def pick_best(pool: Sequence[str], est: Dict[str, Dict[str, Any]], cfg: ScreenConfig) -> Optional[str]:
    have = [a for a in pool if mean_of(est, a, "Y_Q") is not None]
    if not have:
        return None
    best = max(mean_of(est, a, "Y_Q") for a in have)
    tie = [a for a in have if best - mean_of(est, a, "Y_Q") <= TIE_Q + 1e-12]

    def key(a: str):
        arm = cfg.arms[a]
        mem = cfg.mem_gib(arm["profile"], arm["batch_cap"])
        cl = mean_of(est, a, "Y_C")
        return (mem if mem is not None else 1e9, cl if cl is not None else 1e9, cfg.order[a])
    return sorted(tie, key=key)[0]


def select(rows: Sequence[Dict[str, Any]], est: Dict[str, Dict[str, Any]], cfg: ScreenConfig, a0: str,
           excluded: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    excluded = set(excluded or [])
    cl = {a: (mean_of(est, a, "Y_C") or 0.0) for a in cfg.arms}
    history: List[Dict[str, Any]] = []
    while True:
        pass0 = []
        for c in cfg.configs:
            if c in excluded:
                continue
            ev = eval_member(c, est, [a0, c], {}, cfg)
            if ev["pass"]:
                pass0.append(c)
        rec: Dict[str, Any] = {"excluded": sorted(excluded), "pass_first_screen": list(pass0)}
        a1 = pick_best(pass0, est, cfg)
        if a1 is None:
            rec["outcome"] = "no configuration passes the pass rule"
            history.append(rec)
            return {"S": [a0], "alternatives": [], "history": history, "status": "fewer_than_two_alternatives", "design_law": None, "zeta": None}
        rest = [c for c in pass0 if c != a1]
        a2, dl = None, None
        tried = []
        while rest:
            cand = pick_best(rest, est, cfg)
            dl = design_law_check(est, [a0, a1, cand], a0, cfg)
            tried.append({"candidate": cand, "design_law_pass": dl["pass"]})
            if dl["pass"]:
                a2 = cand
                break
            rest.remove(cand)
        rec["a2_candidates"] = tried
        if a2 is None:
            rec["outcome"] = "no second alternative passes the design-law check"
            history.append(rec)
            return {"S": [a0, a1], "alternatives": [a1], "history": history, "status": "fewer_than_two_alternatives", "design_law": dl, "zeta": None}
        members = [a0, a1, a2]
        zt = zeta_hat(rows, members, a0, cl, cfg.eps)
        evals = {a: eval_member(a, est, members, zt["zeta"], cfg, True) for a in (a1, a2)}
        rec.update({"members": members, "zeta": zt["zeta"], "gap": zt["gap"], "tau": {a: evals[a]["tau"] for a in evals},
                    "pass_in_S": {a: evals[a]["pass"] for a in evals}})
        history.append(rec)
        failing = [a for a in (a1, a2) if not evals[a]["pass"]]
        if not failing:
            return {"S": members, "a0": a0, "alternatives": [a1, a2], "history": history, "status": "selected", "design_law": dl,
                    "zeta": zt, "evals": evals,
                    "tau_members": {a: cfg.tau_table.set_tau(cfg.profiles, members) for a in members}}
        excluded |= set(failing)


def node_effects(rows: Sequence[Dict[str, Any]], est: Dict[str, Dict[str, Any]], outcome: str) -> Dict[str, Any]:
    res: Dict[str, List[float]] = {}
    for r in rows:
        m = est.get(r["arm_id"], {}).get("outcomes", {}).get(outcome, {}).get("mean")
        if m is None or not finite(r.get(outcome)):
            continue
        res.setdefault(r["node"], []).append(float(r[outcome]) - m)
    allr = [x for v in res.values() for x in v]
    if len(res) < 2 or len(allr) < len(res) + 2:
        return {"nodes": {}, "F": None}
    gm = sum(allr) / len(allr)
    ssb = sum(len(v) * (sum(v) / len(v) - gm) ** 2 for v in res.values())
    ssw = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) for v in res.values())
    dfb, dfw = len(res) - 1, len(allr) - len(res)
    F = (ssb / dfb) / (ssw / dfw) if ssw > 0 else None
    return {"nodes": {n: {"mean_residual": sum(v) / len(v), "n": len(v)} for n, v in sorted(res.items())}, "F": F, "df": [dfb, dfw]}


def carryover_by_epoch(requests: Sequence[Dict[str, Any]], arm_of_block: Dict[Tuple[str, int], str], arm_ids: Sequence[str],
                       l: int = L_EPOCHS) -> Dict[str, Any]:
    acc: Dict[str, List[List[float]]] = {a: [[] for _ in range(l)] for a in arm_ids}
    for r in requests:
        k = r.get("epoch_in_block")
        lat = r.get("latency_s")
        a = arm_of_block.get((r["node"], r.get("block_id")))
        if a in acc and k is not None and 0 <= k < l and lat is not None and r.get("ok"):
            acc[a][k].append(lat)
    out: Dict[str, Any] = {}
    for a, per in acc.items():
        u = [sum(x) / len(x) if x else None for x in per]
        if sum(1 for x in per if len(x) >= 5) < l * 0.8:
            continue
        uu = [x if x is not None else (u[i - 1] if i and u[i - 1] is not None else 0.0) for i, x in enumerate(u)]
        plat = uu[l - 15:]
        mu = sum(plat) / len(plat)
        sd = (sum((x - mu) ** 2 for x in plat) / (len(plat) - 1)) ** 0.5
        D = [abs(x - mu) for x in uu]
        thr = max(0.1 * (D[0] + D[1]) / 2.0, 2.0 * sd)
        t = next((k for k in range(l) if all(D[j] <= thr for j in range(k, l))), None)
        out[a] = {"t_bar_epochs": t, "plateau": mu, "u": [round(x, 3) for x in uu]}
    ts = [v["t_bar_epochs"] for v in out.values() if v["t_bar_epochs"] is not None]
    return {"per_arm": out, "median": (sorted(ts)[len(ts) // 2] if ts else None), "max": (max(ts) if ts else None)}


def replace_heavy(rows: Sequence[Dict[str, Any]], est: Dict[str, Dict[str, Any]], cfg: ScreenConfig, a0: str, members: Sequence[str],
                  heavy: str) -> Optional[Dict[str, Any]]:
    if heavy == a0:
        return None
    cl = {a: (mean_of(est, a, "Y_C") or 0.0) for a in cfg.arms}
    rest = [c for c in cfg.configs if c not in members and cl[c] < cl[heavy] and eval_member(c, est, [a0, c], {}, cfg)["pass"]]
    while rest:
        cand = pick_best(rest, est, cfg)
        rest.remove(cand)
        new = [cand if m == heavy else m for m in members]
        dl = design_law_check(est, new, a0, cfg)
        if not dl["pass"]:
            continue
        zt = zeta_hat(rows, new, a0, cl, cfg.eps)
        evs = {a: eval_member(a, est, new, zt["zeta"], cfg, True) for a in new if a != a0}
        if all(e["pass"] for e in evs.values()):
            return {"S": new, "replaced": heavy, "by": cand, "design_law": dl, "zeta": zt,
                    "tau_members": {a: cfg.tau_table.set_tau(cfg.profiles, new) for a in new}}
    return None


def keep_non_escalating(members: Sequence[str], a0: str, cfg: ScreenConfig) -> List[str]:
    return [m for m in members if m == a0 or (cfg.arms[m]["kind"] == "static" and not cfg.arms[m].get("kappa"))]
