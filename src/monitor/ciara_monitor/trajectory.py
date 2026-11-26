from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence

from ciara import program as CT
from ciara import looks as LK
from ciara.constants import BUDGETS, BUDGET_KEYS

MODES = ("a5pp", "a5pp_strict", "estimated")
BIND_EPS = 1e-5


def program_at(dec: Dict[str, Any], members: Sequence[str], a0: str, mode: str, budgets: Dict[str, float], zeta: Dict[str, float], share_shift: float = 0.0) -> Dict[str, Any]:
    tab = dec["arms_table"]
    mu = {k: [tab[a]["outcomes"][k]["centre"] for a in members] for k in tab[members[0]]["outcomes"]}
    rr = {k: [tab[a]["outcomes"][k]["radius"] for a in members] for k in tab[members[0]]["outcomes"]}
    p = dec["parameters"]
    one_sided = bool(p.get("one_sided"))
    eta = float(dec["eta_dep"])
    if mode == "estimated":
        return CT.certified_policy(list(members), mu, rr, zeta, eta, a0, budgets, one_sided_quality=one_sided)
    qs = [float(tab[a]["propensity"]) for a in members]
    n_nodes = int(p.get("n_nodes") or 4)
    if share_shift:
        e_design = sum(q * mu["Y_C"][i] for i, q in enumerate(qs))
        e_lower = sum(q * (mu["Y_C"][i] - rr["Y_C"][i]) for i, q in enumerate(qs))
        return CT.certified_policy(list(members), mu, rr, zeta, eta, a0, budgets, one_sided_quality=one_sided, monotone=True, design_share=(e_design - share_shift, e_lower),
                                   strict_support=(mode == "a5pp_strict"), n_nodes=n_nodes)
    return CT.certified_policy(list(members), mu, rr, zeta, eta, a0, budgets, one_sided_quality=one_sided, monotone=True, design_q=qs, strict_support=(mode == "a5pp_strict"), n_nodes=n_nodes)


def uniform_margin(dec: Dict[str, Any], members: Sequence[str], a0: str, mode: str, zeta: Dict[str, float], budgets: Optional[Dict[str, float]] = None,
                   tol: float = 1e-6) -> Dict[str, Any]:
    budgets = dict(BUDGETS if budgets is None else budgets)

    def solve(t: float) -> Dict[str, Any]:
        return program_at(dec, members, a0, mode, {k: v - t for k, v in budgets.items()}, zeta, share_shift=(t if mode != "estimated" else 0.0))

    lo, hi = -1.0, 1.0
    if solve(lo).get("lp_status") != "optimal":
        return {"mode": mode, "t_star": None, "note": "infeasible even at a uniform violation of 1.0"}
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if solve(mid).get("lp_status") == "optimal":
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    t = lo
    res = solve(t - BIND_EPS)
    sl = {k: float(v) for k, v in (res.get("slack") or {}).items()}
    binding = sorted(k for k, v in sl.items() if v < 10 * BIND_EPS)
    sc = res.get("share_constraint") or {}
    if sc and sc.get("margin") is not None and sc["margin"] < 10 * BIND_EPS:
        binding = sorted(binding + ["share"])
    reg = program_at(dec, members, a0, mode, budgets, zeta)
    return {"mode": mode, "t_star": t, "binding_rows": binding, "policy_at_t_star": res.get("pi"), "quality_cost_bound_at_t_star": res.get("quality_cost_bound"),
            "reference_feasible": reg.get("lp_status") == "optimal", "reference_policy": reg.get("pi"), "reference_slack": reg.get("slack"),
            "reference_switch": reg.get("switch"), "reference_quality_cost_bound": reg.get("quality_cost_bound"), "branch": reg.get("branch"),
            "incumbent_violations": reg.get("incumbent_violations"), "share_constraint": reg.get("share_constraint")}


def row_floor(dec: Dict[str, Any], members: Sequence[str]) -> Dict[str, Dict[str, float]]:
    tab = dec["arms_table"]
    eta = float(dec["eta_dep"])
    out = {}
    for k in BUDGET_KEYS:
        ub = {a: tab[a]["outcomes"][k]["upper"] + eta for a in members}
        a_best = min(ub, key=ub.get)
        out[k] = {"best_arm": a_best, "best_upper_with_eta": ub[a_best], "budget": BUDGETS[k], "margin": BUDGETS[k] - ub[a_best],
                  "uppers_with_eta": {a: ub[a] for a in members}}
    return out


def extend_rows(rows: Sequence[Dict[str, Any]], k_obs: int, k_target: int, order: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
    order = list(order) if order is not None else list(range(k_obs))
    by: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        b = int(r["block_id"])
        if b < k_obs and r.get("window", "measure") == "measure":
            by.setdefault(b, []).append(r)
    out: List[Dict[str, Any]] = []
    for j in range(k_target):
        src = by.get(order[j % len(order)], [])
        for r in src:
            q = dict(r)
            q["block_id"] = j
            out.append(q)
    return out


def projected_decision(decide: Callable[..., Dict[str, Any]], rows: Sequence[Dict[str, Any]], k_obs: int, k_target: int, order: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    return decide(extend_rows(rows, k_obs, k_target, order), k_target)


def bootstrap(decide: Callable[..., Dict[str, Any]], rows: Sequence[Dict[str, Any]], k_obs: int, k_targets: Sequence[int], members: Sequence[str], a0: str,
              zeta: Dict[str, float], reps: int = 100, seed: int = 20261026) -> Dict[str, Any]:
    rng = random.Random(seed)
    res: Dict[int, Dict[str, List[float]]] = {k: {"switch": [], "switch8": [], "feasible": [], "t_star": [], "binding": []} for k in k_targets}
    for _ in range(reps):
        order = [rng.randrange(k_obs) for _ in range(k_obs)]
        for kt in k_targets:
            ext = extend_rows(rows, k_obs, kt, order)
            dec = decide(ext, kt)
            res[kt]["switch"].append(1.0 if dec.get("switch") else 0.0)
            res[kt]["switch8"].append(1.0 if decide(ext, kt, 8.0).get("switch") else 0.0)
            ap = program_at(dec, members, a0, "a5pp", BUDGETS, {})
            res[kt]["feasible"].append(1.0 if ap.get("lp_status") == "optimal" else 0.0)
            um = uniform_margin(dec, members, a0, "a5pp", {})
            res[kt]["t_star"].append(float(um["t_star"]) if um.get("t_star") is not None else float("nan"))
            res[kt]["binding"].append(",".join(um.get("binding_rows") or []))
    out: Dict[str, Any] = {"reps": reps, "seed": seed, "targets": {}}
    for kt, d in res.items():
        ts = sorted(x for x in d["t_star"] if x == x)
        q = lambda p: ts[min(len(ts) - 1, int(p * (len(ts) - 1) + 0.5))] if ts else None
        cnt: Dict[str, int] = {}
        for b in d["binding"]:
            cnt[b] = cnt.get(b, 0) + 1
        out["targets"][str(kt)] = {"share_switch": sum(d["switch"]) / len(d["switch"]), "share_switch_t_bar_8": sum(d["switch8"]) / len(d["switch8"]), "share_a5pp_feasible": sum(d["feasible"]) / len(d["feasible"]),
                                   "t_star_quantiles": {"q05": q(0.05), "q25": q(0.25), "q50": q(0.5), "q75": q(0.75), "q95": q(0.95)},
                                   "binding_sets": dict(sorted(cnt.items(), key=lambda kv: -kv[1])[:5])}
    return out


def trajectory(decide: Callable[..., Dict[str, Any]], rows: Sequence[Dict[str, Any]], k_obs: int, members: Sequence[str], a0: str, zeta: Dict[str, float],
               step: int = 4, mode: str = "a5pp") -> List[Dict[str, Any]]:
    ks = sorted(set(list(range(step, k_obs + 1, step)) + [k_obs]))
    out = []
    for k in ks:
        dec = decide(rows, k)
        um = uniform_margin(dec, members, a0, mode, zeta)
        out.append({"rounds": k, "t_star": um.get("t_star"), "binding_rows": um.get("binding_rows"), "reference_feasible": um.get("reference_feasible"),
                    "quality_cost_bound_at_t_star": um.get("quality_cost_bound_at_t_star"), "switch_gated": dec.get("switch")})
    return out
