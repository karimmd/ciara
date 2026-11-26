from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence

from . import program as CT
from . import radius as R
from . import screening as S
from .constants import OUTCOMES

MIN_ALTERNATIVES = 2
COST = tuple(k for k in OUTCOMES if k != "Y_Q")


def standard_errors(est: Dict[str, Dict[str, Any]], arm: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k in OUTCOMES:
        o = est[arm]["outcomes"][k]
        out[k] = math.sqrt(max(o["s2"], 0.0) / o["n"]) if (o["mean"] is not None and o["s2"] is not None and o["n"] > 0) else None
    return out


def shift_adverse(est: Dict[str, Dict[str, Any]], arms: Sequence[str], mult: float = 1.0) -> Dict[str, Dict[str, Any]]:
    new = copy.deepcopy(est)
    for a in arms:
        se = standard_errors(est, a)
        for k in OUTCOMES:
            o = new[a]["outcomes"][k]
            if o["mean"] is None or se[k] is None:
                continue
            o["mean"] = o["mean"] + mult * se[k] if k in COST else o["mean"] - mult * se[k]
    return new


def weights(prog: Dict[str, Any], a0: str, est: Dict[str, Dict[str, Any]], order: Sequence[str]) -> List[Dict[str, Any]]:
    pi = prog.get("pi") or {}
    rows = []
    for i, a in enumerate(order):
        if a == a0:
            continue
        b = (prog.get("bounds") or {}).get(a) or {}
        q = ((b.get("outcomes") or {}).get("Y_Q")) or {}
        lo = (q.get("mean") - q.get("radius")) if q.get("mean") is not None else float("-inf")
        rows.append({"arm": a, "weight": float(pi.get(a, 0.0)), "quality_lower": lo, "order": i})
    rows.sort(key=lambda r: (-round(r["weight"], 9), -r["quality_lower"], r["order"]))
    return rows


def member_bounds(arm: str, est: Dict[str, Dict[str, Any]], members: Sequence[str], a0: str, cfg: S.ScreenConfig, u_fixed: Optional[float] = None,
                  one_sided: bool = False) -> Dict[str, Any]:
    q = S.design_law(members, a0, cfg.eps)[arm]
    tau = cfg.tau_table.set_tau(cfg.profiles, members)
    band = R.beta_dm(cfg.t_bar, cfg.b, cfg.l, tau)
    eta = R.eta_dep(cfg.t_bar, cfg.l_dep, tau)
    u = R.planned_rounds(q, cfg.n_nodes, cfg.k7) if u_fixed is None else float(u_fixed)
    L = R.level(cfg.n_set, one_sided=one_sided)
    out: Dict[str, Any] = {"arm": arm, "q": q, "u": u, "L": L, "band": band, "eta_dep": eta, "tau": tau, "n_blocks": est[arm]["n_blocks"], "outcomes": {}}
    for k in OUTCOMES:
        o = est[arm]["outcomes"][k]
        if o["mean"] is None or o["s2"] is None:
            out["outcomes"][k] = None
            continue
        out["outcomes"][k] = {"mean": o["mean"], "radius": R.closed_radius(o["s2"], u, L, band)}
    return out


def program(est: Dict[str, Dict[str, Any]], members: Sequence[str], a0: str, cfg: S.ScreenConfig, zeta: Optional[Dict[str, float]] = None,
            u_fixed: Optional[float] = None, one_sided: bool = False) -> Dict[str, Any]:
    zeta = dict(zeta or {})
    bounds = {a: member_bounds(a, est, members, a0, cfg, u_fixed, one_sided) for a in members}
    short = [a for a in members if est[a]["n_blocks"] < S.MIN_BLOCKS]
    missing = [(a, k) for a in members for k in OUTCOMES if bounds[a]["outcomes"][k] is None]
    base: Dict[str, Any] = {"members": list(members), "a0": a0, "zeta": zeta, "bounds": bounds,
                            "budgets": dict(cfg.budgets)}
    if short or missing:
        base.update({"feasible": False, "lp_status": "no_estimate", "reason": (f"fewer than {S.MIN_BLOCKS} node-blocks for {short}" if short else f"no estimate for {missing[:3]}")})
        return base
    eta = bounds[a0]["eta_dep"]
    mu = {k: [bounds[a]["outcomes"][k]["mean"] for a in members] for k in OUTCOMES}
    r = {k: [bounds[a]["outcomes"][k]["radius"] for a in members] for k in OUTCOMES}
    res = CT.certified_policy(list(members), mu, r, zeta, eta, a0, dict(cfg.budgets), one_sided_quality=one_sided)
    ub = {k: {a: mu[k][i] + r[k][i] + eta + zeta.get(k, 0.0) for i, a in enumerate(members)} for k in cfg.budgets}
    min_ub = {k: min(v.values()) for k, v in ub.items()}
    infeasible_rows = [k for k, c in cfg.budgets.items() if min_ub[k] > c + 1e-12]
    base.update({"eta_dep": eta, "member_upper_bounds": ub, "min_member_upper_bound": min_ub, "rows_no_member_meets": infeasible_rows,
                 "lp_status": res["lp_status"], "feasible": res["lp_status"] == "optimal", "pi": res.get("pi"), "upper_bounds": res.get("upper_bounds"), "slack": res.get("slack"),
                 "objective_lower_bound": res.get("objective_lower_bound"), "incumbent_feasible": res["incumbent_feasible"], "reason": res.get("reason"),
                 "quality_cost_bound": res.get("quality_cost_bound")})
    if infeasible_rows and base["feasible"]:
        base["feasible"], base["lp_status"] = False, "inconsistent"
        base["reason"] = f"solver reported a point although no member meets {infeasible_rows}"
    return base


def informational_set(rows: Sequence[Dict[str, Any]], est: Dict[str, Dict[str, Any]], cfg: S.ScreenConfig, a0: str, k: int = 2) -> List[str]:
    scored = []
    for c in cfg.configs:
        ev = S.eval_member(c, est, [a0, c], {}, cfg)
        b = ev.get("binding")
        if b is None:
            continue
        scored.append((ev["budgets"][b]["margin"], -cfg.order[c], c))
    scored.sort(reverse=True)
    return [a0] + [c for _, _, c in scored[:k]]


def evaluate(rows: Sequence[Dict[str, Any]], est: Dict[str, Dict[str, Any]], cfg: S.ScreenConfig, a0: str, sel: Dict[str, Any]) -> Dict[str, Any]:
    alts = list(sel.get("alternatives") or [])
    selected = sel.get("status") == "selected" and len(alts) >= MIN_ALTERNATIVES
    out: Dict[str, Any] = {"rule": "at least two alternatives pass the pass rule, and the pessimistic program on the screening estimates of S (mean + radius + eta_dep) is feasible at every budget",
                           "n_alternatives": len(alts), "selected": selected, "alternatives": alts}
    if selected:
        members = list(sel["S"])
        z = ((sel.get("zeta") or {}).get("zeta")) or {}
        prog = program(est, members, a0, cfg, None)
        out.update({"set": members, "program": prog, "with_zeta": program(est, members, a0, cfg, z), "informational": False})
        out["go"] = bool(prog["feasible"])
        out["reason"] = ("two alternatives pass and the certified program is feasible" if out["go"] else
                         "two alternatives pass, but the certified program on S has no feasible point" + (f" (no member meets {prog['rows_no_member_meets']})" if prog.get("rows_no_member_meets") else ""))
        return out
    members = informational_set(rows, est, cfg, a0)
    out.update({"go": False, "informational": True, "set": members,
                "reason": f"{len(alts)} alternative(s) pass the pass rule, two are required"})
    if len(members) >= 2 and all(est[a]["n_blocks"] > 0 for a in members):
        cl = {a: (S.mean_of(est, a, "Y_C") or 0.0) for a in cfg.arms}
        z = S.zeta_hat(rows, members, a0, cl, cfg.eps)["zeta"]
        out["program"] = program(est, members, a0, cfg, None)
        out["with_zeta"] = program(est, members, a0, cfg, z)
    return out
