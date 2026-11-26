from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .constants import BUDGETS, BUDGET_KEYS

TOL = 1e-9
MIN_MASS = 0.01


def solve_simplex_lp(c: Sequence[float], G: Sequence[Sequence[float]], h: Sequence[float], tie_weight: Optional[Sequence[float]] = None) -> Dict:
    c = np.asarray(c, float)
    n = len(c)
    G = np.asarray(G, float).reshape(-1, n)
    h = np.asarray(h, float).reshape(-1)
    tw = np.zeros(n) if tie_weight is None else np.asarray(tie_weight, float)
    A = np.vstack([G, -np.eye(n)])
    d = np.concatenate([h, np.zeros(n)])
    best = None
    if n == 1:
        cand = [np.array([1.0])]
    else:
        cand = []
        for act in itertools.combinations(range(A.shape[0]), n - 1):
            M = np.vstack([A[list(act)], np.ones((1, n))])
            rhs = np.concatenate([d[list(act)], [1.0]])
            try:
                p = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                continue
            cand.append(p)
    for p in cand:
        if np.any(p < -TOL) or np.any(A @ p > d + 1e-9):
            continue
        p = np.clip(p, 0.0, None)
        p = p / p.sum()
        key = (float(c @ p), float(tw @ p))
        if best is None or key[0] > best[0][0] + 1e-12 or (abs(key[0] - best[0][0]) <= 1e-12 and key[1] > best[0][1] + 1e-12):
            best = (key, p)
    if best is None:
        return {"status": "infeasible", "p": None, "value": float("nan")}
    return {"status": "optimal", "p": [float(x) for x in best[1]], "value": best[0][0]}


ESCALATION_ROW = "Y_C"


def monotone_zeta(zeta: Dict[str, float]) -> Dict[str, float]:
    return {k: (v if k == "Y_Q" else 0.0) for k, v in zeta.items()}


def certified_policy(arms: Sequence[str], mu: Dict[str, Sequence[float]], r: Dict[str, Sequence[float]], zeta: Dict[str, float],
                     eta: float, a0: str, budgets: Optional[Dict[str, float]] = None, min_mass: float = MIN_MASS, one_sided_quality: bool = False,
                     monotone: bool = False, design_q: Optional[Sequence[float]] = None, strict_support: bool = False, n_nodes: int = 4,
                     design_share: Optional[Sequence[float]] = None) -> Dict:
    budgets = dict(BUDGETS if budgets is None else budgets)
    n = len(arms)
    i0 = list(arms).index(a0)
    if monotone and design_q is None and design_share is None:
        raise ValueError("A5'' (monotone coupling) needs the design law: pass design_q (or design_share) with the constraint on the escalated share")
    zeta_reg = dict(zeta)
    zeta = monotone_zeta(zeta) if monotone else dict(zeta)
    obj = [mu["Y_Q"][i] - r["Y_Q"][i] for i in range(n)]
    G, h, names = [], [], []
    for k in BUDGET_KEYS:
        if k not in budgets:
            continue
        G.append([mu[k][i] + r[k][i] for i in range(n)])
        h.append(budgets[k] - eta - zeta.get(k, 0.0))
        names.append(k)
    share_c = None
    if design_q is not None or design_share is not None:
        if design_share is not None:
            e_design, e_lower = float(design_share[0]), float(design_share[1])
        else:
            e_design = sum(design_q[i] * mu[ESCALATION_ROW][i] for i in range(n))
            e_lower = sum(design_q[i] * (mu[ESCALATION_ROW][i] - r[ESCALATION_ROW][i]) for i in range(n))
        G.append([mu[ESCALATION_ROW][i] + r[ESCALATION_ROW][i] for i in range(n)])
        h.append(e_design)
        share_c = {"design_share": e_design, "row": ESCALATION_ROW, "design_q": None if design_q is None else [float(x) for x in design_q], "design_share_lower": e_lower}
    forbidden: List[int] = []
    if strict_support:
        if share_c is None:
            raise ValueError("strict_support needs design_q")
        forbidden = [i for i in range(n) if mu[ESCALATION_ROW][i] + r[ESCALATION_ROW][i] > share_c["design_share_lower"] + 1e-12]
        for i in forbidden:
            G.append([1.0 if j == i else 0.0 for j in range(n)])
            h.append(0.0)
        share_c["forbidden_arms"] = [list(arms)[i] for i in forbidden]
    tw = [0.0 if i == i0 else 1.0 for i in range(n)]
    sol = solve_simplex_lp(obj, G, h, tw)
    inc_zeta = zeta
    if monotone and mu[ESCALATION_ROW][i0] + r[ESCALATION_ROW][i0] > share_c["design_share"] + 1e-12:
        inc_zeta = zeta_reg
    inc_ub = {k: mu[k][i0] + r[k][i0] + eta + inc_zeta.get(k, 0.0) for k in names}
    inc_viol = {k: inc_ub[k] - budgets[k] for k in names if inc_ub[k] > budgets[k] + 1e-12}
    inc_feasible = not inc_viol
    branch = "gate" if inc_feasible else "budget_compliance"
    out: Dict = {"arms": list(arms), "a0": a0, "eta_dep": eta, "zeta": dict(zeta), "budgets": budgets, "lp_status": sol["status"],
                 "min_mass": min_mass, "branch": branch, "incumbent_feasible": inc_feasible, "incumbent_upper_bounds": inc_ub,
                 "incumbent_violations": inc_viol, "monotone_coupling": bool(monotone), "zeta_reference": zeta_reg}
    if share_c is not None:
        out["share_constraint"] = share_c
    if sol["status"] != "optimal":
        out.update({"switch": False, "reason": ("LP infeasible: no certified policy exists, the incumbent is kept" if inc_feasible else
                                                "the incumbent exceeds a budget at its upper bound and the LP is infeasible: no certified policy exists, the incumbent is kept"),
                    "pi": None})
        return out
    p = sol["p"]
    mass = 1.0 - p[i0]
    gate = sum(p[i] * ((mu["Y_Q"][i] - r["Y_Q"][i]) - (mu["Y_Q"][i0] + r["Y_Q"][i0])) for i in range(n) if i != i0)
    gate -= eta + 2.0 * zeta.get("Y_Q", 0.0)
    ub = {k: sum(p[i] * (mu[k][i] + r[k][i]) for i in range(n)) + eta + zeta.get(k, 0.0) for k in names}
    slack = {k: budgets[k] - ub[k] for k in names}
    if share_c is not None:
        lhs = sum(p[i] * (mu[ESCALATION_ROW][i] + r[ESCALATION_ROW][i]) for i in range(n))
        e_c = [mu[ESCALATION_ROW][i] for i in range(n)]
        e_pi = sum(p[i] * e_c[i] for i in range(n))
        support = [i for i in range(n) if p[i] > 1e-9]
        share_c.update({"policy_share_upper": lhs, "margin": share_c["design_share"] - lhs, "holds": lhs <= share_c["design_share"] + 1e-9,
                        "margin_to_lower_design_side": share_c["design_share_lower"] - lhs,
                        "support_max_share_center": max(e_c[i] for i in support), "support_max_share_upper": max(mu[ESCALATION_ROW][i] + r[ESCALATION_ROW][i] for i in support),
                        "support_dominance_center": all(e_c[i] <= share_c["design_share"] + 1e-9 for i in support),
                        "support_dominance_strict": all(mu[ESCALATION_ROW][i] + r[ESCALATION_ROW][i] <= share_c["design_share_lower"] + 1e-9 for i in support),
                        "residual_per_gamma": (sum(p[i] * e_c[i] ** 2 for i in range(n)) + (n_nodes - 1) * e_pi ** 2 - n_nodes * share_c["design_share"] * e_pi) / (n_nodes - 1)})
    if one_sided_quality:
        qcost = sum(p[i] * ((mu["Y_Q"][i0] + r["Y_Q"][i0]) - (mu["Y_Q"][i] - r["Y_Q"][i])) for i in range(n) if i != i0) + eta + 2.0 * zeta.get("Y_Q", 0.0)
    else:
        qcost = (mu["Y_Q"][i0] + r["Y_Q"][i0]) - sum(p[i] * (mu["Y_Q"][i] - r["Y_Q"][i]) for i in range(n)) + eta + 2.0 * zeta.get("Y_Q", 0.0)
    if inc_feasible:
        ok = mass >= min_mass - 1e-12 and gate > 0.0
        reason = "certified policy deployed (incumbent certified feasible, paired gate passed)" if ok else (
            "non-incumbent mass below %.2f" % min_mass if mass < min_mass - 1e-12 else "paired gate not positive: the incumbent is kept")
    else:
        ok = mass > 1e-9
        reason = ("certified budget-compliant policy deployed: the incumbent exceeds " + ", ".join(sorted(inc_viol)) + " at its upper bound; the gate does not apply"
                  if ok else "the LP solution equals the incumbent: nothing to deploy")
    out.update({"pi": {a: p[i] for i, a in enumerate(arms)}, "objective_lower_bound": sol["value"], "nonincumbent_mass": mass,
                "gate_value": gate, "gate_applies": inc_feasible, "gate_pass": bool(ok) if inc_feasible else None, "switch": bool(ok), "reason": reason,
                "upper_bounds": ub, "slack": slack, "quality_cost_bound": qcost,
                "quality_cost_note": ("upper bound on J_Q(a0) - J_Q(pi), one-sided form: sum over alternatives of pi (c_Q(a0) + r_Q(a0) - c_Q(a) + r_Q(a)) + eta_dep + 2 zeta_Q; negative means pi is certified better than the incumbent"
                         if one_sided_quality else
                         "upper bound on J_Q(a0) - J_Q(pi): (c_Q(a0) + r_Q(a0)) - sum pi (c_Q - r_Q) + eta_dep + 2 zeta_Q; negative means pi is certified better than the incumbent")})
    return out


def look_kwargs(look: Dict) -> Dict:
    c = look.get("coupling") or {}
    sc = look.get("share_constraint")
    mode = c.get("applied") or c.get("mode")
    if mode in ("a5pp", "a5pp_strict") and sc:
        return {"zeta": dict(c.get("zeta_reference") or look.get("zeta") or {}), "monotone": True, "design_share": (sc["design_share"], sc["design_share_lower"]),
                "strict_support": mode == "a5pp_strict", "n_nodes": int((look.get("parameters") or {}).get("n_nodes") or 4)}
    return {"zeta": dict(look.get("zeta") or {})}
