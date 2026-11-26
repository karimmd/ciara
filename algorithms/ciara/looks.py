from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import band as MB
from . import program as CT
from . import radius as R
from .constants import B_EPOCHS, BUDGET_KEYS, BUDGETS, DELTA, EPOCHS_PER_DAY, EPS, K7, L_DEP, L_EPOCHS, N_NODES, OUTCOMES, T_BAR, finite
from .transient import TauTable

INF_SENTINEL = 1.0e6
BASE_MODES = ("a5pp", "a5pp_strict", "estimated")
COUPLING_MODES = ("a5pp_gated",) + BASE_MODES
RHO_TOL = 1e-12
T_BAR_ROBUST = 8.0


def look_blocks(day: int, l: int = L_EPOCHS) -> int:
    return (day * EPOCHS_PER_DAY) // l


def round_table(rows: Sequence[Dict[str, Any]], arm_ids: Sequence[str], n_nodes: int, k_rounds: int):
    idx = {a: i for i, a in enumerate(arm_ids)}
    Z = -np.ones((k_rounds, n_nodes), dtype=np.int64)
    F = np.full((k_rounds, n_nodes, len(OUTCOMES)), np.nan)
    for r in rows:
        b = int(r["block_id"])
        n = int(r.get("node_index") or int(str(r["node"]).lstrip("E"))) - 1
        if not (0 <= b < k_rounds and 0 <= n < n_nodes) or r.get("arm_id") not in idx or r.get("block_status") != "complete":
            continue
        Z[b, n] = idx[r["arm_id"]]
        for j, k in enumerate(OUTCOMES):
            v = r.get(k)
            if finite(v):
                F[b, n, j] = float(v)
    return Z, F


def look_decision(rows: Sequence[Dict[str, Any]], members: Sequence[str], a0: str, profiles: Dict[str, Sequence[str]], tau_table: TauTable,
                  zeta: Dict[str, float], day: int, budgets: Optional[Dict[str, float]] = None, n_nodes: int = N_NODES, eps: float = EPS,
                  t_bar: float = T_BAR, b: int = B_EPOCHS, l: int = L_EPOCHS, l_dep: int = L_DEP, k7: int = K7, k_rounds: Optional[int] = None,
                  completed_rounds: Optional[int] = None, composition: Optional[Sequence[str]] = None, one_sided: bool = False,
                  coupling_mode: str = "estimated", t_bar_robust: Optional[float] = None, measured_band: Optional[Dict[str, Any]] = None,
                  band_by_outcome: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    if coupling_mode not in COUPLING_MODES:
        raise ValueError(f"coupling_mode must be one of {COUPLING_MODES}")
    budgets = dict(BUDGETS if budgets is None else budgets)
    members = list(members)
    n_set = len(members)
    k_look = k_rounds if k_rounds is not None else look_blocks(day, l)
    Z, F = round_table(rows, members, n_nodes, k_look)
    full_rounds = int(np.sum(np.all(Z >= 0, axis=1)))
    if composition:
        q = np.array([list(composition).count(a) / len(composition) for a in members])
        nplan = np.array([R.planned_rounds_balanced(k7) for _ in members])
    else:
        q = np.array([(1.0 - eps if a == a0 else 0.0) + eps / n_set for a in members])
        nplan = np.array([R.planned_rounds(float(qi), n_nodes, k7) for qi in q])
    tau_s = tau_table.set_tau(profiles, members)
    taus = {a: tau_s for a in members}
    bands = np.array([R.beta_dm(t_bar, b, l, taus[a]) for a in members])
    L = R.level(n_set, one_sided=one_sided)
    band_in = bands
    if band_by_outcome:
        band_in = np.array([[float(band_by_outcome.get(k, bands[i])) for i in range(n_set)] for k in OUTCOMES])
    out = R.c1r_intervals(Z, F, n_nodes, L, [k_look], nplan, band_in, n_arms=n_set)
    cen, rad = out["centre"][0], out["radius"][0]
    mu = {k: [float(cen[j, i]) for i in range(n_set)] for j, k in enumerate(OUTCOMES)}
    rr = {k: [float(rad[j, i]) if math.isfinite(rad[j, i]) else INF_SENTINEL for i in range(n_set)] for j, k in enumerate(OUTCOMES)}
    tau_max = max(taus.values())
    eta = R.eta_dep(t_bar, l_dep, tau_max)
    qs = [float(x) for x in q]

    def program(mode: str) -> Dict[str, Any]:
        if mode == "estimated":
            return CT.certified_policy(members, mu, rr, zeta, eta, a0, budgets, one_sided_quality=one_sided)
        return CT.certified_policy(members, mu, rr, zeta, eta, a0, budgets, one_sided_quality=one_sided, monotone=True, design_q=qs, strict_support=(mode == "a5pp_strict"),
                                   n_nodes=n_nodes)

    def brief(d: Dict[str, Any]) -> Dict[str, Any]:
        sc = d.get("share_constraint") or {}
        return {"switch": d.get("switch"), "reason": d.get("reason"), "lp_status": d.get("lp_status"), "pi": d.get("pi"), "branch": d.get("branch"),
                "quality_cost_bound": d.get("quality_cost_bound"), "slack": d.get("slack"), "forbidden_arms": sc.get("forbidden_arms")}

    gated = None
    if coupling_mode == "a5pp_gated":
        dec = program("a5pp")
        applied = "a5pp"
        a5pp_pi = dec.get("pi")
        sc0 = dec.get("share_constraint") or {}
        rho = sc0.get("residual_per_gamma")
        if dec.get("lp_status") != "optimal":
            gated = {"decided_by": "a5pp", "reason": "the a5pp program is infeasible (so is the strict program, which only removes arms): the incumbent is kept"}
        elif rho <= RHO_TOL:
            gated = {"decided_by": "a5pp", "reason": "rho(pi_hat) <= 0: the a5pp solution is deployed (A5'' in affine form, own-arm interaction of any nonnegative size)"}
        else:
            dec = program("a5pp_strict")
            applied = "a5pp_strict"
            gated = {"decided_by": "a5pp_strict", "reason": ("rho(pi_hat) > 0: the strict program is feasible and its solution is deployed (support condition, A5'' alone)"
                                                              if dec.get("lp_status") == "optimal" else
                                                              "rho(pi_hat) > 0 and the strict program is infeasible: the incumbent is kept")}
        gated.update({"rho": rho, "rho_of": "a5pp", "rho_tolerance": RHO_TOL, "a5pp_pi": a5pp_pi,
                      "support": {k: sc0.get(k) for k in ("support_max_share_center", "support_max_share_upper", "support_dominance_center", "support_dominance_strict", "design_share",
                                                          "design_share_lower")}})
    else:
        dec = program(coupling_mode)
        applied = coupling_mode
    dec["coupling"] = {"mode": coupling_mode, "applied": applied, "zeta_reference": dict(zeta), "zeta_used": dict(dec.get("zeta") or {}), "gated": gated,
                       "sensitivity": {m: brief(program(m)) for m in BASE_MODES if m != applied}}
    nonempty = {a: [float(out["nonempty"][0][j, i]) for j in range(len(OUTCOMES))] for i, a in enumerate(members)}
    arms_tab = {}
    for i, a in enumerate(members):
        arms_tab[a] = {"propensity": float(q[i]), "tau": taus[a], "band": float(bands[i]), "planned_rounds": float(nplan[i]),
                       "nonempty_rounds": dict(zip(OUTCOMES, nonempty[a])),
                       "outcomes": {k: {"centre": mu[k][i], "radius": rr[k][i], "lower": mu[k][i] - rr[k][i], "upper": mu[k][i] + rr[k][i]} for k in OUTCOMES}}
    if t_bar_robust is not None and t_bar_robust != t_bar:
        alt = look_decision(rows, members, a0, profiles, tau_table, zeta, day, budgets, n_nodes, eps, t_bar_robust, b, l, l_dep, k7, k_rounds, completed_rounds, composition, one_sided,
                            coupling_mode, None)
        dec["robustness"] = {"t_bar": t_bar_robust, "reference_t_bar": t_bar, "deciding": False, **sensitivity_summary(alt),
                             "bands": {a: t["band"] for a, t in alt["arms_table"].items()},
                             "uppers": {a: {k: t["outcomes"][k]["upper"] for k in ("Y_L_f1", "Y_L_f2", "Y_D", "Y_C")} for a, t in alt["arms_table"].items()},
                             "same_decision": bool(alt.get("switch") == dec.get("switch") and alt.get("pi") == dec.get("pi"))}
    if measured_band is not None:
        bo = MB.bands_of(measured_band)
        alt = look_decision(rows, members, a0, profiles, tau_table, zeta, day, budgets, n_nodes, eps, t_bar, b, l, l_dep, k7, k_rounds, completed_rounds, composition, one_sided,
                            coupling_mode, None, None, bo)
        reg_band = float(bands[0])
        dec["measured_band"] = {"deciding": False, "reference_t_bar": t_bar, "eta_dep_note": "eta_dep stays the one of the planned t_bar", **sensitivity_summary(alt),
                                "measurement": measured_band, "reference_band": reg_band, "bands": {k: bo.get(k, reg_band) for k in OUTCOMES},
                                "reference_band_kept_for": [k for k in OUTCOMES if k not in bo],
                                "uppers": {a: {k: t["outcomes"][k]["upper"] for k in BUDGET_KEYS} for a, t in alt["arms_table"].items()},
                                "same_decision": bool(alt.get("switch") == dec.get("switch") and alt.get("pi") == dec.get("pi"))}
    dec.update({"day": day, "look_rounds": k_look, "complete_rounds": full_rounds, "level_L": L, "tau_max": tau_max, "eta_dep": eta,
                "arms_table": arms_tab, "stop_campaign": bool(dec.get("switch")), "parameters": {"t_bar": t_bar, "b": b, "l": l, "l_dep": l_dep, "eps": eps, "k7": k7,
                                                                                                    "n_nodes": n_nodes, "delta": DELTA, "balanced": bool(composition), "one_sided": bool(one_sided),
                                                                                                    "composition": list(composition) if composition else None}})
    return dec


def sensitivity_summary(d: Dict[str, Any]) -> Dict[str, Any]:
    bud, inc = d.get("budgets") or {}, d.get("incumbent_upper_bounds") or {}
    return {"switch": d.get("switch"), "reason": d.get("reason"), "pi": d.get("pi"), "lp_status": d.get("lp_status"), "quality_cost_bound": d.get("quality_cost_bound"),
            "eta_dep": d.get("eta_dep"), "branch": d.get("branch"), "applied": (d.get("coupling") or {}).get("applied"), "slack": d.get("slack"),
            "incumbent_margin": {k: bud[k] - v for k, v in inc.items() if k in bud}, "nonincumbent_mass": d.get("nonincumbent_mass"), "gate_value": d.get("gate_value")}
