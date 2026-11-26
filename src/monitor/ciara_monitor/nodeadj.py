from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ciara.constants import BUDGETS, OUTCOMES, finite

from .fileio import now_cst, read_json, read_jsonl, write_json, write_text

from . import nodecheck as NC
from . import trajectory as TR
from .sufficiency import ols_cluster

PI_TOL = 1e-6
RHO_TOL = 1e-12
INF_RADIUS = 1.0e5


def _design(rows: Sequence[Dict[str, Any]], arms: Sequence[str], nodes: Sequence[str], adjust: bool) -> np.ndarray:
    X = np.zeros((len(rows), len(arms) + (len(nodes) - 1 if adjust else 0)))
    for i, r in enumerate(rows):
        X[i, list(arms).index(r["arm_id"])] = 1.0
        if adjust and r["node"] != nodes[0]:
            X[i, len(arms) + list(nodes).index(r["node"]) - 1] = 1.0
    return X


def fit_outcome(rows: Sequence[Dict[str, Any]], members: Sequence[str], a0: str, outcome: str, adjust: bool = True) -> Dict[str, Any]:
    rr = [r for r in rows if finite(r.get(outcome))]
    arms = [a for a in members if any(r["arm_id"] == a for r in rr)]
    nodes = sorted({r["node"] for r in rr})
    out: Dict[str, Any] = {"ok": False, "n": len(rr), "rounds": len({int(r["block_id"]) for r in rr}), "adjust": adjust, "levels": {}, "contrasts": {}}
    if len(arms) < len(members) or a0 not in arms or not nodes:
        return out
    y = np.array([float(r[outcome]) for r in rr])
    fit = ols_cluster(_design(rr, arms, nodes, adjust), y, [int(r["block_id"]) for r in rr])
    if fit["cov"] is None:
        return out
    beta, V = fit["beta"], fit["cov"]
    p = len(beta)

    def comb(w: np.ndarray) -> Dict[str, float]:
        return {"value": float(w @ beta), "se": math.sqrt(max(float(w @ V @ w), 0.0))}

    lv: Dict[str, np.ndarray] = {}
    for i, a in enumerate(arms):
        w = np.zeros(p)
        w[i] = 1.0
        if adjust and len(nodes) > 1:
            w[len(arms):] = 1.0 / len(nodes)
        lv[a] = w
        out["levels"][a] = comb(w)
    for a in arms:
        if a != a0:
            c = comb(lv[a] - lv[a0])
            c["t"] = c["value"] / c["se"] if c["se"] > 0 else None
            out["contrasts"][a] = c
    out.update({"ok": True, "G": fit["G"], "nodes": nodes})
    return out


def adjusted_outcomes(rows: Sequence[Dict[str, Any]], members: Sequence[str], a0: str, k: Optional[int] = None) -> Dict[str, Any]:
    wr = NC.window_rows(rows, members, k)
    return {o: {"adjusted": fit_outcome(wr, members, a0, o, True), "arm_only": fit_outcome(wr, members, a0, o, False)} for o in OUTCOMES}


def adjusted_decision(look: Dict[str, Any], levels: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    dec = copy.deepcopy(look)
    for a, t in dec["arms_table"].items():
        for o, c in t["outcomes"].items():
            if a in levels.get(o, {}):
                c["centre"] = levels[o][a]
                c["lower"], c["upper"] = c["centre"] - c["radius"], c["centre"] + c["radius"]
    return dec


def gated_program(dec: Dict[str, Any], members: Sequence[str], a0: str, zeta: Dict[str, float], budgets: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    budgets = dict(BUDGETS if budgets is None else budgets)
    d = TR.program_at(dec, members, a0, "a5pp", budgets, zeta)
    applied = "a5pp"
    rho = (d.get("share_constraint") or {}).get("residual_per_gamma")
    if d.get("lp_status") == "optimal" and rho is not None and rho > RHO_TOL:
        d = TR.program_at(dec, members, a0, "a5pp_strict", budgets, zeta)
        applied = "a5pp_strict"
    return {"switch": bool(d.get("switch")), "lp_status": d.get("lp_status"), "pi": d.get("pi"), "branch": d.get("branch"), "applied": applied, "rho": rho,
            "reason": d.get("reason"), "quality_cost_bound": d.get("quality_cost_bound"), "slack": d.get("slack"), "gate_value": d.get("gate_value")}


def program_record(dec: Dict[str, Any], members: Sequence[str], a0: str, zeta: Dict[str, float], budgets: Dict[str, float]) -> Dict[str, Any]:
    g = gated_program(dec, members, a0, zeta, budgets)
    um = {m: TR.uniform_margin(dec, members, a0, m, zeta, budgets) for m in ("a5pp", "a5pp_strict")}
    g["uniform_margin"] = {m: {kk: v.get(kk) for kk in ("t_star", "binding_rows", "reference_feasible")} for m, v in um.items()}
    return g


def same_decision(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if bool(a.get("switch")) != bool(b.get("switch")):
        return False
    pa, pb = a.get("pi"), b.get("pi")
    if not pa or not pb:
        return pa == pb
    return all(abs(pa[x] - pb.get(x, 0.0)) <= PI_TOL for x in pa)


def node_adjusted(rows: Sequence[Dict[str, Any]], look: Dict[str, Any], members: Sequence[str], a0: str, k: Optional[int] = None, zeta: Optional[Dict[str, float]] = None,
                  mock: bool = False, rounds_label: Optional[int] = None) -> Dict[str, Any]:
    zeta = dict(zeta if zeta is not None else ((look.get("coupling") or {}).get("zeta_reference") or look.get("zeta") or {}))
    budgets = dict(look.get("budgets") or BUDGETS)
    fits = adjusted_outcomes(rows, members, a0, k)
    tab = look["arms_table"]
    res: Dict[str, Any] = {"written": now_cst(), "members": list(members), "a0": a0, "rounds": k if k is not None else rounds_label, "_mock": bool(mock), "outcomes": {},
                           "look_day": look.get("day"), "look_rounds": look.get("look_rounds"), "coupling_applied": (look.get("coupling") or {}).get("applied")}
    levels: Dict[str, Dict[str, float]] = {}
    failed: List[str] = []
    for o in OUTCOMES:
        fa, fo = fits[o]["adjusted"], fits[o]["arm_only"]
        if not fa["ok"]:
            failed.append(o)
        rec: Dict[str, Any] = {"n": fa["n"], "rounds": fa["rounds"], "ok": fa["ok"], "arms": {}, "contrasts": {}}
        if fa["ok"]:
            levels[o] = {}
            for a, v in fa["levels"].items():
                reg = tab[a]["outcomes"][o]
                levels[o][a] = v["value"]
                rec["arms"][a] = {"adjusted": v["value"], "se": v["se"], "arm_only": fo["levels"][a]["value"] if fo["ok"] else None, "look_centre": reg["centre"], "radius": reg["radius"],
                                  "shift": v["value"] - reg["centre"], "shift_in_se": (v["value"] - reg["centre"]) / v["se"] if v["se"] > 0 else None,
                                  "shift_in_radius": (v["value"] - reg["centre"]) / reg["radius"] if 0 < reg["radius"] < INF_RADIUS else None}
            for a, c in fa["contrasts"].items():
                r0, r1 = tab[a0]["outcomes"][o], tab[a]["outcomes"][o]
                d_reg = r1["centre"] - r0["centre"]
                half = r1["radius"] + r0["radius"]
                co = fo["contrasts"].get(a) if fo["ok"] else None
                rec["contrasts"][a] = {"adjusted": c["value"], "se": c["se"], "t": c["t"], "arm_only": co["value"] if co else None, "se_arm_only": co["se"] if co else None,
                                       "look_contrast": d_reg, "shift": c["value"] - d_reg, "shift_in_se": (c["value"] - d_reg) / c["se"] if c["se"] > 0 else None,
                                       "half_width": half if half < INF_RADIUS else None,
                                       "reference_interval": [d_reg - half, d_reg + half] if half < INF_RADIUS else None,
                                       "adjusted_interval": [c["value"] - half, c["value"] + half] if half < INF_RADIUS else None,
                                       "excludes_zero_reference": (abs(d_reg) > half) if half < INF_RADIUS else False,
                                       "excludes_zero_adjusted": (abs(c["value"]) > half) if half < INF_RADIUS else False}
        res["outcomes"][o] = rec
    res["failed_outcomes"] = failed
    if failed:
        res["program"] = {"computed": False, "reason": "node-adjusted fit failed (rank deficiency or fewer than three rounds) for " + ", ".join(failed)}
    else:
        reg = program_record(look, members, a0, zeta, budgets)
        adj = program_record(adjusted_decision(look, levels), members, a0, zeta, budgets)
        stored = {"switch": bool(look.get("switch")), "pi": look.get("pi")}
        res["program"] = {"computed": True, "reference": reg, "adjusted": adj, "same_decision": same_decision(reg, adj),
                          "reference_reproduces_look": same_decision(reg, stored),
                          "max_abs_weight_change": max((abs((reg.get("pi") or {}).get(a, 0.0) - (adj.get("pi") or {}).get(a, 0.0)) for a in members), default=0.0) if reg.get("pi") and adj.get("pi") else None,
                          "t_star_change": _diff(adj["uniform_margin"]["a5pp"].get("t_star"), reg["uniform_margin"]["a5pp"].get("t_star"))}
    res["flags"] = flags(res)
    return res


def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else a - b


def flags(res: Dict[str, Any]) -> Dict[str, Any]:
    worst_c = (None, None, 0.0)
    worst_k = (None, None, 0.0)
    for o, v in res["outcomes"].items():
        for a, c in v["arms"].items():
            s = abs(c["shift_in_se"] or 0.0)
            if s > worst_c[2]:
                worst_c = (o, a, s)
        for a, c in v["contrasts"].items():
            s = abs(c["shift_in_se"] or 0.0)
            if s > worst_k[2]:
                worst_k = (o, a, s)
    out: Dict[str, Any] = {"rounds": res["rounds"], "mock": res["_mock"], "failed_outcomes": res["failed_outcomes"], "largest_center_shift_in_se": {"outcome": worst_c[0], "arm": worst_c[1], "se": worst_c[2]},
                           "largest_contrast_shift_in_se": {"outcome": worst_k[0], "arm": worst_k[1], "se": worst_k[2]},
                           "binding_contrast_shift_in_se": {o: {a: c["shift_in_se"] for a, c in res["outcomes"][o]["contrasts"].items()} for o in ("Y_C", "Y_L_f1") if res["outcomes"][o]["ok"]}}
    pr = res["program"]
    if pr.get("computed"):
        r, a = pr["reference"], pr["adjusted"]
        out.update({"program_computed": True, "reference_switch": r["switch"], "adjusted_switch": a["switch"], "same_decision": pr["same_decision"], "reference_pi": r.get("pi"), "adjusted_pi": a.get("pi"),
                    "reference_t_star": r["uniform_margin"]["a5pp"].get("t_star"), "adjusted_t_star": a["uniform_margin"]["a5pp"].get("t_star"),
                    "adjusted_binding": a["uniform_margin"]["a5pp"].get("binding_rows"), "reference_reproduces_look": pr["reference_reproduces_look"]})
    else:
        out.update({"program_computed": False, "reason": pr.get("reason")})
    return out


def _f(x: Any, d: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def _pi(p: Optional[Dict[str, float]]) -> str:
    return "incumbent kept (no policy)" if not p else ", ".join(f"{x:.3f} {a.replace('|', chr(92) + '|')}" for a, x in p.items() if x > 1e-9)


def render_text(fl: Dict[str, Any]) -> str:
    if not fl:
        return "Node-adjusted sensitivity: not computed (no result in this run)."
    if fl.get("error"):
        return f"Node-adjusted sensitivity: not computed ({fl['error']})."
    lc, lk = fl["largest_center_shift_in_se"], fl["largest_contrast_shift_in_se"]
    head = f"Node-adjusted sensitivity ({fl['rounds']} rounds{', MOCK' if fl.get('mock') else ''}): "
    if not fl.get("program_computed"):
        return head + f"program not computed ({fl.get('reason')})."
    bind = {o: {a: round(s, 2) for a, s in v.items() if s is not None} for o, v in (fl.get("binding_contrast_shift_in_se") or {}).items()}
    return (head + f"reference switch {fl['reference_switch']} ({_pi(fl['reference_pi'])}), node-adjusted switch {fl['adjusted_switch']} ({_pi(fl['adjusted_pi'])}), same decision {fl['same_decision']}; "
            f"uniform margin t* {_f(fl['reference_t_star'])} reference, {_f(fl['adjusted_t_star'])} adjusted (binding {fl['adjusted_binding']}); "
            f"largest shift of a center {lc['se']:.2f} standard errors ({lc['outcome']}, {lc['arm']}), of a contrast {lk['se']:.2f} ({lk['outcome']}, {lk['arm']}); "
            f"binding contrasts {bind}.")


def render_md(res: Dict[str, Any]) -> str:
    L = [f"# Node-adjusted sensitivity at the stop ({res['rounds']} rounds)\n"]
    if res["_mock"]:
        L.append("**MOCK:** The data are the rounds of a campaign that has not ended, not the stop of the campaign. The numbers describe the procedure and are not results.\n")
    L.append(f"Written {res['written']}. Look day {res.get('look_day')}, rule `{res.get('coupling_applied')}` as reference. Per outcome, the window means of the node-blocks of the members of S are fitted with arm and node "
             "fixed effects (round-clustered standard errors); the level of an arm is its effect plus the mean node effect. The look estimator, the radii and the rule are those of the look; "
             "only the centers differ. Reported, not used to decide.\n")
    pr = res["program"]
    L.append("## Certified program on the node-adjusted centers\n")
    if not pr.get("computed"):
        L.append(f"Not computed: {pr.get('reason')}.\n")
    else:
        r, a = pr["reference"], pr["adjusted"]
        L.append(f"**Same decision:** {pr['same_decision']} (the reference program reproduces the stored decision of the look: {pr['reference_reproduces_look']}).\n")
        L.append("| quantity | reference centers | node-adjusted centers |\n|---|---|---|")
        L.append(f"| decision | {'switch' if r['switch'] else 'incumbent kept'} ({r['reason']}) | {'switch' if a['switch'] else 'incumbent kept'} ({a['reason']}) |")
        L.append(f"| program applied (rho) | {r['applied']} ({_f(r['rho'], 5)}) | {a['applied']} ({_f(a['rho'], 5)}) |")
        L.append(f"| weights | {_pi(r['pi'])} | {_pi(a['pi'])} |")
        L.append(f"| certified quality cost bound | {_f(r['quality_cost_bound'])} | {_f(a['quality_cost_bound'])} |")
        for m in ("a5pp", "a5pp_strict"):
            L.append(f"| uniform margin t* of `{m}` (binding rows) | {_f(r['uniform_margin'][m].get('t_star'))} ({', '.join(r['uniform_margin'][m].get('binding_rows') or []) or 'none'}) | "
                     f"{_f(a['uniform_margin'][m].get('t_star'))} ({', '.join(a['uniform_margin'][m].get('binding_rows') or []) or 'none'}) |")
        if r.get("slack") and a.get("slack"):
            for k in r["slack"]:
                L.append(f"| slack of the row {k} | {_f(r['slack'][k])} | {_f((a['slack'] or {}).get(k))} |")
        L.append(f"\nLargest change of a weight: {_f(pr['max_abs_weight_change'])}; change of t*: {_f(pr['t_star_change'])}.\n")
    L.append("## Certified contrasts to the incumbent\n")
    L.append("Contrast d of each alternative to the incumbent with the certified interval d +- (r_a + r_a0) of the reference radii; shift = node-adjusted minus reference contrast, in clustered standard errors of the adjusted contrast.\n")
    L.append("| outcome | arm | reference d | adjusted d (s.e.) | arm effects only | shift (s.e.) | reference interval | adjusted interval |\n|---|---|---|---|---|---|---|---|")
    for o in list(NC.BINDING) + [x for x in OUTCOMES if x not in NC.BINDING]:
        v = res["outcomes"][o]
        for a, c in v["contrasts"].items():
            ri, ai = c["reference_interval"], c["adjusted_interval"]
            L.append(f"| {o}{' (binding)' if o in NC.BINDING else ''} | {a.replace('|', chr(92) + '|')} | {_f(c['look_contrast'])} | {_f(c['adjusted'])} ({_f(c['se'])}) | {_f(c['arm_only'])} | {_f(c['shift_in_se'], 2)} | "
                     f"{'n/a' if not ri else f'[{ri[0]:.4f}, {ri[1]:.4f}]'} | {'n/a' if not ai else f'[{ai[0]:.4f}, {ai[1]:.4f}]'} |")
    L.append("\n## Arm levels\n")
    L.append("| outcome | arm | reference center | node-adjusted level (s.e.) | shift (s.e.) | shift (radius) |\n|---|---|---|---|---|---|")
    for o in list(NC.BINDING) + [x for x in OUTCOMES if x not in NC.BINDING]:
        for a, c in res["outcomes"][o]["arms"].items():
            L.append(f"| {o} | {a.replace('|', chr(92) + '|')} | {_f(c['look_centre'])} | {_f(c['adjusted'])} ({_f(c['se'])}) | {_f(c['shift_in_se'], 2)} | {_f(c['shift_in_radius'], 3)} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Node-adjusted sensitivity of the certificate (Section VI-C). Analysis only: the look estimator, the radii and the rule stay as in the look.')
    ap.add_argument("--look", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=None, help="rounds of the look (default: complete_rounds of the look file)")
    ap.add_argument("--final", action="store_true", help="the data are the stop of the campaign (no MOCK stamp)")
    a = ap.parse_args(argv)
    look, sel = read_json(a.look), read_json(a.selection)
    k = a.rounds if a.rounds is not None else int(look.get("complete_rounds") or look.get("look_rounds"))
    res = node_adjusted(read_jsonl(a.rows), look, sel["S"], sel["a0"], k, mock=not a.final)
    os.makedirs(a.out, exist_ok=True)
    write_json(os.path.join(a.out, "NODEADJ.json"), res)
    write_text(os.path.join(a.out, "NODEADJ.md"), render_md(res))
    print(json.dumps(res["flags"], default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
