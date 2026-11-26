from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "algorithms"))

from ciara import looks as LK
from ciara.constants import BUDGET_KEYS, BUDGETS, OUTCOMES
from ciara.transient import TauTable

NUM = set(OUTCOMES) | {"block_id"}
ROLE = {"f1prio": "priority arm", "fifo": "FIFO arm"}

PUBLISHED_WEIGHTS = (0.106, 0.548, 0.346)
PUBLISHED_COST_BOUND = 0.0842032261
PUBLISHED_ETA_DEP = 0.0079305463


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            q = dict(r)
            for k in NUM:
                v = q.get(k)
                if v not in (None, ""):
                    q[k] = int(v) if k == "block_id" else float(v)
                elif k != "block_id":
                    q[k] = None
            rows.append(q)
    return rows


def profiles_of(sel, rows, k_rounds):
    members = sel["S"]
    prof = {m: list(sel["arms"][m].get("profiles") or ([sel["arms"][m]["profile"]] if sel["arms"][m].get("profile") else [])) for m in members}
    for k, v in (sel.get("regime_profiles") or {}).items():
        if k in prof and v:
            prof[k] = list(v)
    for r in rows:
        if r["arm_id"] in prof and sel["arms"][r["arm_id"]].get("kind") == "incumbent" and r.get("profiles_in_window") and r["block_id"] < k_rounds:
            prof[r["arm_id"]] = sorted(set(prof[r["arm_id"]]) | set(str(r["profiles_in_window"]).split("+")))
    return prof


def decide(rows, sel, design, tau, day):
    l = int(design["block_epochs"])
    k = LK.look_blocks(day, l)
    return LK.look_decision(rows, sel["S"], sel["a0"], profiles_of(sel, rows, k), tau, sel.get("zeta") or {}, day, k_rounds=k,
                            composition=design["composition"] if design.get("balanced") else None, one_sided=bool(design.get("one_sided")),
                            coupling_mode="a5pp_gated", b=int(design["burn_in_epochs"]), l=l, l_dep=int(design["l_dep_epochs"]))


def role(sel, arm):
    a = sel["arms"][arm]
    return "incumbent" if a.get("kind") == "incumbent" else ROLE.get(a.get("admission"), "alternative")


def fmt(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def report(dec, sel):
    members = sel["S"]
    a0 = sel["a0"]
    deploy = bool(dec["switch"])
    print(f"Look at day {dec['day']}: {dec['look_rounds']} rounds ({dec['complete_rounds']} complete in every node)")
    print(f"  decision: {'DEPLOY the certified policy' if deploy else 'RETAIN the incumbent ' + a0}")
    print(f"  branch: {dec['branch']} (incumbent certified feasible: {'yes' if dec['incumbent_feasible'] else 'no'}); program: {dec['lp_status']}")
    print(f"  reason: {dec['reason']}")
    print(f"  tau = {dec['tau_max']:g} epochs, eta_dep = {dec['eta_dep']:.6f}, level L = {dec['level_L']:.3f}")
    print("  policy weights:")
    pi = dec.get("pi") if deploy else {a: (1.0 if a == a0 else 0.0) for a in members}
    for a in members:
        print(f"    {pi[a]:.3f}  {a} ({role(sel, a)})")
    bounds = dec["upper_bounds"] if deploy else dec["incumbent_upper_bounds"]
    whose = "certified policy" if deploy else "incumbent"
    print(f"  certified upper bound of the {whose} against the budgets (mean + radius + eta_dep + zeta):")
    print(f"    {'outcome':<8} {'bound':>8} {'budget':>8} {'margin':>9}")
    for k in BUDGET_KEYS:
        b = dec["budgets"][k]
        print(f"    {k:<8} {bounds[k]:8.4f} {b:8.2f} {b - bounds[k]:9.4f}{'  exceeds' if bounds[k] > b + 1e-9 else ''}")
    print("  certified upper bound of each arm (mean + radius + eta_dep + zeta):")
    print("    " + f"{'arm':<14}" + "".join(f"{k:>9}" for k in BUDGET_KEYS))
    zeta = dec.get("zeta") or {}
    per_arm = {}
    for a in members:
        o = dec["arms_table"][a]["outcomes"]
        per_arm[a] = {k: o[k]["upper"] + dec["eta_dep"] + zeta.get(k, 0.0) for k in BUDGET_KEYS}
        print("    " + f"{role(sel, a):<14}" + "".join(f"{per_arm[a][k]:9.4f}" for k in BUDGET_KEYS))
    print("    " + f"{'budget':<14}" + "".join(f"{dec['budgets'][k]:9.2f}" for k in BUDGET_KEYS))
    if not deploy and any(abs(per_arm[a0][k] - bounds[k]) > 1e-9 for k in BUDGET_KEYS):
        print("  note: the incumbent as a constant policy keeps the coupling slack zeta_hat on every row when its escalated share exceeds that of the design law (condition A5'' does not\n"
              "  cover it), so its bound above differs from its entry in the per-arm table.")
    if deploy:
        print(f"  cost bound: J_Q({a0}) - J_Q(policy) <= {dec['quality_cost_bound']:.4f}")
    else:
        print("  cost bound: none (no certified policy)")
    print()


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


def check(decs, sel, stored):
    bad = []

    def need(cond, msg):
        if not cond:
            bad.append(msg)
    members = sel["S"]
    d4, d7 = decs.get(4), decs.get(7)
    if d4 is not None:
        need(d4["look_rounds"] == 230, "day 4 is not the look at 230 rounds")
        need(not d4["switch"], "day 4: a policy is certified, the published result is no certifiable policy")
        need(d4["lp_status"] == "infeasible" and d4.get("pi") is None, "day 4: the program is not infeasible")
        need(d4["branch"] == "budget_compliance" and not d4["incumbent_feasible"], "day 4: the incumbent is not marked as not certified feasible")
        need(close(d4["eta_dep"], PUBLISHED_ETA_DEP, 1e-9), f"day 4: eta_dep {d4['eta_dep']} differs from {PUBLISHED_ETA_DEP}")
    if d7 is not None:
        need(d7["look_rounds"] == 403, "day 7 is not the look at 403 rounds")
        need(d7["switch"] and d7["lp_status"] == "optimal", "day 7: no certified policy")
        if d7.get("pi"):
            w = tuple(round(d7["pi"][a], 3) for a in members)
            need(w == PUBLISHED_WEIGHTS, f"day 7: weights {w} differ from the published {PUBLISHED_WEIGHTS}")
            need(close(d7["quality_cost_bound"], PUBLISHED_COST_BOUND, 1e-9), f"day 7: cost bound {d7['quality_cost_bound']} differs from {PUBLISHED_COST_BOUND}")
            need(round(d7["quality_cost_bound"], 3) == 0.084, "day 7: the cost bound does not round to the published 0.084")
            need(all(d7["upper_bounds"][k] <= d7["budgets"][k] + 1e-9 for k in BUDGET_KEYS), "day 7: a certified bound exceeds its budget")
            need(not d7["incumbent_feasible"] and d7["incumbent_violations"] and "Y_C" in d7["incumbent_violations"], "day 7: the incumbent does not exceed the cloud share")
            need(close(d7["eta_dep"], PUBLISHED_ETA_DEP, 1e-9), f"day 7: eta_dep {d7['eta_dep']} differs from {PUBLISHED_ETA_DEP}")
        if stored is not None:
            need(d7["switch"] == stored["switch"], "day 7: decision differs from the stored look")
            if d7.get("pi") and stored.get("pi"):
                need(all(close(d7["pi"][a], stored["pi"][a]) for a in members), "day 7: weights differ from the stored look")
                need(close(d7["quality_cost_bound"], stored["quality_cost_bound"]), "day 7: cost bound differs from the stored look")
                need(all(close(d7["upper_bounds"][k], stored["upper_bounds"][k]) for k in BUDGET_KEYS), "day 7: certified bounds differ from the stored look")
                need(all(close(d7["slack"][k], stored["slack"][k]) for k in BUDGET_KEYS), "day 7: budget margins differ from the stored look")
            for a in members:
                for k in OUTCOMES:
                    t, s = d7["arms_table"][a]["outcomes"][k], stored["arms_table"][a]["outcomes"][k]
                    need(close(t["centre"], s["centre"]) and close(t["radius"], s["radius"]), f"day 7: center or radius of {a}, {k} differs from the stored look")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate the certificate of the certification campaign at its looks.")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "campaign_node_blocks.csv"), help="node-blocks of the campaign (CSV)")
    ap.add_argument("--inputs", default=os.path.join(ROOT, "data", "inputs", "campaign"), help="folder with selection.json, campaign-design.json, switch-transient.json, look-day7.json")
    ap.add_argument("--days", type=int, nargs="+", default=[4, 7], help="look days to evaluate (default: 4 7)")
    ap.add_argument("--check", action="store_true", help="assert the published values")
    a = ap.parse_args(argv)

    sel = read_json(os.path.join(a.inputs, "selection.json"))
    design = read_json(os.path.join(a.inputs, "campaign-design.json"))
    tau = TauTable.load(os.path.join(a.inputs, "switch-transient.json"))
    stored_path = os.path.join(a.inputs, "look-day7.json")
    stored = read_json(stored_path) if os.path.exists(stored_path) else None
    rows = read_rows(a.data)
    print(f"{len(rows)} node-blocks read from {os.path.basename(a.data)}; certified set: " + ", ".join(sel["S"]) + "\n")
    decs = {}
    for day in a.days:
        decs[day] = decide(rows, sel, design, tau, day)
        report(decs[day], sel)
    if a.check:
        bad = check(decs, sel, stored)
        if bad:
            print("CHECK FAILED:")
            for m in bad:
                print("  " + m)
            return 1
        print("check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
