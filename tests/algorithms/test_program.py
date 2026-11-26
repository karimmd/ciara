import random
import unittest

import numpy as np

from ciara import program as C
from ciara.constants import BUDGET_KEYS, BUDGETS_TIGHT, OUTCOMES
from ref.simplex import solve_distribution_lp


def table(q, ex=None):
    mu = {"Y_Q": list(q)}
    r = {"Y_Q": [0.01] * len(q)}
    for k in BUDGET_KEYS:
        mu[k] = [0.01] * len(q)
        r[k] = [0.005] * len(q)
    mu["Y_C"] = [0.1] * len(q)
    if ex:
        mu["Y_L_f2"] = list(ex)
    return mu, r


class LP(unittest.TestCase):
    def test_matches_simplex(self):
        rng = random.Random(3)
        for _ in range(200):
            n = rng.choice([2, 3])
            c = [rng.random() for _ in range(n)]
            m = rng.randint(2, 6)
            G = [[rng.random() for _ in range(n)] for _ in range(m)]
            h = [rng.uniform(0.3, 0.9) for _ in range(m)]
            ref = solve_distribution_lp(c, G, h)
            got = C.solve_simplex_lp(c, G, h)
            self.assertEqual(ref.status, got["status"])
            if ref.status == "optimal":
                self.assertAlmostEqual(ref.value, got["value"], places=7)

    def test_single_arm(self):
        r = C.solve_simplex_lp([0.4], [[0.2]], [0.5])
        self.assertEqual(r["status"], "optimal")
        self.assertAlmostEqual(r["p"][0], 1.0)

    def test_infeasible(self):
        self.assertEqual(C.solve_simplex_lp([1, 1], [[0.9, 0.9]], [0.5])["status"], "infeasible")


class Gate(unittest.TestCase):
    def test_switch_when_alternatives_are_better_and_feasible(self):
        mu, r = table([0.60, 0.70, 0.68])
        for k in r:
            r[k] = [0.002] * 3
        out = C.certified_policy(["a0", "a1", "a2"], mu, r, {}, 0.006, "a0")
        self.assertTrue(out["switch"], out)
        self.assertAlmostEqual(sum(out["pi"].values()), 1.0)
        self.assertGreater(out["pi"]["a1"], 0.9)

    def test_keep_incumbent_when_radius_swamps_gain(self):
        mu, r = table([0.60, 0.61, 0.605])
        r["Y_Q"] = [0.05] * 3
        out = C.certified_policy(["a0", "a1", "a2"], mu, r, {}, 0.006, "a0")
        self.assertFalse(out["switch"])

    def test_budget_binds(self):
        mu, r = table([0.60, 0.80, 0.62], ex=[0.01, 0.045, 0.01])
        out = C.certified_policy(["a0", "a1", "a2"], mu, r, {}, 0.006, "a0", BUDGETS_TIGHT)
        self.assertEqual(out["lp_status"], "optimal")
        self.assertLessEqual(out["upper_bounds"]["Y_L_f2"], 0.05 + 1e-9)
        self.assertAlmostEqual(out["pi"]["a1"], 0.029 / 0.035, places=6)
        self.assertLess(out["pi"]["a1"], 1.0)

    def test_infeasible_keeps_incumbent(self):
        mu, r = table([0.6, 0.7, 0.7], ex=[0.06, 0.06, 0.06])
        out = C.certified_policy(["a0", "a1", "a2"], mu, r, {}, 0.006, "a0", BUDGETS_TIGHT)
        self.assertFalse(out["switch"])
        self.assertEqual(out["lp_status"], "infeasible")
        self.assertEqual(out["branch"], "budget_compliance")

    def test_zeta_enters_the_gate_twice(self):
        mu, r = table([0.60, 0.66, 0.60])
        for k in r:
            r[k] = [0.001] * 3
        no_zeta = C.certified_policy(["a0", "a1", "a2"], mu, r, {}, 0.006, "a0")
        with_zeta = C.certified_policy(["a0", "a1", "a2"], mu, r, {"Y_Q": 0.03}, 0.006, "a0")
        self.assertTrue(no_zeta["switch"])
        self.assertFalse(with_zeta["switch"])
        self.assertAlmostEqual(no_zeta["gate_value"] - with_zeta["gate_value"], 0.06, places=6)

    def test_min_mass(self):
        mu, r = table([0.60, 0.60, 0.60])
        r["Y_Q"] = [0.0] * 3
        out = C.certified_policy(["a0", "a1", "a2"], mu, r, {}, 0.0, "a0")
        self.assertFalse(out["switch"])


class Branches(unittest.TestCase):
    ARMS = ["a0", "a1", "a2"]

    def incumbent_exceeding(self, a0_c=0.40, q=(0.60, 0.62, 0.58), rad=0.002):
        mu, r = table(list(q))
        mu["Y_C"] = [a0_c, 0.10, 0.05]
        for k in r:
            r[k] = [rad] * 3
        return mu, r

    def test_gate_branch_when_the_incumbent_is_certified_feasible(self):
        mu, r = table([0.60, 0.70, 0.68])
        for k in r:
            r[k] = [0.002] * 3
        out = C.certified_policy(self.ARMS, mu, r, {}, 0.006, "a0")
        self.assertEqual(out["branch"], "gate")
        self.assertTrue(out["incumbent_feasible"])
        self.assertTrue(out["gate_applies"])
        self.assertTrue(out["switch"])
        self.assertTrue(out["gate_pass"])
        mu, r = table([0.60, 0.601, 0.60])
        r["Y_Q"] = [0.05] * 3
        out = C.certified_policy(self.ARMS, mu, r, {}, 0.006, "a0")
        self.assertEqual(out["branch"], "gate")
        self.assertFalse(out["switch"])
        self.assertEqual(out["lp_status"], "optimal")

    def test_budget_compliance_branch_deploys_the_lp_solution_without_the_gate(self):
        mu, r = self.incumbent_exceeding(q=(0.60, 0.50, 0.40))
        out = C.certified_policy(self.ARMS, mu, r, {}, 0.006, "a0")
        self.assertEqual(out["branch"], "budget_compliance")
        self.assertFalse(out["incumbent_feasible"])
        self.assertIn("Y_C", out["incumbent_violations"])
        self.assertAlmostEqual(out["incumbent_upper_bounds"]["Y_C"], 0.40 + 0.002 + 0.006, places=9)
        self.assertEqual(out["lp_status"], "optimal")
        self.assertTrue(out["switch"], out)
        self.assertFalse(out["gate_applies"])
        self.assertIsNone(out["gate_pass"])
        self.assertLess(out["gate_value"], 0.0)
        self.assertLess(out["pi"]["a0"], 1.0)
        for k, ub in out["upper_bounds"].items():
            self.assertLessEqual(ub, out["budgets"][k] + 1e-9)
        self.assertLessEqual(out["upper_bounds"]["Y_C"], 0.30 + 1e-9)
        self.assertAlmostEqual(out["pi"]["a0"], (0.30 - 0.006 - 0.10 - 0.002) / (0.40 - 0.10), delta=0.2)

    def test_quality_cost_bound_formula(self):
        mu, r = self.incumbent_exceeding(q=(0.60, 0.62, 0.58), rad=0.004)
        zeta = {"Y_Q": 0.003}
        out = C.certified_policy(self.ARMS, mu, r, zeta, 0.006, "a0")
        self.assertEqual(out["branch"], "budget_compliance")
        pi = out["pi"]
        want = (0.60 + 0.004) - sum(pi[a] * (q - 0.004) for a, q in zip(self.ARMS, (0.60, 0.62, 0.58))) + 0.006 + 2 * 0.003
        self.assertAlmostEqual(out["quality_cost_bound"], want, places=9)
        self.assertIn("quality_cost_note", out)

    def test_quality_cost_bound_is_negative_for_a_certified_better_policy(self):
        mu, r = self.incumbent_exceeding(q=(0.50, 0.70, 0.69), rad=0.002)
        out = C.certified_policy(self.ARMS, mu, r, {}, 0.006, "a0")
        self.assertTrue(out["switch"])
        self.assertLess(out["quality_cost_bound"], 0.0)

    def test_infeasible_lp_keeps_the_incumbent_in_the_compliance_branch(self):
        mu, r = self.incumbent_exceeding()
        mu["Y_C"] = [0.40, 0.35, 0.33]
        out = C.certified_policy(self.ARMS, mu, r, {}, 0.006, "a0")
        self.assertEqual(out["branch"], "budget_compliance")
        self.assertEqual(out["lp_status"], "infeasible")
        self.assertFalse(out["switch"])
        self.assertIsNone(out["pi"])

    def test_the_incumbent_exceeding_a_radius_bound_only_is_not_feasible(self):
        mu, r = table([0.60, 0.62, 0.58])
        mu["Y_C"] = [0.29, 0.10, 0.05]
        for k in r:
            r[k] = [0.01] * 3
        out = C.certified_policy(self.ARMS, mu, r, {}, 0.006, "a0")
        self.assertEqual(out["branch"], "budget_compliance")
        mu["Y_C"] = [0.28, 0.10, 0.05]
        out = C.certified_policy(self.ARMS, mu, r, {}, 0.006, "a0")
        self.assertEqual(out["branch"], "gate")

    def test_zeta_enters_the_feasibility_check_of_the_incumbent(self):
        mu, r = table([0.60, 0.62, 0.58])
        mu["Y_C"] = [0.28, 0.10, 0.05]
        for k in r:
            r[k] = [0.005] * 3
        self.assertEqual(C.certified_policy(self.ARMS, mu, r, {}, 0.006, "a0")["branch"], "gate")
        self.assertEqual(C.certified_policy(self.ARMS, mu, r, {"Y_C": 0.02}, 0.006, "a0")["branch"], "budget_compliance")


ARMS = ["A0", "B", "C"]
Q = [0.5, 0.25, 0.25]


def prog_inputs(rc=None, yc=(0.6, 0.1, 0.2)):
    mu = {"Y_Q": [0.9, 0.5, 0.6], "Y_C": list(yc), "Y_L_f1": [0.05, 0.5, 0.4]}
    r = {k: [0.0, 0.0, 0.0] for k in mu}
    if rc is not None:
        r["Y_C"] = list(rc)
    return mu, r


BUD = {"Y_C": 0.5, "Y_L_f1": 0.9}


class HandSolvedPrograms(unittest.TestCase):
    def test_without_the_constraint(self):
        mu, r = prog_inputs()
        d = C.certified_policy(ARMS, mu, r, {}, 0.0, "A0", BUD)
        self.assertEqual(d["lp_status"], "optimal")
        self.assertAlmostEqual(d["pi"]["A0"], 0.75)
        self.assertAlmostEqual(d["pi"]["C"], 0.25)
        self.assertAlmostEqual(d["objective_lower_bound"], 0.825)
        self.assertNotIn("share_constraint", d)

    def test_with_the_constraint(self):
        mu, r = prog_inputs()
        d = C.certified_policy(ARMS, mu, r, {}, 0.0, "A0", BUD, monotone=True, design_q=Q)
        self.assertAlmostEqual(d["pi"]["A0"], 0.4375)
        self.assertAlmostEqual(d["pi"]["C"], 0.5625)
        self.assertAlmostEqual(d["pi"]["B"], 0.0)
        self.assertAlmostEqual(d["objective_lower_bound"], 0.73125)
        sc = d["share_constraint"]
        self.assertAlmostEqual(sc["design_share"], 0.375)
        self.assertAlmostEqual(sc["policy_share_upper"], 0.375)
        self.assertAlmostEqual(sc["margin"], 0.0, places=9)
        self.assertTrue(sc["holds"])
        self.assertFalse(sc["support_dominance_center"])
        self.assertAlmostEqual(sc["support_max_share_center"], 0.6)
        self.assertAlmostEqual(sc["residual_per_gamma"], 0.013125)

    def test_the_coupling_slack_against_a5pp(self):
        mu, r = prog_inputs()
        zeta = {"Y_C": 0.1, "Y_Q": 0.02, "Y_L_f1": 0.0}
        reg = C.certified_policy(ARMS, mu, r, zeta, 0.0, "A0", BUD)
        self.assertAlmostEqual(reg["pi"]["A0"], 0.5)
        self.assertAlmostEqual(reg["objective_lower_bound"], 0.75)
        mono = C.certified_policy(ARMS, mu, r, zeta, 0.0, "A0", BUD, monotone=True, design_q=Q)
        self.assertEqual(mono["zeta"], {"Y_C": 0.0, "Y_Q": 0.02, "Y_L_f1": 0.0})
        self.assertEqual(mono["zeta_reference"], zeta)
        self.assertAlmostEqual(mono["objective_lower_bound"], 0.73125)
        self.assertTrue(mono["monotone_coupling"])
        self.assertAlmostEqual(mono["slack"]["Y_C"], 0.5 - 0.375)

    def test_the_quality_slack_stays_in_the_gate_and_the_quality_cost(self):
        mu, r = prog_inputs()
        zeta = {"Y_C": 0.1, "Y_Q": 0.02}
        d = C.certified_policy(ARMS, mu, r, zeta, 0.0, "A0", BUD, one_sided_quality=True, monotone=True, design_q=Q)
        self.assertEqual(d["branch"], "budget_compliance")
        self.assertAlmostEqual(d["quality_cost_bound"], 0.5625 * 0.3 + 0.04)
        self.assertAlmostEqual(d["gate_value"], -0.5625 * 0.3 - 0.04)
        d2 = C.certified_policy(ARMS, mu, r, zeta, 0.0, "A0", BUD, one_sided_quality=True, monotone=True, design_q=Q)
        self.assertEqual(d["quality_cost_bound"], d2["quality_cost_bound"])
        free = C.certified_policy(ARMS, mu, r, {"Y_C": 0.1, "Y_Q": 0.0}, 0.0, "A0", BUD, one_sided_quality=True, monotone=True, design_q=Q)
        self.assertAlmostEqual(d["quality_cost_bound"] - free["quality_cost_bound"], 0.04)

    def test_monotone_needs_the_design_law(self):
        mu, r = prog_inputs()
        with self.assertRaises(ValueError):
            C.certified_policy(ARMS, mu, r, {}, 0.0, "A0", BUD, monotone=True)
        d = C.certified_policy(ARMS, mu, r, {}, 0.0, "A0", BUD, monotone=True, design_share=(0.375, 0.375))
        self.assertAlmostEqual(d["pi"]["A0"], 0.4375)

    def test_the_incumbent_keeps_the_coupling_slack_when_it_escalates_more_than_the_design_law(self):
        mu, r = prog_inputs()
        zeta = {"Y_C": 0.1}
        b = {"Y_C": 0.62, "Y_L_f1": 0.9}
        reg = C.certified_policy(ARMS, mu, r, zeta, 0.0, "A0", b)
        self.assertEqual(reg["branch"], "budget_compliance")
        mono = C.certified_policy(ARMS, mu, r, zeta, 0.0, "A0", b, monotone=True, design_q=Q)
        self.assertEqual(mono["branch"], "budget_compliance")
        self.assertAlmostEqual(mono["incumbent_upper_bounds"]["Y_C"], 0.7)
        mu2, r2 = prog_inputs(yc=(0.3, 0.1, 0.5))
        ok = C.certified_policy(ARMS, mu2, r2, zeta, 0.0, "A0", {"Y_C": 0.5, "Y_L_f1": 0.9}, monotone=True, design_q=Q)
        self.assertAlmostEqual(ok["incumbent_upper_bounds"]["Y_C"], 0.3)
        self.assertEqual(ok["branch"], "gate")

    def test_strict_support_forbids_arms_above_the_lower_design_share(self):
        mu, r = prog_inputs(rc=(0.05, 0.0, 0.0))
        d = C.certified_policy(ARMS, mu, r, {}, 0.0, "A0", BUD, monotone=True, design_q=Q, strict_support=True)
        sc = d["share_constraint"]
        self.assertAlmostEqual(sc["design_share_lower"], 0.35)
        self.assertEqual(sc["forbidden_arms"], ["A0"])
        self.assertAlmostEqual(d["pi"]["C"], 1.0)
        self.assertTrue(sc["support_dominance_strict"])
        loose = C.certified_policy(ARMS, mu, r, {}, 0.0, "A0", BUD, monotone=True, design_q=Q)
        self.assertFalse(loose["share_constraint"]["support_dominance_strict"])
        none = C.certified_policy(ARMS, mu, r, {}, 0.0, "A0", BUD, monotone=True, design_share=(0.01, 0.01), strict_support=True)
        self.assertEqual(none["lp_status"], "infeasible")
        with self.assertRaises(ValueError):
            C.certified_policy(ARMS, mu, r, {}, 0.0, "A0", BUD, strict_support=True)

    def test_the_counterexample_of_the_adversarial_check_in_numbers(self):
        mu = {"Y_Q": [0.1, 0.6, 0.65], "Y_C": [0.5, 0.0, 1.0], "Y_L_f1": [0.0, 0.0, 0.0]}
        r = {k: [0.0] * 3 for k in mu}
        d = C.certified_policy(ARMS, mu, r, {}, 0.0, "A0", {"Y_C": 0.5, "Y_L_f1": 0.9}, monotone=True, design_q=Q)
        self.assertAlmostEqual(d["pi"]["B"], 0.5)
        self.assertAlmostEqual(d["pi"]["C"], 0.5)
        self.assertAlmostEqual(d["share_constraint"]["design_share"], 0.5)
        self.assertAlmostEqual(d["share_constraint"]["policy_share_upper"], 0.5)
        self.assertGreater(d["share_constraint"]["residual_per_gamma"], 0.0)
        p = np.array([d["pi"][a] for a in ARMS])
        e = np.array(mu["Y_C"])
        res = (float(p @ e ** 2) + 3 * float(p @ e) ** 2 - 4 * 0.5 * float(p @ e)) / 3
        self.assertAlmostEqual(d["share_constraint"]["residual_per_gamma"], res)

    def test_the_infeasible_program_still_reports_the_design_share(self):
        mu, r = prog_inputs()
        d = C.certified_policy(ARMS, mu, r, {}, 0.0, "A0", {"Y_C": 0.05, "Y_L_f1": 0.9}, monotone=True, design_q=Q)
        self.assertEqual(d["lp_status"], "infeasible")
        self.assertFalse(d["switch"])
        self.assertAlmostEqual(d["share_constraint"]["design_share"], 0.375)

    def test_look_kwargs_reproduce_the_look(self):
        look = {"coupling": {"mode": "a5pp", "zeta_reference": {"Y_C": 0.06, "Y_Q": 0.01}}, "share_constraint": {"design_share": 0.35, "design_share_lower": 0.33}, "zeta": {"Y_C": 0.0, "Y_Q": 0.01},
                "parameters": {"n_nodes": 4}}
        kw = C.look_kwargs(look)
        self.assertEqual(kw["zeta"], {"Y_C": 0.06, "Y_Q": 0.01})
        self.assertTrue(kw["monotone"])
        self.assertEqual(kw["design_share"], (0.35, 0.33))
        self.assertFalse(kw["strict_support"])
        old = C.look_kwargs({"zeta": {"Y_C": 0.02}})
        self.assertEqual(old, {"zeta": {"Y_C": 0.02}})
        self.assertTrue(C.look_kwargs({**look, "coupling": {**look["coupling"], "mode": "a5pp_strict"}})["strict_support"])
        self.assertEqual(C.look_kwargs({**look, "coupling": {"mode": "estimated", "zeta_reference": {}}}), {"zeta": {"Y_C": 0.0, "Y_Q": 0.01}})


class OneSidedCertificate(unittest.TestCase):
    def lp(self, one_sided):
        arms = ["a0", "a1"]
        mu = {k: [0.0, 0.0] for k in OUTCOMES}
        r = {k: [0.0, 0.0] for k in OUTCOMES}
        mu["Y_Q"], r["Y_Q"] = [0.70, 0.60], [0.02, 0.02]
        mu["Y_C"] = [0.60, 0.10]
        return C.certified_policy(arms, mu, r, {}, 0.0, "a0", {"Y_C": 0.35}, one_sided_quality=one_sided)

    def test_hand_solved_cost_bound_in_both_forms(self):
        old, new = self.lp(False), self.lp(True)
        self.assertAlmostEqual(old["pi"]["a0"], 0.5, places=6)
        self.assertAlmostEqual(old["quality_cost_bound"], 0.72 - (0.5 * 0.68 + 0.5 * 0.58), places=9)
        self.assertAlmostEqual(new["quality_cost_bound"], 0.5 * (0.72 - 0.58), places=9)
        self.assertLess(new["quality_cost_bound"], old["quality_cost_bound"])
        self.assertEqual(new["switch"], old["switch"])
        self.assertIn("one-sided", new["quality_cost_note"])

    def test_the_two_forms_agree_when_the_incumbent_has_no_mass(self):
        mu = {k: [0.0, 0.0] for k in OUTCOMES}
        r = {k: [0.0, 0.0] for k in OUTCOMES}
        mu["Y_Q"], r["Y_Q"] = [0.60, 0.70], [0.02, 0.02]
        a = C.certified_policy(["a0", "a1"], mu, r, {}, 0.01, "a0", {"Y_C": 0.35}, one_sided_quality=False)
        b = C.certified_policy(["a0", "a1"], mu, r, {}, 0.01, "a0", {"Y_C": 0.35}, one_sided_quality=True)
        self.assertAlmostEqual(a["pi"]["a0"], 0.0, places=9)
        self.assertAlmostEqual(a["quality_cost_bound"], b["quality_cost_bound"], places=9)


if __name__ == "__main__":
    unittest.main()
