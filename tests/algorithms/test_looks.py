import random
import unittest
from unittest import mock

import numpy as np

from ciara import looks as LK
from ciara import program as CT
import synth
from synth import A0, K1_F, KH_P, MEM3, PROF3, TRUTH_GO, balanced_campaign_rows, tau_table, tau_table_flat

NODES = 4
MEMBERS = ["I1-E", "q35-4b-ih-w4a16|b16|k0", "q35-2b-fp16|b16|k0"]
PROFILES = {"I1-E": ["q35-4b-ih-w4a16", "q35-2b-fp16"], "q35-4b-ih-w4a16|b16|k0": ["q35-4b-ih-w4a16"], "q35-2b-fp16|b16|k0": ["q35-2b-fp16"]}


def campaign_rows(truth, n_blocks, seed=1, eps=0.5, noise=1.0):
    rng = random.Random(seed)
    q = [(1 - eps if a == "I1-E" else 0.0) + eps / 3 for a in MEMBERS]
    force = {b: [rng.choices(MEMBERS, weights=q)[0] for _ in range(NODES)] for b in range(n_blocks)}
    return synth.make_rows(truth, seed=seed, n_blocks=n_blocks, n_nodes=NODES, force=force, block_noise=noise)


TRUTH_BETTER = {"I1-E": {"Y_Q": 0.55}, "q35-4b-ih-w4a16|b16|k0": {"Y_Q": 0.80}, "q35-2b-fp16|b16|k0": {"Y_Q": 0.70}}
TRUTH_SAME = {"I1-E": {"Y_Q": 0.60}, "q35-4b-ih-w4a16|b16|k0": {"Y_Q": 0.60}, "q35-2b-fp16|b16|k0": {"Y_Q": 0.60}}


class Looks(unittest.TestCase):
    def test_look_blocks(self):
        self.assertEqual([LK.look_blocks(d) for d in (4, 7, 10)], [230, 403, 576])

    def test_round_table(self):
        rows = campaign_rows(TRUTH_SAME, 12)
        rows[3]["block_status"] = "interrupted"
        Z, F = LK.round_table(rows, MEMBERS, NODES, 12)
        self.assertEqual(Z.shape, (12, NODES))
        self.assertEqual(int(np.sum(Z == -1)), 1)
        self.assertEqual(F.shape, (12, NODES, 8))

    def test_decision_switches_with_a_large_gain(self):
        rows = campaign_rows(TRUTH_BETTER, 403, seed=3)
        dec = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 7, k_rounds=403)
        self.assertEqual(dec["lp_status"], "optimal")
        self.assertTrue(dec["switch"], dec["reason"])
        self.assertTrue(dec["stop_campaign"])
        self.assertAlmostEqual(sum(dec["pi"].values()), 1.0)
        self.assertEqual(dec["complete_rounds"], 403)
        from ciara.constants import BUDGETS_TIGHT
        d4 = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 4, budgets=BUDGETS_TIGHT, k_rounds=230)
        self.assertFalse(d4["switch"])

    def test_every_arm_uses_the_set_tau(self):
        rows = campaign_rows(TRUTH_SAME, 230, seed=5)
        dec = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 4, k_rounds=230)
        want = tau_table().set_tau(PROFILES, MEMBERS)
        self.assertEqual({t["tau"] for t in dec["arms_table"].values()}, {want})
        self.assertEqual(dec["tau_max"], want)

    def test_decision_keeps_incumbent_without_gain(self):
        rows = campaign_rows(TRUTH_SAME, 230, seed=4)
        dec = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 4, k_rounds=230)
        self.assertFalse(dec["switch"])

    def test_radius_shrinks_between_looks(self):
        rows = campaign_rows(TRUTH_BETTER, 403, seed=5)
        d4 = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 4, k_rounds=230)
        d7 = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 7, k_rounds=403)
        self.assertGreater(d4["arms_table"][MEMBERS[1]]["outcomes"]["Y_Q"]["radius"], d7["arms_table"][MEMBERS[1]]["outcomes"]["Y_Q"]["radius"])

    def test_looks_are_prefix_consistent(self):
        rows = campaign_rows(TRUTH_BETTER, 403, seed=6)
        short = [r for r in rows if r["block_id"] < 230]
        a = LK.look_decision(short, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 4, k_rounds=230)
        b = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 4, k_rounds=230)
        for m in MEMBERS:
            self.assertEqual(a["arms_table"][m]["outcomes"]["Y_Q"]["radius"], b["arms_table"][m]["outcomes"]["Y_Q"]["radius"])

    def test_zeta_blocks_the_switch(self):
        rows = campaign_rows({"I1-E": {"Y_Q": 0.60}, "q35-4b-ih-w4a16|b16|k0": {"Y_Q": 0.70}, "q35-2b-fp16|b16|k0": {"Y_Q": 0.66}}, 403, seed=7)
        free = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 7, k_rounds=403)
        held = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {"Y_Q": 0.05}, 7, k_rounds=403)
        self.assertTrue(free["switch"])
        self.assertFalse(held["switch"])

    def test_no_false_switch_under_the_null(self):
        sw = 0
        for s in range(40):
            rows = campaign_rows(TRUTH_SAME, 403, seed=100 + s)
            dec = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 7, k_rounds=403)
            sw += 1 if dec["switch"] else 0
        self.assertEqual(sw, 0)


TRUTH_INCUMBENT_OVER_CLOUD = {"I1-E": {"Y_Q": 0.62, "Y_C": 0.46}, "q35-4b-ih-w4a16|b16|k0": {"Y_Q": 0.50, "Y_C": 0.05},
                              "q35-2b-fp16|b16|k0": {"Y_Q": 0.45, "Y_C": 0.02}}


class BudgetComplianceLooks(unittest.TestCase):
    def test_default_budgets_and_nodes(self):
        from ciara.constants import BUDGETS, N_NODES
        self.assertEqual(N_NODES, 4)
        self.assertEqual((BUDGETS["Y_L_f1"], BUDGETS["Y_L_f2"], BUDGETS["Y_D"], BUDGETS["Y_E_A"], BUDGETS["Y_C"]), (0.10, 0.10, 0.10, 0.5, 0.30))

    def test_gate_branch_is_unchanged_when_the_incumbent_is_feasible(self):
        rows = campaign_rows(TRUTH_BETTER, 403, seed=3)
        dec = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 7, k_rounds=403)
        self.assertEqual(dec["branch"], "gate")
        self.assertTrue(dec["incumbent_feasible"])
        self.assertTrue(dec["gate_applies"])
        self.assertTrue(dec["switch"])
        self.assertIsNotNone(dec["quality_cost_bound"])
        self.assertLess(dec["quality_cost_bound"], 0.0)

    def test_gate_branch_keeps_the_incumbent_without_gain(self):
        rows = campaign_rows(TRUTH_SAME, 403, seed=4)
        dec = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 7, k_rounds=403)
        self.assertEqual(dec["branch"], "gate")
        self.assertFalse(dec["switch"])

    def test_compliance_branch_deploys_a_certified_policy_that_is_not_better_in_quality(self):
        rows = campaign_rows(TRUTH_INCUMBENT_OVER_CLOUD, 403, seed=8)
        dec = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 7, k_rounds=403)
        self.assertEqual(dec["branch"], "budget_compliance")
        self.assertFalse(dec["incumbent_feasible"])
        self.assertIn("Y_C", dec["incumbent_violations"])
        self.assertEqual(dec["lp_status"], "optimal")
        self.assertTrue(dec["switch"], dec["reason"])
        self.assertTrue(dec["stop_campaign"])
        self.assertFalse(dec["gate_applies"])
        self.assertGreater(dec["nonincumbent_mass"], 0.0)
        self.assertGreater(dec["quality_cost_bound"], 0.0)
        self.assertLess(dec["quality_cost_bound"], 0.30)
        self.assertLessEqual(dec["upper_bounds"]["Y_C"], 0.30 + 1e-9)

    def test_compliance_branch_waits_while_no_certified_policy_exists(self):
        rows = campaign_rows(TRUTH_INCUMBENT_OVER_CLOUD, 230, seed=8)
        dec = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 4, k_rounds=230)
        self.assertEqual(dec["branch"], "budget_compliance")
        if dec["switch"]:
            for k, ub in dec["upper_bounds"].items():
                self.assertLessEqual(ub, dec["budgets"][k] + 1e-9)
        else:
            self.assertIn(dec["lp_status"], ("infeasible", "optimal"))

    def test_infeasible_compliance_keeps_the_incumbent(self):
        t = {"I1-E": {"Y_Q": 0.62, "Y_C": 0.46}, "q35-4b-ih-w4a16|b16|k0": {"Y_Q": 0.5, "Y_C": 0.40}, "q35-2b-fp16|b16|k0": {"Y_Q": 0.45, "Y_C": 0.38}}
        rows = campaign_rows(t, 403, seed=9)
        dec = LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, 7, k_rounds=403)
        self.assertEqual(dec["branch"], "budget_compliance")
        self.assertEqual(dec["lp_status"], "infeasible")
        self.assertFalse(dec["switch"])
        self.assertIn("incumbent is kept", dec["reason"])


if __name__ == "__main__":
    unittest.main()


class BalancedLooks(unittest.TestCase):
    def test_every_arm_is_present_in_every_round(self):
        rows = balanced_campaign_rows(TRUTH_GO, 230)
        Z, F = LK.round_table(rows, MEM3, 4, 230)
        for i in range(3):
            self.assertTrue(np.all((Z == i).any(axis=1)))
        self.assertEqual(int((Z == 0).sum(axis=1).min()), 2)

    def test_look_uses_rounds_as_the_count_and_the_one_sided_level(self):
        rows = balanced_campaign_rows(TRUTH_GO, 230)
        old = LK.look_decision(rows, MEM3, A0, PROF3, tau_table_flat(), {}, 4, k_rounds=230)
        new = LK.look_decision(rows, MEM3, A0, PROF3, tau_table_flat(), {}, 4, k_rounds=230, composition=[A0, A0, KH_P, K1_F], one_sided=True)
        self.assertAlmostEqual(new["level_L"], np.log(3 * 8 / 0.05))
        self.assertAlmostEqual(old["level_L"], np.log(2 * 3 * 8 / 0.05))
        self.assertEqual(new["arms_table"][A0]["propensity"], 0.5)
        self.assertEqual(new["arms_table"][KH_P]["propensity"], 0.25)
        self.assertEqual(new["arms_table"][KH_P]["nonempty_rounds"]["Y_C"], 230.0)
        self.assertEqual(new["arms_table"][A0]["nonempty_rounds"]["Y_C"], 230.0)
        self.assertEqual(new["arms_table"][KH_P]["planned_rounds"], 403.0)
        self.assertLess(new["arms_table"][KH_P]["outcomes"]["Y_C"]["radius"], old["arms_table"][KH_P]["outcomes"]["Y_C"]["radius"])
        self.assertTrue(new["parameters"]["balanced"] and new["parameters"]["one_sided"])

    def test_look_decision_passes_the_flag(self):
        rows = balanced_campaign_rows(TRUTH_GO, 230)
        d1 = LK.look_decision(rows, MEM3, A0, PROF3, tau_table_flat(), {}, 4, k_rounds=230, composition=[A0, A0, KH_P, K1_F], one_sided=True)
        d0 = LK.look_decision(rows, MEM3, A0, PROF3, tau_table_flat(), {}, 4, k_rounds=230, composition=[A0, A0, KH_P, K1_F], one_sided=False)
        if d1.get("quality_cost_note"):
            self.assertIn("one-sided", d1["quality_cost_note"])
            self.assertNotIn("one-sided", d0["quality_cost_note"])
        self.assertEqual(d1["parameters"]["one_sided"], True)


class LooksUnderA5pp(unittest.TestCase):
    def rows(self, n=403):
        return balanced_campaign_rows(TRUTH_GO, n, seed=7)

    def decide(self, mode, zeta, n=403, rows=None, day=7):
        return LK.look_decision(rows or self.rows(n), MEM3, A0, PROF3, tau_table_flat(), zeta, day, k_rounds=n, composition=[A0, A0, KH_P, K1_F], one_sided=True, coupling_mode=mode)

    def test_the_coupling_slack_is_dropped_on_the_cost_rows_and_kept_on_quality(self):
        zeta = {"Y_C": 0.063, "Y_E_B": 0.058, "Y_L_f1": 0.01, "Y_Q": 0.012}
        d = self.decide("a5pp", zeta)
        self.assertEqual(d["coupling"]["mode"], "a5pp")
        self.assertEqual(d["coupling"]["zeta_used"], {"Y_C": 0.0, "Y_E_B": 0.0, "Y_L_f1": 0.0, "Y_Q": 0.012})
        self.assertEqual(d["coupling"]["zeta_reference"], zeta)
        self.assertEqual(d["zeta"], d["coupling"]["zeta_used"])
        self.assertTrue(d["share_constraint"]["holds"])
        self.assertAlmostEqual(d["share_constraint"]["design_share"], sum(q * d["arms_table"][a]["outcomes"]["Y_C"]["centre"] for q, a in zip((0.5, 0.25, 0.25), MEM3)))
        self.assertEqual(set(d["coupling"]["sensitivity"]), {"a5pp_strict", "estimated"})

    def test_a_slack_that_blocks_the_estimated_program_does_not_block_a5pp(self):
        zeta = {"Y_C": 0.2, "Y_Q": 0.012}
        est = self.decide("estimated", zeta)
        mono = self.decide("a5pp", zeta)
        self.assertEqual(est["lp_status"], "infeasible")
        self.assertFalse(est["switch"])
        self.assertEqual(mono["lp_status"], "optimal")
        self.assertTrue(mono["switch"], mono["reason"])
        self.assertEqual(mono["coupling"]["sensitivity"]["estimated"]["lp_status"], "infeasible")
        self.assertFalse(mono["coupling"]["sensitivity"]["estimated"]["switch"])
        self.assertEqual(est["coupling"]["sensitivity"]["a5pp"]["lp_status"], "optimal")
        self.assertGreaterEqual(mono["quality_cost_bound"], 2 * 0.012 - 1e-12 - 1.0)

    def test_the_policy_meets_the_share_constraint_and_the_cloud_budget(self):
        mono = self.decide("a5pp", {"Y_Q": 0.01})
        sc = mono["share_constraint"]
        self.assertLessEqual(sc["policy_share_upper"], sc["design_share"] + 1e-9)
        self.assertLessEqual(mono["upper_bounds"]["Y_C"], 0.30 + 1e-9)

    def test_the_strict_mode_reports_the_forbidden_arms(self):
        d = self.decide("a5pp_strict", {"Y_Q": 0.01})
        self.assertEqual(d["coupling"]["mode"], "a5pp_strict")
        self.assertIn(A0, d["share_constraint"]["forbidden_arms"])
        if d.get("pi"):
            self.assertAlmostEqual(d["pi"][A0], 0.0)

    def test_estimated_is_the_default_and_unchanged(self):
        rows = self.rows(230)
        old = LK.look_decision(rows, MEM3, A0, PROF3, tau_table_flat(), {"Y_C": 0.02}, 4, k_rounds=230, composition=[A0, A0, KH_P, K1_F], one_sided=True)
        self.assertEqual(old["coupling"]["mode"], "estimated")
        self.assertEqual(old["zeta"], {"Y_C": 0.02})
        self.assertNotIn("share_constraint", old)
        with self.assertRaises(ValueError):
            LK.look_decision(rows, MEM3, A0, PROF3, tau_table_flat(), {}, 4, k_rounds=230, coupling_mode="other")

    def gated(self, rho=None, strict_budgets=None):
        real = CT.certified_policy

        def wrapped(*a, **k):
            if k.get("strict_support") and strict_budgets is not None:
                a = a[:6] + (strict_budgets,) + a[7:]
            d = real(*a, **k)
            if rho is not None and k.get("monotone") and not k.get("strict_support") and d.get("share_constraint") and "residual_per_gamma" in d["share_constraint"]:
                d["share_constraint"]["residual_per_gamma"] = rho
            return d
        with mock.patch.object(LK.CT, "certified_policy", wrapped):
            return self.decide("a5pp_gated", {"Y_C": 0.063, "Y_Q": 0.012})

    def test_gated_branch_rho_nonpositive_deploys_the_a5pp_solution(self):
        g = self.gated()
        a = self.decide("a5pp", {"Y_C": 0.063, "Y_Q": 0.012})
        self.assertLessEqual(g["share_constraint"]["residual_per_gamma"], 0.0)
        self.assertEqual(g["coupling"]["mode"], "a5pp_gated")
        self.assertEqual(g["coupling"]["applied"], "a5pp")
        self.assertEqual(g["coupling"]["gated"]["decided_by"], "a5pp")
        self.assertEqual(g["pi"], a["pi"])
        self.assertTrue(g["switch"])
        self.assertEqual(set(g["coupling"]["sensitivity"]), {"a5pp_strict", "estimated"})
        self.assertEqual(g["coupling"]["gated"]["rho"], g["share_constraint"]["residual_per_gamma"])

    def test_gated_branch_rho_positive_with_a_feasible_strict_program_deploys_the_strict_solution(self):
        g = self.gated(rho=0.02)
        s = self.decide("a5pp_strict", {"Y_C": 0.063, "Y_Q": 0.012})
        c = g["coupling"]
        self.assertEqual((c["applied"], c["gated"]["decided_by"]), ("a5pp_strict", "a5pp_strict"))
        self.assertAlmostEqual(c["gated"]["rho"], 0.02)
        self.assertTrue(g["switch"])
        self.assertEqual(g["pi"], s["pi"])
        self.assertTrue(g["share_constraint"]["forbidden_arms"])
        self.assertIsNotNone(c["gated"]["a5pp_pi"])
        self.assertEqual(set(c["sensitivity"]), {"a5pp", "estimated"})
        self.assertTrue(CT.look_kwargs({**g, "parameters": {"n_nodes": 4}})["strict_support"])

    def test_gated_branch_rho_positive_with_an_infeasible_strict_program_keeps_the_incumbent(self):
        g = self.gated(rho=0.02, strict_budgets={"Y_C": 0.0, "Y_L_f1": 0.9})
        c = g["coupling"]
        self.assertEqual(c["gated"]["decided_by"], "a5pp_strict")
        self.assertIn("incumbent is kept", c["gated"]["reason"])
        self.assertEqual(g["lp_status"], "infeasible")
        self.assertFalse(g["switch"])
        self.assertFalse(g["stop_campaign"])
        self.assertIsNone(g["pi"])


if __name__ == "__main__":
    unittest.main()
