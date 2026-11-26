import unittest

import numpy as np

from ciara import pilot_program as GR
from ciara import pilot_selection as SR
from ciara import radius as R
from ciara import screening as S
from ciara.constants import OUTCOMES
import synth
from synth import A0, K1_F, KH_F, KH_P, W4, W8, tau_table_flat as tau_table

PROBE_ARMS = [A0, KH_F, KH_P, K1_F]


def est_of(spec, s2=0.001, n_blocks=8):
    est = {}
    for arm, over in spec.items():
        m = dict(SMALL)
        m.update(over)
        est[arm] = {"n_blocks": n_blocks, "outcomes": {k: {"mean": m[k], "s2": s2, "n": n_blocks, "n_w": 100, "s2_floor": 0.0, "s2_sample": s2, "values": []} for k in OUTCOMES}}
    return est


SMALL = {k: 0.01 for k in OUTCOMES} | {"Y_Q": 0.5, "Y_E_A": 0.2, "Y_E_B": 0.2, "Y_E_C": 0.2, "Y_C": 0.05}


def cfg_of(arms):
    return S.ScreenConfig(synth.arms_restricted(), tau_table(), lambda p, c: 5.0, used_profiles={"I1-E": [W4]}, cloud_share_mode=S.CLOUD_SET_LEVEL)


A0, A1, A2 = "I1-E", f"{W4}|b8|kH", f"{W8}|b8|kA"
MEMBERS = [A0, A1, A2]


class Program(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_of(MEMBERS)
        tau = self.cfg.tau_table.set_tau(self.cfg.profiles, MEMBERS)
        self.assertEqual(tau, 1)
        self.band = R.beta_dm(self.cfg.t_bar, self.cfg.b, self.cfg.l, tau)
        self.eta = R.eta_dep(self.cfg.t_bar, self.cfg.l_dep, tau)
        self.L = R.level(3)
        self.r_alt = R.closed_radius(0.001, R.planned_rounds(1 / 6, 4, self.cfg.k7), self.L, self.band)
        self.r_a0 = R.closed_radius(0.001, R.planned_rounds(2 / 3, 4, self.cfg.k7), self.L, self.band)

    def test_radii_follow_the_planned_count_of_the_arm(self):
        est = est_of({A0: {}, A1: {}, A2: {}})
        b = GR.member_bounds(A0, est, MEMBERS, A0, self.cfg)
        self.assertAlmostEqual(b["u"], 398.0247, places=3)
        self.assertAlmostEqual(b["outcomes"]["Y_C"]["radius"], self.r_a0)
        a = GR.member_bounds(A1, est, MEMBERS, A0, self.cfg)
        self.assertAlmostEqual(a["u"], 208.652, places=3)
        self.assertAlmostEqual(a["outcomes"]["Y_C"]["radius"], self.r_alt)
        self.assertGreater(self.r_alt, self.r_a0)
        self.assertAlmostEqual(a["eta_dep"], self.eta)

    def test_feasible_interval_solved_by_hand(self):
        est = est_of({A0: {"Y_C": 0.46, "Y_L_f1": 0.02, "Y_Q": 0.60}, A1: {"Y_C": 0.20, "Y_L_f1": 0.06, "Y_Q": 0.70}, A2: {"Y_C": 0.9, "Y_L_f1": 0.5, "Y_L_f2": 0.5, "Y_Q": 0.2}})
        c0, c1 = 0.46 + self.r_a0 + self.eta, 0.20 + self.r_alt + self.eta
        t0, t1 = 0.02 + self.r_a0 + self.eta, 0.06 + self.r_alt + self.eta
        hi = (0.30 - c1) / (c0 - c1)
        lo = (t1 - 0.10) / (t1 - t0)
        self.assertLess(lo, hi)
        self.assertGreater(lo, 0.0)
        prog = GR.program(est, MEMBERS, A0, self.cfg)
        self.assertTrue(prog["feasible"], prog)
        self.assertEqual(prog["lp_status"], "optimal")
        self.assertAlmostEqual(prog["pi"][A0], lo, places=6)
        self.assertAlmostEqual(prog["pi"][A2], 0.0, places=9)
        self.assertAlmostEqual(prog["upper_bounds"]["Y_L_f1"], 0.10, places=6)
        self.assertLessEqual(prog["upper_bounds"]["Y_C"], 0.30 + 1e-9)
        self.assertEqual(prog["rows_no_member_meets"], [])

    def test_infeasible_when_the_two_rows_leave_no_interval(self):
        est = est_of({A0: {"Y_C": 0.46, "Y_L_f1": 0.02}, A1: {"Y_C": 0.20, "Y_L_f1": 0.12}, A2: {"Y_C": 0.9, "Y_L_f1": 0.5}})
        c0, c1 = 0.46 + self.r_a0 + self.eta, 0.20 + self.r_alt + self.eta
        t0, t1 = 0.02 + self.r_a0 + self.eta, 0.12 + self.r_alt + self.eta
        hi = (0.30 - c1) / (c0 - c1)
        lo = (t1 - 0.10) / (t1 - t0)
        self.assertGreater(lo, hi)
        prog = GR.program(est, MEMBERS, A0, self.cfg)
        self.assertFalse(prog["feasible"])
        self.assertEqual(prog["lp_status"], "infeasible")
        self.assertEqual(prog["rows_no_member_meets"], [])

    def test_cloud_row_that_no_member_meets_is_infeasible_whatever_the_mixture(self):
        est = est_of({A0: {"Y_C": 0.46}, A1: {"Y_C": 0.27}, A2: {"Y_C": 0.26}})
        self.assertGreater(0.26 + self.r_alt + self.eta, 0.30)
        prog = GR.program(est, MEMBERS, A0, self.cfg)
        self.assertFalse(prog["feasible"])
        self.assertEqual(prog["rows_no_member_meets"], ["Y_C"])
        self.assertAlmostEqual(prog["min_member_upper_bound"]["Y_C"], 0.26 + self.r_alt + self.eta, places=9)

    def test_boundary_member_exactly_at_budget_is_feasible(self):
        m = 0.30 - self.r_alt - self.eta
        est = est_of({A0: {"Y_C": 0.46}, A1: {"Y_C": m}, A2: {"Y_C": 0.9, "Y_L_f1": 0.5}})
        prog = GR.program(est, MEMBERS, A0, self.cfg)
        self.assertTrue(prog["feasible"])
        self.assertAlmostEqual(prog["pi"][A1], 1.0, places=6)
        est[A1]["outcomes"]["Y_C"]["mean"] = m + 1e-6
        self.assertFalse(GR.program(est, MEMBERS, A0, self.cfg)["feasible"])

    def test_zeta_enters_only_the_sensitivity(self):
        est = est_of({A0: {"Y_C": 0.46}, A1: {"Y_C": 0.28 - self.r_alt - self.eta + 0.0}, A2: {"Y_C": 0.9, "Y_L_f1": 0.5}})
        sel = {"status": "selected", "alternatives": [A1, A2], "S": MEMBERS, "zeta": {"zeta": {"Y_C": 0.05}}}
        go = GR.evaluate([], est, self.cfg, A0, sel)
        self.assertTrue(go["go"])
        self.assertEqual(go["with_zeta"]["lp_status"], "infeasible")
        self.assertEqual(go["with_zeta"]["rows_no_member_meets"], ["Y_C"])

    def test_missing_estimates_give_no_program(self):
        est = est_of({A0: {}, A1: {}, A2: {}}, n_blocks=2)
        prog = GR.program(est, MEMBERS, A0, self.cfg)
        self.assertFalse(prog["feasible"])
        self.assertEqual(prog["lp_status"], "no_estimate")

    def test_member_order_does_not_change_the_answer(self):
        est = est_of({A0: {"Y_C": 0.46, "Y_L_f1": 0.02, "Y_Q": 0.60}, A1: {"Y_C": 0.20, "Y_L_f1": 0.06, "Y_Q": 0.70}, A2: {"Y_C": 0.9, "Y_L_f1": 0.5, "Y_Q": 0.2}})
        p1 = GR.program(est, [A0, A1, A2], A0, self.cfg)
        p2 = GR.program(est, [A0, A2, A1], A0, self.cfg)
        self.assertAlmostEqual(p1["pi"][A0], p2["pi"][A0], places=9)
        self.assertAlmostEqual(p1["pi"][A1], p2["pi"][A1], places=9)


class Evaluate(unittest.TestCase):
    def test_fewer_than_two_alternatives_is_no_go_with_an_informational_program(self):
        cfg = cfg_of(MEMBERS)
        est = est_of({A0: {"Y_C": 0.46}, A1: {"Y_C": 0.2}, A2: {"Y_C": 0.2}, f"{W4}|b8|k1": {"Y_L_f2": 0.2}, f"{W8}|b8|k1": {"Y_L_f2": 0.2}, f"{W8}|b8|kH": {"Y_L_f2": 0.2},
                      f"{W4}|b8|kA": {"Y_L_f2": 0.2}, "I2": {}, "I3": {}})
        sel = {"status": "fewer_than_two_alternatives", "alternatives": [A1], "S": [A0, A1]}
        go = GR.evaluate([], est, cfg, A0, sel)
        self.assertFalse(go["go"])
        self.assertTrue(go["informational"])
        self.assertEqual(go["set"][0], A0)
        self.assertEqual(len(go["set"]), 3)
        self.assertIn("two are required", go["reason"])

    def test_selected_with_feasible_program_is_go(self):
        cfg = cfg_of(MEMBERS)
        est = est_of({A0: {"Y_C": 0.46}, A1: {"Y_C": 0.15}, A2: {"Y_C": 0.2}})
        sel = {"status": "selected", "alternatives": [A1, A2], "S": MEMBERS, "zeta": {"zeta": {}}}
        go = GR.evaluate([], est, cfg, A0, sel)
        self.assertTrue(go["go"])
        self.assertFalse(go["informational"])

    def test_selected_but_cloud_row_unmet_is_no_go(self):
        cfg = cfg_of(MEMBERS)
        est = est_of({A0: {"Y_C": 0.46}, A1: {"Y_C": 0.30}, A2: {"Y_C": 0.34}})
        sel = {"status": "selected", "alternatives": [A1, A2], "S": MEMBERS, "zeta": {"zeta": {}}}
        go = GR.evaluate([], est, cfg, A0, sel)
        self.assertFalse(go["go"])
        self.assertIn("Y_C", go["reason"])


def truth(go):
    t = {}
    for a in synth.arms_restricted():
        if a["kind"] == "static":
            t[a["id"]] = {"Y_Q": 0.68, "Y_L_f1": 0.02, "Y_L_f2": 0.30 if go else 0.02, "Y_D": 0.02, "Y_C": 0.46}
    t["I1-E"] = {"Y_Q": 0.69, "Y_C": 0.46}
    t["I2"] = {"Y_Q": 0.45, "Y_C": 0.1, "Y_L_f1": 0.5}
    t["I3"] = {"Y_Q": 0.68, "Y_C": 0.46}
    if go:
        t[f"{W4}|b8|kH"] = {"Y_Q": 0.66, "Y_L_f1": 0.02, "Y_L_f2": 0.02, "Y_D": 0.02, "Y_C": 0.15}
        t[f"{W8}|b8|kA"] = {"Y_Q": 0.65, "Y_L_f1": 0.02, "Y_L_f2": 0.02, "Y_D": 0.02, "Y_C": 0.25}
    return t


U, N14 = 576, 14


def cfg4():
    return S.ScreenConfig(synth.arms_balanced(), tau_table(), lambda p, c: 5.0, n_set=4, used_profiles={A0: [W4, W8]}, cloud_share_mode=S.CLOUD_SET_LEVEL)


class BalancedProgram(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg4()
        self.tau = self.cfg.tau_table.set_tau(self.cfg.profiles, PROBE_ARMS)
        self.band = R.beta_dm(self.cfg.t_bar, self.cfg.b, self.cfg.l, self.tau)
        self.eta = R.eta_dep(self.cfg.t_bar, self.cfg.l_dep, self.tau)
        self.L1 = R.level(4, one_sided=True)
        self.L2 = R.level(4)
        self.s2 = 0.001
        self.r = R.closed_radius(self.s2, U, self.L1, self.band)

    def est(self, spec, n=N14):
        return est_of({a: spec.get(a, {}) for a in PROBE_ARMS}, s2=self.s2, n_blocks=n)

    def test_level_is_one_sided_and_the_count_is_the_rounds(self):
        self.assertAlmostEqual(self.L1, np.log(4 * 8 / 0.05))
        self.assertAlmostEqual(self.L2 - self.L1, np.log(2.0))
        est = self.est({})
        b = GR.member_bounds(A0, est, PROBE_ARMS, A0, self.cfg, u_fixed=R.planned_rounds_balanced(U), one_sided=True)
        self.assertEqual(b["u"], 576.0)
        self.assertAlmostEqual(b["L"], self.L1)
        self.assertAlmostEqual(b["outcomes"]["Y_C"]["radius"], self.r)
        two = GR.member_bounds(A0, est, PROBE_ARMS, A0, self.cfg, u_fixed=576.0, one_sided=False)
        self.assertGreater(two["outcomes"]["Y_C"]["radius"], self.r)
        old = GR.member_bounds(KH_P, est, PROBE_ARMS, A0, self.cfg)
        self.assertLess(old["u"], 576.0)

    def test_standard_errors_and_the_adverse_shift(self):
        est = self.est({KH_P: {"Y_C": 0.2, "Y_Q": 0.7}})
        se = GR.standard_errors(est, KH_P)
        self.assertAlmostEqual(se["Y_C"], (self.s2 / N14) ** 0.5)
        sh = GR.shift_adverse(est, [KH_P], 1.0)
        self.assertAlmostEqual(sh[KH_P]["outcomes"]["Y_C"]["mean"], 0.2 + se["Y_C"])
        self.assertAlmostEqual(sh[KH_P]["outcomes"]["Y_Q"]["mean"], 0.7 - se["Y_Q"])
        self.assertEqual(est[KH_P]["outcomes"]["Y_C"]["mean"], 0.2)
        self.assertEqual(sh[A0]["outcomes"]["Y_C"]["mean"], est[A0]["outcomes"]["Y_C"]["mean"])

    def test_feasible_mixture_solved_by_hand(self):
        bad = {"Y_C": 0.9, "Y_L_f1": 0.5, "Y_L_f2": 0.5, "Y_D": 0.5, "Y_Q": 0.2}
        est = self.est({A0: {"Y_C": 0.46, "Y_L_f1": 0.02, "Y_Q": 0.60}, KH_P: {"Y_C": 0.20, "Y_L_f1": 0.09, "Y_Q": 0.70}, KH_F: bad, K1_F: bad})
        c0, c1 = 0.46 + self.r + self.eta, 0.20 + self.r + self.eta
        t0, t1 = 0.02 + self.r + self.eta, 0.09 + self.r + self.eta
        hi, lo = (0.30 - c1) / (c0 - c1), (t1 - 0.10) / (t1 - t0)
        self.assertLess(lo, hi)
        self.assertGreater(lo, 0.0)
        prog = GR.program(est, PROBE_ARMS, A0, self.cfg, None, 576.0, True)
        self.assertTrue(prog["feasible"], prog)
        self.assertAlmostEqual(prog["pi"][A0], lo, places=6)
        self.assertAlmostEqual(prog["pi"][KH_P], 1.0 - lo, places=6)
        self.assertAlmostEqual(prog["pi"][KH_F], 0.0, places=9)
        self.assertAlmostEqual(prog["upper_bounds"]["Y_L_f1"], 0.10, places=6)
        self.assertEqual(prog["rows_no_member_meets"], [])

    def test_two_sided_level_can_remove_the_feasible_point(self):
        r2 = R.closed_radius(self.s2, U, self.L2, self.band)
        self.assertGreater(r2, self.r)
        m = 0.30 - self.eta - (self.r + r2) / 2.0
        bad = {"Y_C": 0.9, "Y_L_f1": 0.5, "Y_L_f2": 0.5, "Y_D": 0.5}
        est = self.est({A0: {"Y_C": 0.46}, KH_P: {"Y_C": m}, KH_F: bad, K1_F: bad})
        self.assertTrue(GR.program(est, PROBE_ARMS, A0, self.cfg, None, 576.0, True)["feasible"])
        self.assertFalse(GR.program(est, PROBE_ARMS, A0, self.cfg, None, 576.0, False)["feasible"])

    def test_ranking_by_weight_with_ties_by_quality_lower_bound(self):
        prog = {"pi": {A0: 0.3, KH_F: 0.1, KH_P: 0.6, K1_F: 0.0}, "bounds": {a: {"outcomes": {"Y_Q": {"mean": 0.6, "radius": 0.01}}} for a in PROBE_ARMS}}
        r = GR.weights(prog, A0, {}, PROBE_ARMS)
        self.assertEqual([x["arm"] for x in r], [KH_P, KH_F, K1_F])
        prog = {"pi": {A0: 0.2, KH_F: 0.4, KH_P: 0.4, K1_F: 0.0}, "bounds": {a: {"outcomes": {"Y_Q": {"mean": q, "radius": 0.01}}} for a, q in zip(PROBE_ARMS, (0.5, 0.60, 0.70, 0.65))}}
        self.assertEqual([x["arm"] for x in GR.weights(prog, A0, {}, PROBE_ARMS)][:2], [KH_P, KH_F])
        prog["bounds"][KH_P]["outcomes"]["Y_Q"]["mean"] = 0.60
        self.assertEqual([x["arm"] for x in GR.weights(prog, A0, {}, PROBE_ARMS)][:2], [KH_F, KH_P])


def truth(go):
    t = {}
    for a in synth.arms_restricted():
        if a["kind"] == "static":
            t[a["id"]] = {"Y_Q": 0.68, "Y_L_f1": 0.02, "Y_L_f2": 0.30 if go else 0.02, "Y_D": 0.02, "Y_C": 0.46}
    t["I1-E"] = {"Y_Q": 0.69, "Y_C": 0.46}
    t["I2"] = {"Y_Q": 0.45, "Y_C": 0.1, "Y_L_f1": 0.5}
    t["I3"] = {"Y_Q": 0.68, "Y_C": 0.46}
    if go:
        t[f"{W4}|b8|kH"] = {"Y_Q": 0.66, "Y_L_f1": 0.02, "Y_L_f2": 0.02, "Y_D": 0.02, "Y_C": 0.15}
        t[f"{W8}|b8|kA"] = {"Y_Q": 0.65, "Y_L_f1": 0.02, "Y_L_f2": 0.02, "Y_D": 0.02, "Y_C": 0.25}
    return t


class Analysis(unittest.TestCase):
    def run_analysis(self, go_truth, flag=True):
        arms = synth.arms_restricted()
        rows = synth.make_rows(truth(go_truth), arms=arms, n_nodes=4, n_blocks=17 * 2, seed=7, block_noise=0.3)
        return SR.analyze(rows, arms, tau_table(), synth.table_default(), cloud_share_mode=S.CLOUD_SET_LEVEL, go_rule=flag)

    def test_go_when_alternatives_with_low_cloud_share_pass(self):
        res = self.run_analysis(True)
        self.assertEqual(res["selection"]["status"], "selected", res["selection"]["history"])
        self.assertEqual(res["incumbent"]["a0"], "I1-E")
        self.assertTrue(res["go_rule"]["go"], res["go_rule"]["reason"])
        self.assertTrue(res["go_rule"]["program"]["feasible"])

    def test_no_go_when_two_alternatives_pass_but_every_member_exceeds_the_cloud_budget(self):
        res = self.run_analysis(False)
        self.assertEqual(res["selection"]["status"], "selected", res["selection"]["history"])
        gr = res["go_rule"]
        self.assertFalse(gr["go"])
        self.assertEqual(gr["program"]["rows_no_member_meets"], ["Y_C"])
        self.assertEqual(gr["program"]["lp_status"], "infeasible")

    def test_without_the_flag_nothing_changes(self):
        res = self.run_analysis(False, flag=False)
        self.assertNotIn("go_rule", res)


if __name__ == "__main__":
    unittest.main()
