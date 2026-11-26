import unittest

from ciara import pilot_selection as SR
from ciara import screening as S
from ciara.constants import BUDGETS_TIGHT
import synth
from synth import tau_table
from test_pilot_selection import PILOT, truth_two_pass


class Estimates(unittest.TestCase):
    def test_estimates_are_dm_means(self):
        arms = synth.arms_default()
        rows = synth.make_rows(truth_two_pass(), arms=arms, seed=4)
        est = S.arm_estimates(S.usable_rows(rows), [a["id"] for a in arms])
        a = "q35-4b-ih-w8a8|b16|k0"
        ys = [r["Y_Q"] for r in rows if r["arm_id"] == a]
        self.assertAlmostEqual(est[a]["outcomes"]["Y_Q"]["mean"], sum(ys) / len(ys))
        self.assertEqual(est[a]["n_blocks"], len(ys))
        o = est[a]["outcomes"]["Y_L_f1"]
        self.assertGreaterEqual(o["s2"], o["s2_floor"])


class Zeta(unittest.TestCase):
    def test_recovers_planted_slope(self):
        arms = synth.arms_default()
        cl = {a["id"]: (0.47 if (a["kind"] == "static" and a["kappa"] == 1) else 0.0) for a in arms}
        cl["I1-E"] = 0.15
        truth = {a["id"]: {"Y_C": cl[a["id"]]} for a in arms}
        rows = synth.make_rows(truth, arms=arms, seed=11, n_blocks=400, coupling={"Y_L_f2": 0.10}, cl=cl, block_noise=0.5)
        slopes = S.coupling_slopes(S.usable_rows(rows), "I1-E", cl)
        self.assertAlmostEqual(slopes["Y_L_f2"], 0.10, delta=0.03)
        self.assertAlmostEqual(slopes["Y_Q"], 0.0, delta=0.05)

    def test_gap_and_zeta(self):
        cl = {"a0": 0.2, "a1": 0.0, "a2": 0.0}
        gap = S.escalation_gap(["a0", "a1", "a2"], "a0", cl)
        self.assertAlmostEqual(gap, 2 * 0.2 / 3)
        cl2 = {"a0": 0.0, "a1": 0.0, "a2": 0.0}
        self.assertEqual(S.escalation_gap(["a0", "a1", "a2"], "a0", cl2), 0.0)


class Replacement(unittest.TestCase):
    def setUp(self):
        self.arms = synth.arms_default()
        t = {}
        for a in self.arms:
            if a["kind"] == "static":
                t[a["id"]] = {"Y_Q": 0.40, "Y_L_f2": 0.2}
        t["q35-4b-ih-w4a16|b16|k0"] = {"Y_Q": 0.68, "Y_C": 0.20, "Y_L_f2": 0.002}
        t["q35-2b-fp16|b16|k0"] = {"Y_Q": 0.66, "Y_L_f2": 0.002}
        t["q35-2b-fp16|b8|k0"] = {"Y_Q": 0.64, "Y_L_f2": 0.002}
        t["q35-0.8b-fp16|b16|k0"] = {"Y_Q": 0.55}
        t["I1-E"] = {"Y_Q": 0.60, "Y_C": 0.05}
        t["I2"] = {"Y_Q": 0.50}
        t["I3"] = {"Y_Q": 0.50}
        self.rows = synth.make_rows(t, arms=self.arms, seed=8, block_noise=0.2)
        self.est = S.arm_estimates(S.usable_rows(self.rows), [a["id"] for a in self.arms])
        self.cfg = S.ScreenConfig(self.arms, tau_table(), SR.mem_fn_from_table(synth.table_default()),
                                  used_profiles={"I1-E": ["q35-4b-ih-w4a16", "q35-2b-fp16"]}, **PILOT)

    def test_selected_set_has_a_heavy_member(self):
        sel = S.select(self.rows, self.est, self.cfg, "I1-E")
        self.assertEqual(sel["status"], "selected", sel["history"])
        self.assertIn("q35-4b-ih-w4a16|b16|k0", sel["S"])

    def test_replacement_takes_next_candidate_with_smaller_cloud_share(self):
        sel = S.select(self.rows, self.est, self.cfg, "I1-E")
        rep = S.replace_heavy(self.rows, self.est, self.cfg, "I1-E", sel["S"], "q35-4b-ih-w4a16|b16|k0")
        self.assertIsNotNone(rep)
        self.assertNotIn("q35-4b-ih-w4a16|b16|k0", rep["S"])
        self.assertEqual(rep["by"], "q35-2b-fp16|b8|k0")

    def test_incumbent_is_not_replaced(self):
        sel = S.select(self.rows, self.est, self.cfg, "I1-E")
        self.assertIsNone(S.replace_heavy(self.rows, self.est, self.cfg, "I1-E", sel["S"], "I1-E"))

    def test_keep_non_escalating(self):
        kept = S.keep_non_escalating(["I1-E", "q35-4b-ih-w4a16|b16|k1", "q35-2b-fp16|b16|k0"], "I1-E", self.cfg)
        self.assertEqual(kept, ["I1-E", "q35-2b-fp16|b16|k0"])


class RescreeningKappa(unittest.TestCase):
    def test_every_alternative_counts_as_escalating(self):
        arms = synth.arms_rescreen()
        cfg = S.ScreenConfig(arms, tau_table(), SR.mem_fn_from_table(synth.table_default()))
        ids = ["I1-E", "q35-4b-ih-w8a8|b16|kA", "q35-4b-ih-w4a16|b8|kH", "q35-4b-ih-w8a8|b8|k1"]
        self.assertEqual(S.keep_non_escalating(ids, "I1-E", cfg), ["I1-E"])


if __name__ == "__main__":
    unittest.main()
