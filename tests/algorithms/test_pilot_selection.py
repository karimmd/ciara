import unittest

from ciara import pilot_selection as SR
from ciara import screening as S
from ciara.constants import BUDGETS_TIGHT
import synth
from synth import tau_table

PILOT = {"n_nodes": 5, "budgets": dict(BUDGETS_TIGHT)}


def truth_two_pass():
    t = {}
    for a in synth.arms_default():
        if a["kind"] == "static":
            t[a["id"]] = {"Y_Q": 0.40, "Y_L_f2": 0.2}
            if a["kappa"] == 1:
                t[a["id"]] = {"Y_C": 0.47, "Y_Q": 0.70}
    t["q35-4b-ih-w8a8|b16|k0"] = {"Y_Q": 0.66}
    t["q35-2b-fp16|b16|k0"] = {"Y_Q": 0.64}
    t["q35-4b-ih-w4a16|b16|k0"] = {"Y_Q": 0.65, "Y_L_f2": 0.2}
    t["I1-E"] = {"Y_Q": 0.60, "Y_C": 0.12}
    t["I2"] = {"Y_Q": 0.59, "Y_C": 0.2}
    t["I3"] = {"Y_Q": 0.58, "Y_C": 0.2}
    return t


class Selection(unittest.TestCase):
    def run_select(self, truth, **kw):
        arms = synth.arms_default()
        rows = synth.make_rows(truth, arms=arms, **kw)
        return SR.analyze(rows, arms, tau_table(), synth.table_default(), **PILOT)

    def test_selects_two_alternatives_and_the_incumbent(self):
        res = self.run_select(truth_two_pass())
        self.assertEqual(res["incumbent"]["a0"], "I1-E")
        sel = res["selection"]
        self.assertEqual(sel["status"], "selected", sel["history"])
        self.assertEqual(sel["alternatives"][0], "q35-4b-ih-w8a8|b16|k0")
        self.assertEqual(sel["alternatives"][1], "q35-2b-fp16|b16|k0")
        self.assertEqual(sel["S"][0], "I1-E")
        for c, ev in res["pass_table"].items():
            if c.endswith("|k1"):
                self.assertFalse(ev["pass"])
        self.assertFalse(res["pass_table"]["q35-4b-ih-w4a16|b16|k0"]["pass"])
        self.assertTrue(sel["design_law"]["pass"])

    def test_stops_when_fewer_than_two_pass(self):
        t = truth_two_pass()
        t["q35-2b-fp16|b16|k0"] = {"Y_Q": 0.64, "Y_L_f1": 0.3}
        res = self.run_select(t)
        self.assertEqual(res["selection"]["status"], "fewer_than_two_alternatives")
        self.assertEqual(len(res["selection"]["alternatives"]), 1)

    def test_no_configuration_passes(self):
        t = {}
        for a in synth.arms_default():
            if a["kind"] == "static":
                t[a["id"]] = {"Y_L_f2": 0.2}
        res = self.run_select(t)
        self.assertEqual(res["selection"]["status"], "fewer_than_two_alternatives")
        self.assertEqual(res["selection"]["alternatives"], [])

    def test_tie_goes_to_lower_memory(self):
        t = truth_two_pass()
        t["q35-4b-ih-w8a8|b16|k0"] = {"Y_Q": 0.660}
        t["q35-2b-fp16|b16|k0"] = {"Y_Q": 0.662}
        t["q35-2b-fp16|b8|k0"] = {"Y_Q": 0.600}
        res = self.run_select(t, block_noise=0.2)
        self.assertEqual(res["selection"]["alternatives"][0], "q35-2b-fp16|b16|k0")

    def test_incumbent_ties_go_to_i1(self):
        t = truth_two_pass()
        t["I2"] = {"Y_Q": 0.603, "Y_C": 0.2}
        t["I1-E"] = {"Y_Q": 0.600, "Y_C": 0.12}
        res = self.run_select(t, block_noise=0.1)
        self.assertEqual(res["incumbent"]["a0"], "I1-E")

    def test_incumbent_without_feasible_regime_takes_smallest_violation(self):
        t = truth_two_pass()
        t["I1-E"] = {"Y_Q": 0.7, "Y_L_f2": 0.30}
        t["I2"] = {"Y_Q": 0.5, "Y_L_f2": 0.08}
        t["I3"] = {"Y_Q": 0.5, "Y_L_f2": 0.20}
        res = self.run_select(t)
        self.assertEqual(res["incumbent"]["a0"], "I2")

    def test_design_law_check_skips_an_alternative(self):
        t = truth_two_pass()
        t["I1-E"] = {"Y_Q": 0.60, "Y_C": 0.29}
        t["q35-2b-fp16|b16|k0"] = {"Y_Q": 0.64, "Y_C": 0.27}
        res = self.run_select(t)
        self.assertIn(res["selection"]["status"], ("selected", "fewer_than_two_alternatives"))


class SetDependentSelection(unittest.TestCase):
    def test_set_dependent_loop_removes_a_failing_member(self):
        arms = synth.arms_default()
        t = {}
        for a in arms:
            if a["kind"] == "static":
                t[a["id"]] = {"Y_Q": 0.40, "Y_L_f2": 0.2}
        t["q35-4b-ih-w4a16|b16|k0"] = {"Y_Q": 0.66, "Y_L_f2": 0.010}
        t["q35-2b-fp16|b16|k0"] = {"Y_Q": 0.64, "Y_L_f2": 0.002}
        t["q35-2b-fp16|b8|k0"] = {"Y_Q": 0.63, "Y_L_f2": 0.002}
        t["I1-E"] = {"Y_Q": 0.60, "Y_C": 0.12}
        t["I2"] = {"Y_Q": 0.50}
        t["I3"] = {"Y_Q": 0.50}
        cl = {a["id"]: (0.47 if (a["kind"] == "static" and a["kappa"] == 1) else 0.0) for a in arms}
        cl["I1-E"] = 0.12
        rows = synth.make_rows(t, arms=arms, seed=2, n_blocks=300, coupling={"Y_L_f2": 0.10}, cl=cl, block_noise=0.2)
        res = SR.analyze(rows, arms, tau_table(), synth.table_default(), **PILOT)
        sel = res["selection"]
        self.assertEqual(sel["status"], "selected", sel["history"])
        self.assertEqual(len(sel["history"]), 2)
        self.assertNotIn("q35-4b-ih-w4a16|b16|k0", sel["alternatives"])
        self.assertGreater(sel["zeta"]["zeta"]["Y_L_f2"], 0.004)


class CloudSwitch(unittest.TestCase):
    def truth(self):
        t = truth_two_pass()
        t["q35-4b-ih-w8a8|b16|k1"] = {"Y_Q": 0.72, "Y_C": 0.47}
        t["q35-2b-fp16|b16|k1"] = {"Y_Q": 0.71, "Y_C": 0.47}
        return t

    def test_per_configuration_form_fails_escalating_configurations(self):
        arms = synth.arms_default()
        res = SR.analyze(synth.make_rows(self.truth(), arms=arms), arms, tau_table(), synth.table_default(), **PILOT)
        self.assertFalse(res["pass_table"]["q35-4b-ih-w8a8|b16|k1"]["pass"])
        self.assertEqual(res["parameters"]["cloud_share_mode"], "per_configuration")

    def test_set_level_waives_the_row_but_keeps_the_design_law(self):
        arms = synth.arms_default()
        res = SR.analyze(synth.make_rows(self.truth(), arms=arms), arms, tau_table(), synth.table_default(), cloud_share_mode="set_level", **PILOT)
        ev = res["pass_table"]["q35-4b-ih-w8a8|b16|k1"]
        self.assertTrue(ev["pass"])
        self.assertTrue(ev["budgets"]["Y_C"]["waived"])
        self.assertLess(ev["budgets"]["Y_C"]["margin"], 0)
        sel = res["selection"]
        self.assertEqual(sel["status"], "selected")
        self.assertEqual(sel["alternatives"][0], "q35-4b-ih-w8a8|b16|k1")
        self.assertTrue(sel["design_law"]["pass"])
        self.assertLessEqual(sel["design_law"]["mixture"]["Y_C"]["value"], 0.30)

    def test_set_level_design_law_rejects_two_escalating_alternatives(self):
        arms = synth.arms_default()
        t = self.truth()
        t["I1-E"] = {"Y_Q": 0.60, "Y_C": 0.28}
        res = SR.analyze(synth.make_rows(t, arms=arms), arms, tau_table(), synth.table_default(), cloud_share_mode="set_level", **PILOT)
        S_ = res["selection"]["S"]
        self.assertEqual(res["selection"]["status"], "selected")
        self.assertFalse(all(m.endswith("k1") for m in S_[1:]))


if __name__ == "__main__":
    unittest.main()
