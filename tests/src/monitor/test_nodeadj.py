import json
import os
import random
import sys
import unittest
from pathlib import Path

try:
    from ciara_monitor import nodeadj as NA
except ImportError as exc:
    raise unittest.SkipTest(f"ciara_monitor.trajectory is not installed: {exc}")
from ciara import looks as LK
from ciara.constants import OUTCOMES
from ciara.transient import TauTable

from test_style import BANNED, DASH

MEMBERS = ["A0", "B", "C"]
NODES = ["E1", "E2", "E3", "E4"]
PROFILES = {"A0": ["q35-4b-ih-w4a16"], "B": ["q35-4b-ih-w8a8"], "C": ["q35-4b-ih-w4a16"]}
ROOT = str(Path(__file__).resolve().parents[3])
TAU = TauTable.load(os.path.join(ROOT, "data", "inputs", "campaign", "switch-transient.json"))
LEVEL = {"A0": {"Y_Q": 0.70, "Y_C": 0.45, "Y_L_f1": 0.02}, "B": {"Y_Q": 0.72, "Y_C": 0.15, "Y_L_f1": 0.04}, "C": {"Y_Q": 0.69, "Y_C": 0.25, "Y_L_f1": 0.03}}
COMP = ["A0", "A0", "B", "C"]


def make_rows(K, rng, node_effect=None, assign=None, noise=0.01):
    rows = []
    for b in range(K):
        comp = assign(b, rng) if assign else COMP[b % 4:] + COMP[:b % 4]
        for i, (n, arm) in enumerate(zip(NODES, comp)):
            r = {"node": n, "node_index": i + 1, "block_id": b, "arm_id": arm, "window": "measure", "block_status": "complete"}
            for o in OUTCOMES:
                r[o] = LEVEL[arm].get(o, 0.05) + (node_effect or {}).get((n, o), 0.0) + rng.gauss(0.0, noise)
            rows.append(r)
    return rows


def run_look_on(rows, K):
    return LK.look_decision(rows, MEMBERS, "A0", PROFILES, TAU, {}, 4, k_rounds=K, composition=COMP, one_sided=True, coupling_mode="a5pp_gated")


class Fit(unittest.TestCase):
    def test_levels_and_contrasts_recover_a_noise_free_additive_design(self):
        arm = {"A0": 1.0, "B": 2.0}
        eff = {"E1": 0.0, "E2": 0.5, "E3": -0.25, "E4": 1.0}
        rows = []
        for b in range(12):
            for n, a in zip(NODES, ["A0", "A0", "B", "A0"] if b % 2 else ["B", "A0", "B", "B"]):
                rows.append({"node": n, "block_id": b, "arm_id": a, "Y_C": arm[a] + eff[n] + 0.001 * ((b * 7 + len(n) * len(a)) % 5 - 2)})
        f = NA.fit_outcome(rows, ["A0", "B"], "A0", "Y_C", True)
        self.assertTrue(f["ok"])
        self.assertAlmostEqual(f["levels"]["A0"]["value"], 1.0 + 0.3125, delta=0.003)
        self.assertAlmostEqual(f["levels"]["B"]["value"], 2.0 + 0.3125, delta=0.003)
        self.assertAlmostEqual(f["contrasts"]["B"]["value"], 1.0, delta=0.003)
        g = NA.fit_outcome(rows, ["A0", "B"], "A0", "Y_C", False)
        self.assertGreater(abs(g["contrasts"]["B"]["value"] - 1.0), 0.05)

    def test_rank_deficiency_and_few_rounds_are_reported_not_raised(self):
        rows = [{"node": "E1", "block_id": b, "arm_id": "A0", "Y_C": 0.1} for b in range(5)]
        self.assertFalse(NA.fit_outcome(rows, ["A0", "B"], "A0", "Y_C")["ok"])
        rows2 = [{"node": n, "block_id": b, "arm_id": a, "Y_C": 0.1 * i} for b in range(2) for i, (n, a) in enumerate(zip(NODES, COMP))]
        self.assertFalse(NA.fit_outcome(rows2, MEMBERS, "A0", "Y_C")["ok"])


class Sensitivity(unittest.TestCase):
    K = 240

    @classmethod
    def setUpClass(cls):
        rng = random.Random(11)
        cls.rows0 = make_rows(cls.K, rng)
        cls.look0 = run_look_on(cls.rows0, cls.K)

    def test_without_a_node_effect_both_estimates_agree(self):
        look = self.look0
        self.assertTrue(look["switch"], look["reason"])
        res = NA.node_adjusted(self.rows0, look, MEMBERS, "A0", self.K)
        pr = res["program"]
        self.assertTrue(pr["computed"])
        self.assertTrue(pr["reference_reproduces_look"], "the reference program recomputed from the look file equals the stored decision")
        self.assertTrue(pr["same_decision"])
        self.assertLess(pr["max_abs_weight_change"], 0.01)
        for o in OUTCOMES:
            for a, c in res["outcomes"][o]["arms"].items():
                self.assertAlmostEqual(c["adjusted"], c["look_centre"], delta=5e-4, msg=(o, a))
        self.assertLess(res["flags"]["largest_center_shift_in_se"]["se"], 1.0)

    def test_a_node_effect_with_unbalanced_assignment_is_removed(self):
        def assign(b, rng):
            others = ["A0", "A0", "C"]
            rng.shuffle(others)
            return [others[0], "B", others[1], others[2]] if rng.random() < 0.8 else ["B", others[0], others[1], others[2]]
        K = 280
        rows = make_rows(K, random.Random(12), node_effect={("E2", "Y_C"): 0.12}, assign=assign)
        look = LK.look_decision(rows, MEMBERS, "A0", PROFILES, TAU, {}, 4, k_rounds=K, composition=COMP, one_sided=True, coupling_mode="a5pp_gated")
        res = NA.node_adjusted(rows, look, MEMBERS, "A0", K)
        c = res["outcomes"]["Y_C"]["contrasts"]["B"]
        truth = LEVEL["B"]["Y_C"] - LEVEL["A0"]["Y_C"]
        self.assertLess(abs(c["adjusted"] - truth), 4 * c["se"] + 0.01)
        self.assertGreater(abs(c["arm_only"] - truth), 0.05)
        self.assertGreater(abs(c["look_contrast"] - truth), 0.05)
        self.assertGreater(abs(c["shift_in_se"]), 5.0)
        self.assertTrue(res["flags"]["largest_contrast_shift_in_se"]["se"] > 5.0)
        quiet = res["outcomes"]["Y_L_f1"]["contrasts"]["B"]
        self.assertLess(abs(quiet["shift_in_se"]), 6.0)
        self.assertLess(abs(quiet["adjusted"] - (LEVEL["B"]["Y_L_f1"] - LEVEL["A0"]["Y_L_f1"])), 0.01)

    def test_the_program_on_adjusted_centers_uses_the_reference_radii(self):
        res = NA.node_adjusted(self.rows0, self.look0, MEMBERS, "A0", self.K)
        dec = NA.adjusted_decision(self.look0, {"Y_C": {"B": 0.5}})
        t = dec["arms_table"]["B"]["outcomes"]["Y_C"]
        self.assertEqual(t["centre"], 0.5)
        self.assertEqual(t["radius"], self.look0["arms_table"]["B"]["outcomes"]["Y_C"]["radius"])
        self.assertAlmostEqual(t["upper"], 0.5 + t["radius"])
        self.assertEqual(self.look0["arms_table"]["B"]["outcomes"]["Y_C"]["centre"] == 0.5, False)
        shifted = NA.gated_program(dec, MEMBERS, "A0", {}, None)
        self.assertNotEqual(shifted["pi"], res["program"]["reference"]["pi"])

    def test_contrast_intervals_use_the_sum_of_the_reference_radii(self):
        res = NA.node_adjusted(self.rows0, self.look0, MEMBERS, "A0", self.K)
        c = res["outcomes"]["Y_C"]["contrasts"]["B"]
        tab = self.look0["arms_table"]
        half = tab["B"]["outcomes"]["Y_C"]["radius"] + tab["A0"]["outcomes"]["Y_C"]["radius"]
        self.assertAlmostEqual(c["half_width"], half)
        self.assertAlmostEqual(c["adjusted_interval"][1] - c["adjusted_interval"][0], 2 * half)
        self.assertAlmostEqual(c["reference_interval"][0], c["look_contrast"] - half)

    def test_a_failed_fit_leaves_the_program_not_computed(self):
        rows = [r for r in self.rows0 if r["arm_id"] != "C"]
        res = NA.node_adjusted(rows, self.look0, MEMBERS, "A0", self.K)
        self.assertFalse(res["program"]["computed"])
        self.assertFalse(res["flags"]["program_computed"])
        self.assertIn("not computed", NA.render_text(res["flags"]))
        self.assertEqual(res["failed_outcomes"], list(OUTCOMES))
        self.assertEqual(DASH.findall(NA.render_md(res)), [])

    def test_mock_stamp_report_line_and_writing_rules(self):
        res = NA.node_adjusted(self.rows0, self.look0, MEMBERS, "A0", self.K, mock=True)
        md = NA.render_md(res)
        self.assertIn("**MOCK:**", md)
        self.assertNotIn("**MOCK:**", NA.render_md(NA.node_adjusted(self.rows0, self.look0, MEMBERS, "A0", self.K, mock=False)))
        line = NA.render_text(res["flags"])
        self.assertIn("MOCK", line)
        self.assertIn("same decision True", line)
        self.assertIn("not computed", NA.render_text({"error": "x"}))
        self.assertIn("not computed", NA.render_text({}))
        for text in (md, line, json.dumps(res["flags"])):
            self.assertEqual(BANNED.findall(text), [])
            self.assertEqual(DASH.findall(text), [])

    def test_flags_are_json_serializable_and_complete(self):
        res = NA.node_adjusted(self.rows0, self.look0, MEMBERS, "A0", self.K)
        fl = json.loads(json.dumps(res["flags"]))
        for k in ("reference_switch", "adjusted_switch", "same_decision", "reference_pi", "adjusted_pi", "reference_t_star", "adjusted_t_star", "binding_contrast_shift_in_se"):
            self.assertIn(k, fl)
        json.dumps(res)


if __name__ == "__main__":
    unittest.main()
