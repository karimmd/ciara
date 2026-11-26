import os
import random
import sys
import unittest

from ciara_monitor import nodecheck as NC

MEMBERS = ["A0", "B", "C"]
NODES = ["E1", "E2", "E3", "E4"]


def make_rows(K, rng, arm_effect=None, node_effect=None, inter=None, throttle=None, noise=1.0):
    rows = []
    for b in range(K):
        comp = ["A0", "A0", "B", "C"]
        rng.shuffle(comp)
        for n, arm in zip(NODES, comp):
            th = rng.random() * 0.5 if n == "E2" else 0.0
            v = rng.gauss(0.0, noise) + (arm_effect or {}).get(arm, 0.0) + (node_effect or {}).get(n, 0.0) + (inter or {}).get((arm, n), 0.0) + (throttle or 0.0) * th
            rows.append({"node": n, "block_id": b, "arm_id": arm, "window": "measure", "block_status": "complete", "Y_C": v, "Y_L_f1": rng.gauss(0, noise), "gpu_throttled_frac": th})
    return rows


class Tests(unittest.TestCase):
    def test_f_by_hand_on_a_tiny_design(self):
        rows = [{"node": n, "block_id": i, "arm_id": "A", "Y_C": v + i} for n, v in (("E1", 0.0), ("E2", 2.0), ("E3", 0.0)) for i in range(3)]
        t = NC.nested_f(rows, "Y_C")
        ssb = 3 * ((1 - 5 / 3) ** 2 * 2 + (3 - 5 / 3) ** 2)
        self.assertAlmostEqual(t["node"]["F"], (ssb / 2) / (6 / 6), places=9)
        self.assertEqual(t["node"]["df"], [2, 6])

    def test_a_node_effect_is_found_and_no_effect_is_not(self):
        rng = random.Random(1)
        r = NC.node_check(make_rows(120, rng, node_effect={"E2": 0.8}), MEMBERS, "A0", 120)
        self.assertIn("Y_C", r["flags"]["node_effect"])
        self.assertNotIn("Y_L_f1", r["flags"]["node_effect"])
        r0 = NC.node_check(make_rows(120, random.Random(2)), MEMBERS, "A0", 120)
        self.assertEqual(r0["flags"]["node_effect"], [])
        self.assertEqual(r0["flags"]["interaction"], [])

    def test_an_interaction_is_found_and_an_additive_node_effect_is_not_an_interaction(self):
        r = NC.node_check(make_rows(150, random.Random(3), inter={("B", "E3"): 1.5}), MEMBERS, "A0", 150)
        self.assertIn("Y_C", r["flags"]["interaction"])
        r2 = NC.node_check(make_rows(150, random.Random(4), node_effect={"E3": 1.0}), MEMBERS, "A0", 150)
        self.assertEqual(r2["flags"]["interaction"], [])
        self.assertIn("Y_C", r2["flags"]["node_effect"])

    def test_false_positive_rate_of_the_unadjusted_tests_is_near_five_percent(self):
        hits = [0, 0]
        reps = 150
        for s in range(reps):
            t = NC.nested_f(make_rows(40, random.Random(100 + s)), "Y_C")
            hits[0] += t["node"]["p"] < 0.05
            hits[1] += t["interaction"]["p"] < 0.05
        for h in hits:
            self.assertLess(abs(h / reps - 0.05), 0.045)

    def test_throttle_slope_and_share_by_node(self):
        r = NC.node_check(make_rows(150, random.Random(5), throttle=2.0), MEMBERS, "A0", 150)
        self.assertIn("Y_C", r["flags"]["throttle_slope"])
        self.assertGreater(r["outcomes"]["Y_C"]["throttle_slope"]["slope"], 1.0)
        self.assertGreater(r["throttle_by_node"]["E2"]["mean_throttled_share"], 0.15)
        self.assertEqual(r["throttle_by_node"]["E1"]["mean_throttled_share"], 0.0)

    def test_leave_one_node_out_changes_the_contrast_by_hand(self):
        rows = [{"node": n, "block_id": b, "arm_id": a, "Y_C": v} for b, (n, a, v) in enumerate(
            [("E1", "A0", 0.5), ("E1", "B", 0.3), ("E2", "A0", 0.5), ("E2", "B", 0.3), ("E3", "A0", 0.5), ("E3", "B", 0.1), ("E4", "A0", 0.5), ("E4", "B", 0.1)])]
        lo = NC.leave_one_node_out(rows, ["A0", "B"], "A0", "Y_C")
        c = lo["contrasts"]["B"]
        self.assertAlmostEqual(c["full"], 0.2 - 0.5, places=12)
        self.assertAlmostEqual(c["change_without"]["E3"], ((0.3 + 0.3 + 0.1) / 3 - 0.5) - (0.2 - 0.5), places=12)
        self.assertEqual(c["worst_node"] in ("E1", "E2", "E3", "E4"), True)

    def test_binding_rows_move_flag_and_report_line(self):
        rows = make_rows(100, random.Random(6), inter={("B", "E4"): 3.0, ("A0", "E4"): -3.0})
        r = NC.node_check(rows, MEMBERS, "A0", 100)
        self.assertIn("B", r["flags"]["binding_rows_move"]["Y_C"])
        md = NC.render_md(r)
        self.assertIn("binding row of the certified program", md)

    def test_incomplete_blocks_and_other_windows_are_left_out(self):
        rows = make_rows(20, random.Random(7))
        rows.append(dict(rows[0], block_status="partial", Y_C=99.0))
        rows.append(dict(rows[0], window="burn", Y_C=99.0))
        wr = NC.window_rows(rows, MEMBERS, 20)
        self.assertEqual(len(wr), 80)


if __name__ == "__main__":
    unittest.main()
