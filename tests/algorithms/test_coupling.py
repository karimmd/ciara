import json
import math
import os
import tempfile
import unittest

import numpy as np

from ciara import coupling as CPT
import synth
from synth import A0, K1_F, KH_P, MEM3, TRUTH_GO


class Slopes(unittest.TestCase):
    def test_fixed_effects_slope_by_hand(self):
        x = [0, 1, 2, 10, 11, 12]
        y = [0, 2, 4, 5, 7, 9]
        s = CPT.fe_slope(x, y, ["a"] * 3 + ["b"] * 3)
        self.assertAlmostEqual(s["slope"], 2.0)
        self.assertAlmostEqual(s["se"], 0.0, places=9)
        self.assertEqual(s["n_groups"], 2)
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0, 0.0, 3.0, 2.0, 5.0]
        s = CPT.fe_slope(x, y, ["g"] * 5)
        sxx = 10.0
        sxy = sum((a - 2) * (b - 2.2) for a, b in zip(x, y))
        b1 = sxy / sxx
        rss = sum((b - 2.2 - b1 * (a - 2)) ** 2 for a, b in zip(x, y))
        self.assertAlmostEqual(s["slope"], b1)
        self.assertAlmostEqual(s["se"], math.sqrt(rss / 3 / sxx))
        self.assertAlmostEqual(s["slope"], float(np.polyfit(x, y, 1)[0]))
        self.assertAlmostEqual(s["t"], b1 / s["se"])

    def test_the_group_effect_does_not_enter_the_slope(self):
        x = [0, 1, 2, 3, 0, 1, 2, 3]
        y = [0.1 * v for v in x[:4]] + [100 + 0.1 * v for v in x[4:]]
        s = CPT.fe_slope(x, y, ["a"] * 4 + ["b"] * 4)
        self.assertAlmostEqual(s["slope"], 0.1)
        self.assertEqual(CPT.fe_slope([1, 1, 1], [1, 2, 3], ["a"] * 3)["slope"], None)

    def test_the_clustered_error_is_reported_with_the_clusters(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=40)
        y = 0.5 * x + rng.normal(size=40) * 0.3
        s = CPT.fe_slope(x, y, ["g"] * 40, clusters=[i // 4 for i in range(40)])
        self.assertEqual(s["n_clusters"], 10)
        self.assertIsNotNone(s["se_cluster"])
        self.assertAlmostEqual(s["t"], s["slope"] / s["se_cluster"])

    def test_round_queue_from_a_gateway_log(self):
        rows = [{"block_status": "complete", "block_id": b, "t_lo_mono_ns": 1000 * b + 100, "t_hi_mono_ns": 1000 * b + 900} for b in (0, 1) for _ in range(2)]
        recs = [
            {"rec": "request", "t_arrive": 150, "t_admit": 250, "queue_len_at_arrival": 1, "in_flight_at_arrival": 2},
            {"rec": "request", "t_arrive": 500, "t_admit": 700, "queue_len_at_arrival": 0, "in_flight_at_arrival": 1},
            {"rec": "request", "t_arrive": 50, "t_admit": 60, "queue_len_at_arrival": 9, "in_flight_at_arrival": 9},
            {"rec": "request", "t_arrive": 1500, "t_admit": 1510, "queue_len_at_arrival": 5, "in_flight_at_arrival": 3},
            {"rec": "request", "t_arrive": 950, "t_admit": 960, "queue_len_at_arrival": 7, "in_flight_at_arrival": 7},
            {"rec": "other", "t_arrive": 200},
        ]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cloud-gw.jsonl")
            with open(p, "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            q = CPT.round_queue(p, rows)
            self.assertEqual(sorted(q), [0, 1])
            self.assertAlmostEqual(q[0]["backlog"], (3 + 1) / 2)
            self.assertAlmostEqual(q[0]["wait_s"], (100 + 200) / 2 / 1e9)
            self.assertEqual(q[0]["n"], 2)
            self.assertAlmostEqual(q[1]["backlog"], 8.0)
            self.assertEqual(sorted(CPT.round_queue(p, rows, k_rounds=1)), [0])

    def test_look_slopes_recover_a_known_slope(self):
        rows = []
        queue = {}
        for b in range(30):
            queue[b] = {"backlog": 1.0 + (b % 5), "wait_s": 0.001 * (b % 5), "n": 10}
            for n, arm in enumerate([A0, A0, KH_P, K1_F]):
                base = {A0: 0.2, KH_P: 0.5, K1_F: 0.7}[arm]
                rows.append({"block_status": "complete", "window": "measure", "block_id": b, "node": f"E{n + 1}", "arm_id": arm, "Y_L_f1": base + 0.05 * queue[b]["backlog"],
                             "Y_C": base, "Y_Q": 0.6})
        res = CPT.look_slopes(rows, queue, MEM3)
        pooled = res["by_measure"]["backlog"]["Y_L_f1"]["pooled"]
        self.assertAlmostEqual(pooled["slope"], 0.05)
        self.assertAlmostEqual(res["by_measure"]["backlog"]["Y_L_f1"]["members"][KH_P]["slope"], 0.05)
        self.assertAlmostEqual(res["by_measure"]["backlog"]["Y_C"]["pooled"]["slope"], 0.0)
        self.assertEqual(res["rounds_with_queue"], 30)
        md = CPT.render_slopes_md("t", res, "backlog")
        self.assertIn("Y_L_f1", md)

    def test_the_share_constraint_in_cloud_work(self):
        recs = ([{"rec": "request", "arm_id": "x", "t_admit": 0, "t_done": 2_000_000_000}] * 3 + [{"rec": "request", "arm_id": "y", "t_admit": 10, "t_done": 10 + 3_000_000_000}] * 2 +
                [{"rec": "request", "arm_id": "z", "t_admit": None, "t_done": 5}])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "g.jsonl")
            with open(p, "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            sv = CPT.cloud_service_by_arm(p, ["x", "y", "z"])
        self.assertEqual(sv, {"x": {"service_s": 2.0, "n": 3}, "y": {"service_s": 3.0, "n": 2}})
        tab = {"x": {"propensity": 0.5, "outcomes": {"Y_C": {"centre": 0.4, "upper": 0.45}}}, "y": {"propensity": 0.5, "outcomes": {"Y_C": {"centre": 0.2, "upper": 0.25}}}}
        dec = {"arms_table": tab, "pi": {"x": 0.25, "y": 0.75}}
        w = CPT.work_check(dec, sv)
        self.assertAlmostEqual(w["policy_work_upper"], 0.25 * 0.45 * 2.0 + 0.75 * 0.25 * 3.0)
        self.assertAlmostEqual(w["design_work"], 0.5 * 0.4 * 2.0 + 0.5 * 0.2 * 3.0)
        self.assertAlmostEqual(w["margin"], 0.7 - 0.7875)
        self.assertFalse(w["holds"])
        self.assertFalse(CPT.work_check({"arms_table": tab, "pi": None}, sv)["available"])

    def test_screening_slope_with_arm_fixed_effects(self):
        arms = synth.arms_balanced()
        cl = {a: TRUTH_GO[a]["Y_C"] for a in TRUTH_GO}
        rows = synth.make_rows(TRUTH_GO, seed=5, n_nodes=4, n_blocks=80, arms=arms, coupling={"Y_L_f1": 0.5}, cl=cl)
        res = CPT.screening_slopes(rows)
        s = res["outcomes"]["Y_L_f1"]
        self.assertGreater(s["slope"], 0.3)
        self.assertLess(s["slope"], 0.7)
        self.assertGreater(s["t"], 3.0)
        self.assertLess(abs(res["outcomes"]["Y_Q"]["t"]), 4.0)
        self.assertEqual(res["n_rows"], 320)


if __name__ == "__main__":
    unittest.main()
