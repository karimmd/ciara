import json
import math
import os
import tempfile
import unittest

import numpy as np

from ciara import band as BD
from ciara import looks as LK
from ciara import radius as R
from ciara.constants import BUDGET_KEYS, OUTCOMES
from synth import tau_table
from test_looks import MEMBERS, PROFILES, TRUTH_BETTER, TRUTH_SAME, campaign_rows


def block(rnd, window_value, plateau=1.0, outcome="Y_C", node="E1"):
    x = [plateau] * 50
    for k in range(25, 30):
        x[k] = window_value
    return {"node": node, "round": rnd, "arm": "a", "prev": "b", "curves": {o: (list(x) if o == outcome else [plateau] * 50) for o in OUTCOMES}}


class Measurement(unittest.TestCase):
    def test_window_carryover_by_hand(self):
        c = BD.window_carryover([block(1, 2.0)], "Y_C")
        self.assertAlmostEqual(float(c[0]), 0.2)
        self.assertAlmostEqual(float(BD.window_carryover([block(1, 1.0)], "Y_C")[0]), 0.0)

    def test_a_block_without_a_plateau_or_a_window_has_no_value(self):
        b = block(1, 2.0)
        b["curves"]["Y_C"][30:] = [None] * 20
        self.assertTrue(np.isnan(BD.window_carryover([b], "Y_C")[0]))
        b = block(1, 2.0)
        b["curves"]["Y_C"][25:30] = [None] * 5
        self.assertTrue(np.isnan(BD.window_carryover([b], "Y_C")[0]))

    def test_clustered_standard_error_by_hand(self):
        v = np.array([0.1, 0.3, -0.1, 0.2, 0.0])
        m, se, n, g = BD.clustered_mean_se(v, [1, 2, 3, 4, 5])
        self.assertAlmostEqual(m, 0.1)
        self.assertAlmostEqual(se, math.sqrt(5 / 4 * float(np.sum((v - 0.1) ** 2))) / 5)
        self.assertEqual((n, g), (5, 5))
        v2 = np.array([0.1, 0.1, 0.3, 0.3, -0.1, -0.1, 0.2, 0.2, 0.0, 0.0])
        m2, se2, n2, g2 = BD.clustered_mean_se(v2, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        self.assertAlmostEqual(m2, 0.1)
        self.assertAlmostEqual(se2, math.sqrt(5 / 4 * float(np.sum((2 * (v - 0.1)) ** 2))) / 10)
        self.assertEqual((n2, g2), (10, 5))

    def test_too_few_blocks_or_rounds_give_no_band(self):
        self.assertEqual(BD.clustered_mean_se(np.array([0.1, 0.2, 0.3, 0.4]), [1, 2, 3, 4])[0], None)
        self.assertEqual(BD.clustered_mean_se(np.array([0.1] * 8), [1, 1, 1, 1, 2, 2, 2, 2])[0], None)
        res = BD.summarize([block(r, 2.0) for r in range(1, 4)])
        self.assertIsNone(res["outcomes"]["Y_C"]["upper"])
        self.assertEqual(BD.bands_of(res), {})

    def test_upper_limit_is_absolute_mean_plus_1_96_se(self):
        vals = [-0.5, -0.3, -0.7, -0.4, -0.6]
        blocks = [block(r, 1.0 + 5.0 * v) for r, v in enumerate(vals, 1)]
        res = BD.summarize(blocks)["outcomes"]["Y_C"]
        v = np.array(vals)
        se = math.sqrt(5 / 4 * float(np.sum((v - v.mean()) ** 2))) / 5
        self.assertAlmostEqual(res["mean"], -0.5)
        self.assertAlmostEqual(res["upper"], 0.5 + 1.96 * se)
        self.assertAlmostEqual(BD.bands_of(BD.summarize(blocks))["Y_C"], 0.5 + 1.96 * se)

    def test_end_to_end_on_synthetic_logs(self):
        with tempfile.TemporaryDirectory() as d:
            ep = 30 * 10 ** 9
            t0 = 10 ** 12
            arms = {1: "x", 2: "y", 3: "y", 4: "x", 5: "y", 6: "x", 7: "y", 8: "x"}
            with open(os.path.join(d, "epochs.jsonl"), "w") as fe, open(os.path.join(d, "blocks.jsonl"), "w") as fb, open(os.path.join(d, "requests.jsonl"), "w") as fr:
                for rnd, arm in arms.items():
                    fb.write(json.dumps({"rec": "block", "node": "E1", "block_id": rnd, "arm_id": arm}) + "\n")
                    fb.write(json.dumps({"rec": "block_end", "node": "E1", "block_id": rnd, "status": "complete", "epochs_logged": 50}) + "\n")
                    for k in range(50):
                        e = rnd * 50 + k
                        t = t0 + e * ep
                        fe.write(json.dumps({"rec": "epoch", "node": "E1", "epoch": e, "block_id": rnd, "epoch_in_block": k, "t_start_mono_ns": t}, separators=(",", ":")) + "\n")
                        fr.write(json.dumps({"node": "E1", "t_arrival": t + 5 * 10 ** 9, "family": 1, "class": "A", "q": 0.5, "exceed": 0.0, "deadline_violation": 0.0, "e_norm": 0.1,
                                             "cloud_used": 25 <= k < 30}) + "\n")
            res = BD.measure(d, os.path.join(d, "requests.jsonl"))
            self.assertEqual(res["n_switch_blocks"], 6)
            o = res["outcomes"]["Y_C"]
            self.assertAlmostEqual(o["mean"], 1.0 * 5 / 25)
            self.assertAlmostEqual(o["se"], 0.0)
            self.assertAlmostEqual(o["upper"], 0.2)
            self.assertAlmostEqual(res["outcomes"]["Y_Q"]["mean"], 0.0)
            self.assertAlmostEqual(res["outcomes"]["Y_E_A"]["upper"], 0.0)
            self.assertIsNone(res["outcomes"]["Y_E_B"]["upper"] if res["outcomes"]["Y_E_B"]["n_blocks"] else None)
            r2 = BD.measure(d, os.path.join(d, "requests.jsonl"), k_rounds=6)
            self.assertEqual(r2["n_switch_blocks"], 3)


def model_band(tau=4.0):
    return R.beta_dm(5.0, 25, 50, tau)


def measured(upper):
    return {"n_switch_blocks": 100, "n_rounds": 50, "outcomes": {k: {"mean": 0.0, "se": 0.0, "upper": upper(k), "n_blocks": 100, "n_rounds": 50} for k in OUTCOMES}}


class Sensitivity(unittest.TestCase):
    def decide(self, truth, n, seed, day, **kw):
        rows = campaign_rows(truth, n, seed=seed)
        return LK.look_decision(rows, MEMBERS, "I1-E", PROFILES, tau_table(), {}, day, k_rounds=n, **kw)

    def test_radius_with_a_one_dimensional_or_a_per_outcome_band_is_the_same(self):
        rng = np.random.default_rng(1)
        Z = rng.integers(0, 3, size=(40, 4))
        F = rng.random((40, 4, 8))
        a = R.c1r_intervals(Z, F, 4, 6.0, [40], [20.0, 20.0, 20.0], [0.003, 0.003, 0.003], n_arms=3)
        b = R.c1r_intervals(Z, F, 4, 6.0, [40], [20.0, 20.0, 20.0], np.full((8, 3), 0.003), n_arms=3)
        self.assertTrue(np.array_equal(a["radius"], b["radius"]))
        c = R.c1r_intervals(Z, F, 4, 6.0, [40], [20.0, 20.0, 20.0], [0.003, 0.003, 0.003], n_arms=3)
        self.assertTrue(np.array_equal(a["radius"], c["radius"]))

    def test_measured_band_equal_to_the_model_band_reproduces_the_decision(self):
        for truth, seed in ((TRUTH_BETTER, 3), (TRUTH_SAME, 4)):
            base = self.decide(truth, 403, seed, 7)
            dec = self.decide(truth, 403, seed, 7, measured_band=measured(lambda k: model_band(base["tau_max"])))
            m = dec["measured_band"]
            self.assertEqual(dec["switch"], base["switch"])
            self.assertEqual(dec["pi"], base["pi"])
            self.assertEqual(dec["slack"], base["slack"])
            self.assertTrue(m["same_decision"])
            self.assertEqual(m["switch"], base["switch"])
            self.assertEqual(m["pi"], base["pi"])
            self.assertEqual(m["slack"], base["slack"])
            self.assertEqual(m["quality_cost_bound"], base["quality_cost_bound"])
            self.assertEqual(m["reference_band_kept_for"], [])
            for a, t in dec["arms_table"].items():
                for k in BUDGET_KEYS:
                    self.assertAlmostEqual(m["uppers"][a][k], t["outcomes"][k]["upper"], places=12)

    def test_the_sensitivity_does_not_change_the_decision(self):
        plain = self.decide(TRUTH_BETTER, 403, 3, 7)
        with_m = self.decide(TRUTH_BETTER, 403, 3, 7, measured_band=measured(lambda k: 0.5))
        for key in ("switch", "reason", "pi", "slack", "quality_cost_bound", "eta_dep", "stop_campaign"):
            self.assertEqual(plain[key], with_m[key], key)
        self.assertEqual(plain["arms_table"], with_m["arms_table"])
        self.assertFalse(with_m["measured_band"]["deciding"])

    def test_a_larger_band_moves_the_upper_bounds_and_margins_by_the_difference(self):
        base = self.decide(TRUTH_BETTER, 403, 3, 7)
        reg = model_band(base["tau_max"])
        big = 0.02
        dec = self.decide(TRUTH_BETTER, 403, 3, 7, measured_band=measured(lambda k: big))
        m = dec["measured_band"]
        delta = big - reg
        for a, t in dec["arms_table"].items():
            for k in BUDGET_KEYS:
                self.assertAlmostEqual(m["uppers"][a][k] - t["outcomes"][k]["upper"], delta, places=10)
        for k, v in m["incumbent_margin"].items():
            self.assertAlmostEqual(base["budgets"][k] - base["incumbent_upper_bounds"][k] - v, delta, places=10)
        self.assertEqual(m["bands"]["Y_C"], big)
        if base["slack"] and m["slack"]:
            for k in base["slack"]:
                self.assertLessEqual(m["slack"][k], base["slack"][k] + 1e-9)
        if base["quality_cost_bound"] is not None and m["quality_cost_bound"] is not None and base["pi"] == m["pi"]:
            self.assertGreaterEqual(m["quality_cost_bound"], base["quality_cost_bound"] - 1e-12)

    def test_an_outcome_without_a_measurement_keeps_the_model_band(self):
        base = self.decide(TRUTH_BETTER, 403, 3, 7)
        res = measured(lambda k: 0.03)
        res["outcomes"]["Y_C"]["upper"] = None
        res["outcomes"]["Y_E_B"]["upper"] = None
        dec = self.decide(TRUTH_BETTER, 403, 3, 7, measured_band=res)
        m = dec["measured_band"]
        self.assertEqual(m["reference_band_kept_for"], ["Y_E_B", "Y_C"])
        self.assertAlmostEqual(m["bands"]["Y_C"], model_band(base["tau_max"]))
        for a, t in dec["arms_table"].items():
            self.assertAlmostEqual(m["uppers"][a]["Y_C"], t["outcomes"]["Y_C"]["upper"], places=12)
            self.assertAlmostEqual(m["uppers"][a]["Y_D"] - t["outcomes"]["Y_D"]["upper"], 0.03 - model_band(base["tau_max"]), places=10)


if __name__ == "__main__":
    unittest.main()
