import math
import random
import unittest

from ciara import contraction as B

N_EP = 13 * 40


def synthetic(kind, seed=1, noise=0.5, deterministic=False):
    rng = random.Random(seed)
    s = []
    for e in range(N_EP):
        b, k = divmod(e, 40)
        heavy = b in B.hold_blocks()["H"]
        if kind == "stationary":
            level = (10.0 if heavy else 2.0)
            prev = 2.0 if heavy else 10.0
            level = level + (prev - level) * (0.25 ** (k + 1)) if b > 0 else level
        elif kind == "ramp":
            level = (2.0 + 0.5 * k) if heavy else (2.0 if k > 2 else 12.0 - 3 * k)
        else:
            level = 5.0
        s.append(level + (0.0 if deterministic else rng.gauss(0, noise)))
    return s


class Series(unittest.TestCase):
    def test_hold_layout(self):
        h = B.hold_blocks()
        self.assertEqual(h["H"], [1, 3, 5, 7, 9, 11])
        self.assertEqual(h["L"], [0, 2, 4, 6, 8, 10, 12])
        sched = B.context_schedule()
        self.assertEqual([s["context_id"] for s in sched["segments"]], ["L"] + ["H", "L"] * 6)

    def test_plateau_test_false_failure_rate(self):
        fails = sum(1 for sd in range(400) if not B.plateau_test(synthetic("stationary", seed=sd))["pass"])
        self.assertTrue(0.005 < fails / 400 < 0.07, fails / 400)

    def test_stationary_passes(self):
        res = B.verdict({k: synthetic("stationary", seed=i, deterministic=True) for i, k in enumerate(B.SERIES)})
        self.assertTrue(res["pass"], res["failed"])
        self.assertLessEqual(res["t_bar_joint_max"], 5)

    def test_ramp_fails_the_plateau_test(self):
        res = B.verdict({"cloud_queue": synthetic("ramp"), "mean_latency_s": synthetic("flat"), "tail_exceedance": synthetic("flat", seed=3)})
        self.assertFalse(res["pass"])
        self.assertIn("cloud_queue", res["failed"])
        self.assertFalse(res["series"]["cloud_queue"]["plateau"]["pass"])

    def test_slow_decay_fails_tbar(self):
        s = []
        rng = random.Random(4)
        for e in range(N_EP):
            b, k = divmod(e, 40)
            heavy = b in B.hold_blocks()["H"]
            s.append((10.0 if heavy else 2.0) + (8.0 * (0.9 ** k) * (1 if heavy else -1) if b > 0 else 0) * -1 + rng.gauss(0, 0.1))
        r = B.tbar_after_switch(s, "H")
        self.assertIsNotNone(r["t_bar"])
        self.assertGreater(r["t_bar"], 5)

    def test_short_transient_passes_settling(self):
        for sd in range(5):
            res = B.verdict({k: synthetic("stationary", seed=10 * sd + i) for i, k in enumerate(B.SERIES)}, settle_mult=B.SETTLE_SE_MULT)
            for k in B.SERIES:
                self.assertTrue(res["series"][k]["settle_into_heavy"]["pass"], (sd, k, res["series"][k]["settle_into_heavy"]))
                self.assertTrue(res["series"][k]["settle_out_of_heavy"]["pass"], (sd, k))

    def long_transient(self, seed=3, tau=12.0):
        rng = random.Random(seed)
        s = []
        for e in range(N_EP):
            b, k = divmod(e, 40)
            heavy = b in B.hold_blocks()["H"]
            base = 10.0 if heavy else 2.0
            gap = (-8.0 if heavy else 8.0) if b > 0 else 0.0
            s.append(base + gap * math.exp(-k / tau) + rng.gauss(0, 0.3))
        return s

    def test_long_transient_fails_settling(self):
        r = B.settle_test(self.long_transient(), "H")
        self.assertFalse(r["pass"], r)
        self.assertFalse(B.settle_test(self.long_transient(), "L")["pass"])
        res = B.verdict({k: self.long_transient(seed=i) for i, k in enumerate(B.SERIES)})
        self.assertFalse(res["pass"])
        self.assertEqual(res["failed"], list(B.SERIES))

    def test_old_statistic_is_reported_not_deciding(self):
        res = B.verdict({k: synthetic("stationary", seed=i) for i, k in enumerate(B.SERIES)})
        for k in B.SERIES:
            self.assertIn("t_bar", res["series"][k]["t_bar_into_heavy"])
            self.assertEqual(res["series"][k]["pass"], res["series"][k]["plateau"]["pass"] and res["series"][k]["settle_into_heavy"]["pass"] and res["series"][k]["settle_out_of_heavy"]["pass"])
        self.assertIn("pass_t_bar_reading", res)

    def stationary_noise(self, seed):
        rng = random.Random(seed)
        return {k: [5.0 + rng.gauss(0, 1.0) for _ in range(N_EP)] for k in B.SERIES}

    def test_shuffled_stationary_series_fail_at_the_calibrated_rate(self):
        base = self.stationary_noise(7)
        cal = B.calibrate_settle(base, n_shuffles=600, seed=5)
        self.assertGreater(cal["false_failure_joint_at_base"], 0.08)
        self.assertGreaterEqual(cal["smallest_threshold"], 3.5)
        self.assertLessEqual(cal["false_failure_joint_at_used"], 0.05)
        thr = cal["threshold_used"]
        runs = 400
        fails = sum(1 for sd in range(runs) if max(B.settle_stats(self.stationary_noise(1000 + sd))) > thr)
        self.assertLess(fails / runs, 0.08)
        self.assertGreater(sum(1 for sd in range(runs) if max(B.settle_stats(self.stationary_noise(1000 + sd))) > 3.0) / runs, 0.08)

    def test_calibration_keeps_the_base_threshold_when_it_meets_the_rate(self):
        rng = random.Random(2)
        cyc = 40
        n = (1 + 2 * cyc) * 40
        ser = {k: [5.0 + rng.gauss(0, 1.0) for _ in range(n)] for k in B.SERIES}
        cal = B.calibrate_settle(ser, n_shuffles=400, seed=1, cycles=cyc)
        self.assertLessEqual(cal["false_failure_joint_at_base"], 0.05)
        self.assertEqual(cal["threshold_used"], 3.0)
        self.assertFalse(cal["raised"])

    def test_missing_epochs_are_handled(self):
        s = synthetic("stationary", deterministic=True)
        for i in (45, 46, 47):
            s[i] = None
        res = B.verdict({k: s for k in B.SERIES})
        self.assertTrue(res["pass"])

    def test_epoch_series(self):
        t0, ep = 1000, 30
        reqs = [{"epoch": 0, "ok": True, "latency_s": 2.0, "exceed": 0.0}, {"epoch": 0, "ok": True, "latency_s": 4.0, "exceed": 1.0}, {"epoch": 1, "ok": False, "latency_s": None, "exceed": 1.0}]
        poll = [{"mono_ns": t0 + 5, "cloud_gw": {"queue": 3, "inflight": 2}}, {"mono_ns": t0 + 10, "cloud_gw": {"queue": 5, "inflight": 2}}, {"mono_ns": t0 + 40, "cloud_gw": {"queue": 1, "inflight": 0}}]
        s = B.epoch_series(reqs, poll, t0, ep, 3)
        self.assertEqual(s["mean_latency_s"], [3.0, None, None])
        self.assertEqual(s["tail_exceedance"], [0.5, 1.0, None])
        self.assertEqual(s["cloud_queue"], [6.0, 1.0, None])

    def test_pick_vectors(self):
        sel = {"S": ["I1-E", "a", "b"], "screened_means": {"I1-E": {"Y_C": 0.47, "Y_L_f1": 0.02, "Y_L_f2": 0.03}, "a": {"Y_C": 0.0, "Y_L_f1": 0.01, "Y_L_f2": 0.02}, "b": {"Y_C": 0.0, "Y_L_f1": 0.01, "Y_L_f2": 0.01}}}
        v = B.pick_vectors(sel)
        self.assertEqual(v["heavy"], "I1-E")
        self.assertEqual(v["light"], "b")


if __name__ == "__main__":
    unittest.main()
