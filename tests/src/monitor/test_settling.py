import json
import math
import os
import random
import sys
import tempfile
import unittest

import numpy as np

from ciara_monitor import settling as ST

ARMS = ["I1-E", "c32|f1prio", "c32|fifo"]


def synth_blocks(n_rounds, tau, amp, rng, noise=1.0, transient=True, nodes=("E1", "E2", "E3", "E4")):
    out = []
    for rd in range(1, n_rounds + 1):
        for n in nodes:
            arm = rng.choice(ARMS)
            prev = rng.choice([a for a in ARMS if a != arm])
            cur = {}
            for s in ST.SERIES:
                base = rng.gauss(0, 0.3)
                cur[s] = [base + (amp * math.exp(-k / tau) if transient else 0.0) + rng.gauss(0, noise) for k in range(ST.BLOCK_EPOCHS)]
            out.append({"node": n, "round": rd, "arm": arm, "prev": prev, "profile_switched": True, "curves": cur})
    return out


class Deviations(unittest.TestCase):
    def test_deviation_is_from_the_plateau_of_the_block(self):
        x = [5.0] * 30 + [1.0] * 20
        x[0] = 9.0
        b = {"curves": {s: list(x) for s in ST.SERIES}}
        D = ST.deviations([b], "cloud_queue")
        self.assertAlmostEqual(D[0, 0], 8.0)
        self.assertAlmostEqual(D[0, 5], 4.0)
        self.assertAlmostEqual(D[0, 40], 0.0)

    def test_a_block_without_a_plateau_is_dropped(self):
        x = [1.0] * 30 + [None] * 20
        D = ST.deviations([{"curves": {s: list(x) for s in ST.SERIES}}], "cloud_share")
        self.assertTrue(np.isnan(D).all())

    def test_settled_epoch_by_hand(self):
        m = np.zeros(30)
        se = np.ones(30)
        self.assertEqual(ST.settled_epoch(m, se, 4.08), 0)
        m[0], m[1], m[3] = 10.0, 6.0, 5.0
        self.assertEqual(ST.settled_epoch(m, se, 4.08), 4)
        m[29] = 9.0
        self.assertIsNone(ST.settled_epoch(m, se, 4.08))
        m2 = np.zeros(30)
        m2[2] = 3.5
        self.assertEqual(ST.settled_epoch(m2, se, 4.08), 0)
        self.assertEqual(ST.settled_epoch(m2, se, 3.0), 3)


class Fit(unittest.TestCase):
    def check(self, tau, amp=1.5, n_rounds=100, seed=3):
        rng = random.Random(seed)
        res = ST.analyze(synth_blocks(n_rounds, tau, amp, rng), reps=120, seed=seed)
        return res

    def test_fit_recovers_known_t_bar_and_the_interval_covers_it(self):
        for tau in (1.0, 3.0, 8.0):
            res = self.check(tau)
            lo, hi = res["t_bar_ci"]
            self.assertTrue(res["identified"], tau)
            self.assertLess(abs(res["t_bar"] - tau), 0.25 * tau + 0.2, (tau, res["t_bar"]))
            self.assertLessEqual(lo, tau * 1.05, (tau, lo, hi))
            self.assertGreaterEqual(hi, tau * 0.95, (tau, lo, hi))

    def test_decision_needed_follows_the_upper_limit_against_five(self):
        self.assertFalse(self.check(1.0)["decision_needed"])
        self.assertTrue(self.check(8.0)["decision_needed"])

    def test_settled_epoch_grows_with_t_bar(self):
        fast = self.check(1.0)["settled_epoch_pooled"]
        slow = self.check(6.0)["settled_epoch_pooled"]
        for s in ST.SERIES:
            self.assertLessEqual(fast[s], 6, s)
            self.assertGreater(slow[s], fast[s], s)
            self.assertGreaterEqual(slow[s], 8, s)

    def test_no_transient_settles_at_once_and_is_not_identified(self):
        rng = random.Random(5)
        res = ST.analyze(synth_blocks(100, 1.0, 0.0, rng, transient=False), reps=120, seed=5)
        self.assertFalse(res["identified"])
        self.assertIsNone(res["t_bar"])
        self.assertTrue(all(v == 0 for v in res["settled_epoch_pooled"].values()), res["settled_epoch_pooled"])
        self.assertFalse(res["decision_needed"])
        self.assertIn("no transient detected", ST.summary_line(res))

    def test_null_false_settling_rate_is_small(self):
        rng = random.Random(11)
        late = 0
        for i in range(20):
            res = ST.analyze(synth_blocks(40, 1.0, 0.0, rng, transient=False), reps=10, seed=i)
            late += sum(1 for v in res["settled_epoch_pooled"].values() if v != 0)
        self.assertLessEqual(late, 3)

    def test_groups_are_pooled_into_and_out_of_each_arm(self):
        res = ST.analyze(synth_blocks(30, 2.0, 1.5, random.Random(1)), reps=10)
        self.assertEqual(set(res["groups"]), {"pooled", "pooled, engine switch"} | {f"into {a}" for a in ARMS} | {f"out of {a}" for a in ARMS})
        self.assertEqual(sum(res["groups"][f"into {a}"]["n_blocks"] for a in ARMS), res["groups"]["pooled"]["n_blocks"])


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class Mapping(unittest.TestCase):
    def build_run(self, d):
        T0 = 10 ** 12
        ep, bl = [], []
        arms = ["I1-E", "c32|fifo", "c32|fifo"]
        for b in range(3):
            bl.append({"rec": "block", "node": "E1", "block_id": b, "arm_id": arms[b], "profile_switched": b == 1})
            bl.append({"rec": "block_end", "node": "E1", "block_id": b, "status": "complete", "epochs_logged": 50})
            for k in range(50):
                e = b * 50 + k
                ep.append({"rec": "epoch", "node": "E1", "epoch": e, "block_id": b, "epoch_in_block": k, "t_start_mono_ns": T0 + e * ST.EPOCH_NS})
        write_jsonl(os.path.join(d, "epochs.jsonl"), ep)
        write_jsonl(os.path.join(d, "blocks.jsonl"), bl)
        reqs, cloud = [], []
        for e in range(50, 100):
            k = e - 50
            hot = 1.0 if k < 3 else 0.0
            for j in range(10):
                t = T0 + e * ST.EPOCH_NS + int((j + 0.5) * ST.EPOCH_NS / 10)
                reqs.append({"node": "E1", "t_arrival": t, "family": 1 + j % 2, "ok": True, "latency_s": 1.0 + hot, "exceed": hot if j < 8 else 0.0, "deadline_violation": 0.0, "cloud_used": j < 3 + 4 * hot})
                cloud.append({"rec": "request", "node": "E1", "t_arrive": t, "queue_len_at_arrival": 1 + 2 * hot, "in_flight_at_arrival": 1})
        reqs.append({"node": "E1", "t_arrival": T0 - 5, "family": 1, "ok": True, "latency_s": 9.0, "exceed": 1.0, "deadline_violation": 1.0, "cloud_used": True})
        write_jsonl(os.path.join(d, "requests.jsonl"), reqs)
        write_jsonl(os.path.join(d, "cloud-gw.jsonl"), cloud)
        return T0

    def test_requests_map_to_epochs_by_arrival_time_and_only_the_switch_block_is_selected(self):
        with tempfile.TemporaryDirectory() as d:
            T0 = self.build_run(d)
            grid = ST.Grid(ST.iter_jsonl(os.path.join(d, "epochs.jsonl")))
            self.assertEqual(grid.locate("E1", T0 + 51 * ST.EPOCH_NS + 5), (51, 51))
            self.assertIsNone(grid.locate("E1", T0 - 1))
            self.assertIsNone(grid.locate("E1", T0 + 151 * ST.EPOCH_NS))
            acc = ST.accumulate(ST.iter_jsonl(os.path.join(d, "requests.jsonl")), ST.iter_jsonl(os.path.join(d, "cloud-gw.jsonl")), grid)
            blocks = ST.switch_blocks(d, grid, acc)
            self.assertEqual([(b["round"], b["arm"], b["prev"]) for b in blocks], [(1, "c32|fifo", "I1-E")])
            c = blocks[0]["curves"]
            self.assertAlmostEqual(c["mean_latency_s"][0], 2.0)
            self.assertAlmostEqual(c["mean_latency_s"][10], 1.0)
            self.assertAlmostEqual(c["cloud_queue"][1], 4.0)
            self.assertAlmostEqual(c["cloud_queue"][20], 2.0)
            self.assertAlmostEqual(c["cloud_share"][0], 0.7)
            self.assertAlmostEqual(c["cloud_share"][20], 0.3)
            self.assertAlmostEqual(c["tail_f1"][0], 4 / 5)
            self.assertAlmostEqual(c["tail_f2"][0], 4 / 5)
            self.assertAlmostEqual(c["deadline_miss"][0], 0.0)

    def test_run_writes_files_and_the_transient_is_found(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
            self.build_run(d)
            res = ST.run(d, os.path.join(d, "requests.jsonl"), out, reps=5)
            self.assertEqual(res["n_switch_blocks"], 1)
            self.assertTrue(os.path.exists(os.path.join(out, "SETTLING.md")))
            self.assertTrue(os.path.exists(os.path.join(out, "SETTLING.json")))
            self.assertIn("not computed", ST.summary_line({"error": "x"}))


class Text(unittest.TestCase):
    def test_render_and_summary_line(self):
        res = ST.analyze(synth_blocks(40, 2.0, 1.5, random.Random(2)), reps=20)
        md = ST.render_md(res)
        self.assertIn("Fitted t_bar", md)
        self.assertIn("into c32\\|f1prio", md)
        line = ST.summary_line(res)
        self.assertTrue(line.startswith("Settling after a switch"))
        self.assertIn("against the planned 5", line)
        self.assertIn("MOCK", ST.render_md(dict(res, _mock=True)))


if __name__ == "__main__":
    unittest.main()
