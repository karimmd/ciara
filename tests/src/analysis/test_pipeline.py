import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

from ciara_analysis import energy_bridge, pipeline
from ciara_analysis.grid import BURN_IN, MEASURE, POST, PRE, Grid
from ciara_analysis.join import build_requests
from ciara_analysis.logs import RunLogs
from ciara_analysis.outcomes import cell_outcomes, gpu_window, mean, quantile
from ciara_analysis.quality import score_request
import synth
from synth import client_rec, cloud_rec, edge_rec

TARGETS = {1: 5.0, 2: 20.0}


def fake_scorer(item, reply):
    return (1.0 if reply == "good" else 0.0), "scored"


SCORERS = {1: fake_scorer, 2: fake_scorer}


def tmpdir():
    d = tempfile.mkdtemp(prefix="ciara-analysis-", dir=os.environ.get("CIARA_TEST_TMP") or None)
    return d


def build_example(run_dir):
    c = [
        client_rec("n1B000-0", "B", 1, 2.4, 1.0, reply="good"),
        client_rec("n1B000-1", "B", 1, 3.0, 6.0, reply="bad"),
        client_rec("n1C000-0", "C", 2, 3.0, 4.0, reply="good"),
        client_rec("n1C000-1", "C", 2, 4.0, 6.0, reply="bad"),
        client_rec("n1A000-0", "A", 2, 3.0, 2.0, reply="good"),
        client_rec("n1A000-1", "A", 2, 4.0, 30.0, reply="good"),
        client_rec("n1B000-2", "B", 1, 4.5, 3.0, error="timeout", status=None, reply=None),
        client_rec("n1B000-3", "B", 1, 1.0, 2.0, reply="good"),
        client_rec("n1B000-4", "B", 1, 17.9, 5.0, reply="good"),
        client_rec("n1B000-5", "B", 1, 14.0, 1.0, reply="good"),
    ]
    e = [
        edge_rec("n1B000-0", 2.4), edge_rec("n1B000-1", 3.0),
        edge_rec("n1C000-0", 3.0, served_by="cloud"), edge_rec("n1C000-1", 4.0, served_by="cloud"),
        edge_rec("n1A000-0", 3.0), edge_rec("n1A000-1", 4.0),
        edge_rec("n1B000-2", 4.5, status=502), edge_rec("n1B000-3", 1.0), edge_rec("n1B000-4", 17.9),
    ]
    cl = [cloud_rec("n1C000-0", 3.0), cloud_rec("n1C000-1", 4.0)]
    synth.write_run(run_dir, c, e, cl)


class GridTests(unittest.TestCase):
    def test_phases_follow_the_grid(self):
        g = Grid(synth.T0, synth.EPOCH_NS, 6, 2, 3)
        self.assertEqual(g.locate(synth.ns(-0.1))["phase"], PRE)
        self.assertEqual(g.locate(synth.ns(0))["phase"], BURN_IN)
        self.assertEqual(g.locate(synth.ns(1.99))["phase"], BURN_IN)
        self.assertEqual(g.locate(synth.ns(2))["phase"], MEASURE)
        self.assertEqual(g.locate(synth.ns(5.99))["block_id"], 0)
        self.assertEqual(g.locate(synth.ns(6))["block_id"], 1)
        self.assertEqual(g.locate(synth.ns(6))["phase"], BURN_IN)
        self.assertEqual(g.locate(synth.ns(18))["phase"], POST)
        self.assertEqual(g.measure_range(1), (8, 12))
        self.assertEqual(g.burn_in_range(1), (6, 8))


class QualityRule(unittest.TestCase):
    def test_failed_and_truncated_replies_score_zero(self):
        self.assertEqual(score_request(SCORERS, 1, "i", "good", "stop", True), (1.0, "scored"))
        self.assertEqual(score_request(SCORERS, 1, "i", "good", "length", True), (0.0, "truncated"))
        self.assertEqual(score_request(SCORERS, 1, "i", "good", "length", True, truncated_is_wrong=False), (1.0, "scored"))
        self.assertEqual(score_request(SCORERS, 1, "i", None, None, False), (0.0, "failed"))
        self.assertEqual(score_request({}, 1, "i", "good", "stop", True), (None, "no_scorer"))


class StatHelpers(unittest.TestCase):
    def test_mean_and_quantile(self):
        self.assertEqual(mean([1, None, 3]), 2.0)
        self.assertIsNone(mean([None]))
        self.assertEqual(quantile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertAlmostEqual(quantile([0, 10], 0.95), 9.5)

    def test_gpu_window_energy_delta_and_scaling(self):
        s = [{"event": "sample", "mono_ns": synth.ns(2) + i * 500_000_000,
              "gpus": [{"gpu": 3, "energy_mj": 1000 * i * 100, "power_w": 200.0, "util_gpu": 50, "throttled": i == 3}]}
             for i in range(0, 80)]
        r = gpu_window(s, 3, synth.ns(2), synth.ns(6))
        self.assertAlmostEqual(r["energy_j"], 8000.0, delta=1.0)
        self.assertEqual(r["power_w_mean"], 200.0)
        self.assertGreater(r["throttled_frac"], 0)
        self.assertIsNone(gpu_window(s, 4, synth.ns(2), synth.ns(6))["energy_j"])


class JoinAndOutcomes(unittest.TestCase):
    def setUp(self):
        self.d = tmpdir()
        self.run_dir = os.path.join(self.d, "run")
        build_example(self.run_dir)
        self.res = pipeline.run(self.run_dir, os.path.join(self.run_dir, "schedule"), TARGETS, scorers=SCORERS,
                                node_gpu=None)
        self.req = {r["req"]: r for r in self.res["requests"]}

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_every_client_request_is_joined_once_and_counted(self):
        rep = self.res["report"]
        self.assertEqual((rep["n_client"], rep["n_edge"], rep["n_cloud"]), (10, 9, 2))
        self.assertEqual(rep["n_found_edge"], 9)
        self.assertEqual(rep["n_found_cloud"], 2)
        self.assertEqual(rep["client_without_edge"], ["n1B000-5"])
        self.assertEqual(rep["client_duplicates"] + rep["edge_duplicates"] + rep["cloud_duplicates"], [])
        self.assertEqual(rep["edge_without_client"], [])

    def test_assignment_to_node_block_epoch_by_completion_time(self):
        r = self.req
        self.assertEqual((r["n1B000-0"]["node"], r["n1B000-0"]["block_id"], r["n1B000-0"]["epoch"], r["n1B000-0"]["phase"]),
                         ("E1", 0, 2, MEASURE))
        self.assertEqual((r["n1A000-1"]["block_id"], r["n1A000-1"]["epoch"], r["n1A000-1"]["phase"]), (1, 7, BURN_IN))
        self.assertEqual(r["n1A000-1"]["block_id_arrival"], 0)
        self.assertEqual(r["n1B000-3"]["phase"], BURN_IN)
        self.assertEqual(r["n1B000-4"]["phase"], POST)
        self.assertIsNone(r["n1B000-4"]["block_id"])
        self.assertEqual(r["n1B000-0"]["arm_of_block"], "A1")
        self.assertEqual(r["n1A000-1"]["arm_of_block"], "A2")
        self.assertEqual(r["n1A000-1"]["arm_of_block_at_admit"], "A1")

    def test_measurement_row_of_block_0_equals_the_hand_computation(self):
        row = [r for r in self.res["measure"] if r["block_id"] == 0][0]
        self.assertEqual(row["n"], 6)
        self.assertEqual((row["n_f1"], row["n_f2"]), (3, 3))
        self.assertAlmostEqual(row["Y_Q"], 3 / 6)
        self.assertAlmostEqual(row["Y_L_f1"], 2 / 3)
        self.assertAlmostEqual(row["Y_L_f2"], 0.0)
        self.assertAlmostEqual(row["Y_D"], 2 / 6)
        self.assertAlmostEqual(row["Y_C"], 2 / 6)
        self.assertEqual((row["n_failed"], row["n_cloud"]), (1, 2))
        self.assertEqual(row["n_arrived_earlier_block"], 0)

    def test_arm_propensity_context_and_configuration_are_those_of_the_block(self):
        rows = {r["block_id"]: r for r in self.res["measure"]}
        self.assertEqual(len(rows), 3)
        self.assertEqual([rows[i]["arm_id"] for i in range(3)], ["A1", "A2", "I2"])
        self.assertEqual({r["propensity"] for r in rows.values()}, {1 / 3})
        self.assertEqual(rows[0]["context_id"], "pooled")
        self.assertEqual((rows[1]["cfg_profile"], rows[1]["cfg_batch_cap"]), ("p2", 16))
        self.assertEqual(rows[2]["cfg_router"], "I2")
        self.assertEqual(rows[1]["block_status"], "complete")
        self.assertEqual((rows[1]["epoch_lo"], rows[1]["epoch_hi"]), (8, 12))

    def test_burn_in_is_a_separate_table(self):
        burn = {r["block_id"]: r for r in self.res["burn_in"]}
        meas = {r["block_id"]: r for r in self.res["measure"]}
        self.assertEqual((burn[0]["epoch_lo"], burn[0]["epoch_hi"]), (0, 2))
        self.assertEqual(burn[0]["n"], 1)
        self.assertEqual(burn[1]["n"], 1)
        self.assertEqual(burn[1]["n_switch_epochs_window"], 1)
        self.assertEqual(meas[1]["n_switch_epochs_window"], 0)
        self.assertEqual(meas[1]["n"], 0)
        self.assertIsNone(meas[1]["Y_Q"])
        self.assertEqual(sum(r["n"] for r in meas.values()) + sum(r["n"] for r in burn.values()), 9)

    def test_gateway_loss_is_flagged_not_hidden(self):
        row = [r for r in self.res["measure"] if r["block_id"] == 2][0]
        self.assertEqual((row["n"], row["n_no_gateway_record"]), (1, 1))
        self.assertAlmostEqual(row["Y_Q"], 1.0)
        self.assertAlmostEqual(row["Y_C"], 0.0)

    def test_energy_outcome_is_the_class_mean_of_clipped_normalized_energy(self):
        en = energy_bridge.device_energy(self.run_dir, os.path.join(self.run_dir, "schedule"), overhead="constant")
        for q in en.values():
            self.assertTrue(0.0 <= q["e_norm"] <= 1.0)
        row = [r for r in self.res["measure"] if r["block_id"] == 0][0]
        b = [en[k]["e_norm"] for k in ("n1B000-0", "n1B000-1", "n1B000-2")]
        c = [en[k]["e_norm"] for k in ("n1C000-0", "n1C000-1")]
        self.assertAlmostEqual(row["Y_E_B"], sum(b) / 3, places=6)
        self.assertAlmostEqual(row["Y_E_C"], sum(c) / 2, places=6)
        self.assertAlmostEqual(row["Y_E_A"], en["n1A000-0"]["e_norm"], places=6)

    def test_all_outcomes_are_in_the_unit_interval(self):
        for tbl in (self.res["measure"], self.res["burn_in"]):
            for r in tbl:
                for k in ("Y_Q", "Y_L_f1", "Y_L_f2", "Y_D", "Y_E_A", "Y_E_B", "Y_E_C", "Y_C"):
                    if r[k] is not None:
                        self.assertTrue(0.0 <= r[k] <= 1.0, (k, r[k]))

    def test_files_are_written_in_csv_and_jsonl(self):
        out = self.res["out_dir"]
        for name in ("outcomes.csv", "outcomes.jsonl", "outcomes_burnin.csv", "outcomes_burnin.jsonl", "requests.jsonl",
                     "join_report.json"):
            self.assertTrue(os.path.exists(os.path.join(out, name)), name)
        with open(os.path.join(out, "outcomes.csv")) as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["arm_id"], "A1")
        self.assertEqual(rows[1]["Y_Q"], "")
        with open(os.path.join(out, "outcomes.jsonl")) as fh:
            js = [json.loads(line) for line in fh]
        self.assertEqual(js[0]["Y_Q"], 0.5)
        self.assertIsNone(js[1]["Y_Q"])

    def test_cell_outcomes_of_an_empty_cell(self):
        o = cell_outcomes([])
        self.assertEqual(o["n"], 0)
        self.assertIsNone(o["Y_C"])
        self.assertIsNone(o["Y_L_f1"])


class CompactReplyScoring(unittest.TestCase):
    def test_quality_is_scored_on_the_full_gateway_output_not_the_compact_client_reply(self):
        d = tmpdir()
        try:
            run_dir = os.path.join(d, "run")
            c = [client_rec("n1A000-0", "A", 2, 2.4, 1.0, reply="compact-bad")]
            e = [edge_rec("n1A000-0", 2.4, reply_text="good", finish_reason="stop", delivery="compact", compact_extracted=True,
                          compact_bytes=14)]
            synth.write_run(run_dir, c, e, [])
            logs = RunLogs(run_dir)
            rows, _ = build_requests(logs.client_records(), logs.edge_records(), logs.cloud_records(), logs.blocks(),
                                     logs.manifest(), SCORERS)
            self.assertEqual((rows[0]["q"], rows[0]["delivery"], rows[0]["compact_bytes"]), (1.0, "compact", 14))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Duplicates(unittest.TestCase):
    def test_duplicate_and_orphan_records_are_reported(self):
        d = tmpdir()
        try:
            run_dir = os.path.join(d, "run")
            c = [client_rec("n1B000-0", "B", 1, 2.4, 1.0, reply="good")]
            e = [edge_rec("n1B000-0", 2.4), edge_rec("n1B000-0", 2.4), edge_rec("zzz-0", 2.4)]
            synth.write_run(run_dir, c, e, [cloud_rec("orphan-0", 3.0)])
            logs = RunLogs(run_dir)
            rows, rep = build_requests(logs.client_records(), logs.edge_records(), logs.cloud_records(), logs.blocks(),
                                       logs.manifest(), SCORERS)
            self.assertEqual(rep["edge_duplicates"], ["n1B000-0"])
            self.assertEqual(rep["edge_without_client"], ["zzz-0"])
            self.assertEqual(rep["cloud_without_client"], ["orphan-0"])
            self.assertEqual(len(rows), 1)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class FrozenScorers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from ciara_analysis.quality import SuiteScorer
            cls.iot = SuiteScorer("iot", "task_acc")
            cls.sensor = SuiteScorer("sensor", "answer_acc")
        except Exception as exc:
            raise unittest.SkipTest(f"frozen suites unavailable: {exc}")

    def test_iot_gold_reply_scores_one_and_a_wrong_reply_zero(self):
        iid, it = next(iter(self.iot.items.items()))
        gold = it["_row"]["gold_calls"]
        good = json.dumps({"calls": [{"name": g["name"], "arguments": g["arguments"]} for g in gold]})
        self.assertEqual(self.iot.score(iid, good), (1.0, "scored"))
        self.assertEqual(self.iot.score(iid, '{"calls": [{"name": "NoSuchTool", "arguments": {}}]}'), (0.0, "scored"))
        self.assertEqual(self.iot.score("no-such-item", good), (None, "unknown_item"))

    def test_sensor_gold_answer_scores_one_and_a_wrong_answer_zero(self):
        iid, it = next(iter(self.sensor.items.items()))
        gold = it["_row"]["answer"]
        self.assertEqual(self.sensor.score(iid, json.dumps({"answer": gold})), (1.0, "scored"))
        self.assertEqual(self.sensor.score(iid, "no idea"), (0.0, "scored"))


if __name__ == "__main__":
    unittest.main()
