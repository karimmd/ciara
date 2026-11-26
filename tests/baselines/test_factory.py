import json
import math
import unittest
from pathlib import Path

import ciara_baselines.factory
from ciara_baselines.factory import NAMES, make_incumbent
from ciara_baselines.mixorsplit import MosParams
from ciara_baselines.profile import ProfileTable, switch_overhead
from ciara_baselines.race import (
    RaceParams, RaceRouter, build_prefill_request, first_token_score, probe_log_record, score_from_prefill_response,
    score_request,
)

TARGETS = {"f1": 8.0, "f2": 30.0}


class Obs(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def control_obs(epoch=1, rate=0.3, block_id=7, queue_len=2, mib=11264):
    return Obs(
        node="E1", node_index=0, epoch=epoch, block_id=block_id, epoch_in_block=epoch % 50, context_id=0,
        epoch_s=30.0, t_mono_ns=10 ** 12, wall_ns=10 ** 18, arrival_rate=rate, memory_budget_mib=mib,
        gateway={"queue_len": queue_len, "in_flight": 3, "arrivals": 9, "completed": 8, "escalated": 1, "errors": 0,
                 "counters": {"arrived": 9}, "recent": None, "config": {"profile": "q35-4b-fp16"}},
        engine={"kv_cache_usage": 0.4}, profile="q35-4b-fp16", config={"profile": "q35-4b-fp16"},
        first_epoch_of_block=(epoch % 50 == 0),
    )


class FactoryTests(unittest.TestCase):
    def test_names(self):
        self.assertEqual(NAMES, ("I1-E", "I1-S", "I1-T", "I1-oracle", "I2", "I2-all", "I3", "I3-all", "RecServe",
                                 "RecServe-all", "CB"))
        with self.assertRaises(ValueError):
            make_incumbent("I9")

    def test_i1_variants_return_kappa_router_dicts(self):
        for name in ("I1-E", "I1-T"):
            a = make_incumbent(name, latency_targets_s=TARGETS)
            d = a.decide(control_obs())
            self.assertEqual(d["router"], "kappa")
            self.assertTrue(d["profile"].startswith("q35-4b"))
            self.assertIn(d["batch_cap"], (8, 16))
            self.assertNotIn("arm_id", d)
            self.assertNotIn("base_arm", d)
            json.dumps(d)
        s = make_incumbent("I1-S", latency_targets_s=TARGETS, design_load=0.3)
        self.assertEqual(s.decide(control_obs(rate=9.0)), s.decide(control_obs(rate=0.1)))

    def test_i1_needs_targets_and_static_needs_design_load(self):
        with self.assertRaises(ValueError):
            make_incumbent("I1-E")
        with self.assertRaises(ValueError):
            make_incumbent("I1-S", latency_targets_s=TARGETS)
        with self.assertRaises(ValueError):
            make_incumbent("I1-oracle", latency_targets_s=TARGETS)

    def test_oracle_with_its_own_table(self):
        path = Path(ciara_baselines.factory.DEFAULT_TABLE)
        d = make_incumbent("I1-oracle", latency_targets_s=TARGETS, oracle_table_path=str(path)).decide(control_obs())
        self.assertEqual(d["router"], "kappa")

    def test_headline_and_all_variants_differ_only_in_scope(self):
        h2 = make_incumbent("I2", base_profile="q35-4b-fp16", base_batch_cap=8).decide(control_obs())
        a2 = make_incumbent("I2-all", base_profile="q35-4b-fp16", base_batch_cap=8).decide(control_obs())
        self.assertEqual((h2["router"], a2["router"]), ("I2", "I2"))
        self.assertEqual(h2["router_params"]["escalable_families"], ["f2"])
        self.assertEqual(a2["router_params"]["escalable_families"], ["f1", "f2"])
        self.assertEqual(h2["router_params"]["score"], "first_token")
        h3 = make_incumbent("I3", base_profile="q35-4b-fp16", base_batch_cap=8).decide(control_obs())
        a3 = make_incumbent("I3-all", base_profile="q35-4b-fp16", base_batch_cap=8).decide(control_obs())
        self.assertEqual(h3["router_params"]["eligible_families"], ["f2"])
        self.assertEqual(a3["router_params"]["eligible_families"], ["f1", "f2"])
        for d in (h2, a2, h3, a3):
            self.assertEqual((d["profile"], d["batch_cap"], d["escalate_family2"]), ("q35-4b-fp16", 8, False))
            json.dumps(d)

    def test_base_from_static_selection_when_not_given(self):
        d = make_incumbent("I2", latency_targets_s=TARGETS, design_load=0.3).decide(control_obs())
        self.assertTrue(d["profile"].startswith("q35-4b"))
        with self.assertRaises(ValueError):
            make_incumbent("I3")

    def test_inadmissible_base_is_refused(self):
        with self.assertRaises(ValueError):
            make_incumbent("I2", base_profile="q35-4b-fp16", base_batch_cap=16)
        with self.assertRaises(ValueError):
            make_incumbent("I2", base_profile="q35-4b-fp16")

    def test_overrides_and_typos(self):
        a = make_incumbent("I2", base_profile="q35-4b-fp16", base_batch_cap=8, window=50,
                           params={"score": "semantic_entropy"})
        rp = a.decide(control_obs())["router_params"]
        self.assertEqual((rp["window"], rp["score"]), (50, "semantic_entropy"))
        with self.assertRaises(ValueError):
            make_incumbent("I3", base_profile="q35-4b-fp16", base_batch_cap=8, windows=5)


class AdapterTests(unittest.TestCase):
    def test_obs_conversion(self):
        a = make_incumbent("I1-E", latency_targets_s=TARGETS, design_load=0.25)
        o = a.to_obs(control_obs(epoch=4, rate=0.5, mib=11264))
        self.assertEqual((o.epoch, o.arrival_rate, o.block_id), (4, 0.5, 7))
        self.assertAlmostEqual(o.memory_budget_gib, 10.45)
        self.assertEqual(o.telemetry["gateway"]["queue_len"], 2)
        o2 = a.to_obs(control_obs(rate=None))
        self.assertEqual(o2.arrival_rate, 0.5)
        fresh = make_incumbent("I1-E", latency_targets_s=TARGETS, design_load=0.25)
        self.assertEqual(fresh.to_obs(control_obs(rate=None)).arrival_rate, 0.25)

    def test_begin_block_resets_the_guard(self):
        a = make_incumbent("I1-E", latency_targets_s=TARGETS)
        a.decide(control_obs(epoch=1, rate=0.2))
        self.assertIsNotNone(a.inner._current)
        a.begin_block({"node": "E1", "block_id": 8, "epoch": 50, "context_id": 0, "arm_id": "I1-E", "resumed": False})
        self.assertIsNone(a.inner._current)
        self.assertEqual(a.blocks[0]["block_id"], 8)

    def test_begin_block_passes_horizon_to_race(self):
        a = make_incumbent("I2", base_profile="q35-4b-fp16", base_batch_cap=8, horizon=120)
        a.begin_block({"block_id": 3})
        r = a.inner.router
        r.route({"req_id": "x", "family": 2, "race_u": 0.1, "messages": []}, {})
        self.assertEqual(r.cal["f2"].T, 120)

    def test_memory_budget_change_is_seen(self):
        a = make_incumbent("I1-E", latency_targets_s=TARGETS)
        d1 = a.decide(control_obs(epoch=1, rate=0.2, mib=11264))
        d2 = a.decide(control_obs(epoch=2, rate=0.2, mib=6144))
        self.assertNotEqual((d1["profile"], d1["batch_cap"]), (d2["profile"], d2["batch_cap"]))


class ScopeAndScoreTests(unittest.TestCase):
    def test_headline_scope_defaults(self):
        self.assertEqual(RaceParams().escalable_families, ("f2",))
        self.assertEqual(MosParams().eligible_families, ("f2",))
        self.assertEqual(RaceParams().score, "first_token")

    def test_unknown_score_is_refused(self):
        with self.assertRaises(ValueError):
            RaceParams(score="entropy")

    def test_first_token_score(self):
        self.assertAlmostEqual(first_token_score(math.log(0.9)), 0.1)
        self.assertAlmostEqual(first_token_score(0.0), 0.0)
        self.assertAlmostEqual(first_token_score(math.log(0.25)), 0.75)

    def test_prefill_request_and_response(self):
        meta = {"req_id": "r-1", "family": 2, "messages": [{"role": "user", "content": "q"}], "model": "m"}
        req = build_prefill_request(meta, RaceParams())
        b = req["body"]
        self.assertEqual((b["max_tokens"], b["temperature"], b["logprobs"], b["top_logprobs"], b["stream"]),
                         (1, 0.0, True, 5, False))
        self.assertEqual(req["headers"]["X-Ciara-Kind"], "race_prefill")
        self.assertEqual(score_request(meta, RaceParams())["headers"]["X-Ciara-Req"], "r-1.prefill")
        resp = {"choices": [{"logprobs": {"content": [{"token": "{", "logprob": math.log(0.8),
                "top_logprobs": [{"token": "{", "logprob": math.log(0.8)}, {"token": "[", "logprob": math.log(0.1)}]}]}}]}
        self.assertAlmostEqual(score_from_prefill_response(resp), 0.2)

    def test_semantic_entropy_stays_a_named_option(self):
        meta = {"req_id": "r-1", "family": 2, "messages": [], "max_tokens": 1024}
        req = score_request(meta, RaceParams(score="semantic_entropy"))
        self.assertEqual((req["body"]["n"], req["headers"]["X-Ciara-Kind"]), (5, "race_probe"))
        with self.assertRaises(NotImplementedError):
            score_request(meta, RaceParams(score="sep"))

    def test_log_record_kind_follows_the_score(self):
        meta = {"req_id": "r", "family": 2}
        rec = probe_log_record(meta, {"n": 1, "prompt_tokens": 100, "completion_tokens": 1}, 0, 10 ** 9, 0.2, "first_token")
        self.assertEqual(rec["kind"], "race_prefill")

    def test_router_state_names_the_score(self):
        r = RaceRouter(RaceParams(alpha={"f1": 0.3, "f2": 0.3}), horizon=10)
        r.route({"req_id": "a", "family": 2, "race_u": 0.1, "messages": []}, {})
        st = r.state()
        self.assertEqual(st["score"], "first_token")
        self.assertEqual(st["escalable_families"], ["f2"])
        self.assertIn("f2", st["calibrators"])
        self.assertEqual(r.events[-1]["score"], "first_token")
        self.assertEqual(r.route({"req_id": "b", "family": 1, "messages": []}, {}), "edge")

    def test_overhead_from_switch_test(self):
        o = switch_overhead()
        self.assertEqual(o.switch_time_s("q35-4b-ih-w8a8", "q35-4b-ih-w4a16"), 1.76)
        self.assertEqual(o.switch_time_s("q35-2b-fp16", "q35-4b-fp16"), 50.0)
        self.assertEqual(o.switch_time_s("q35-4b-fp16", "q35-2b-fp16"), 98.0)
        self.assertEqual(o.switch_time_s("q35-4b-fp16", "q35-4b-fp16"), 0.0)
        t = ProfileTable.load(ciara_baselines.factory.DEFAULT_TABLE)
        self.assertEqual(t.overhead.switch_time_s("q35-4b-fp16", "q35-0.8b-fp16"), 98.0)


if __name__ == "__main__":
    unittest.main()
