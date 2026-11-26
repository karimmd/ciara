from __future__ import annotations

import math
import os
import random
import sys
import unittest


from ciara_edge import routers, scoring
from ciara_baselines.recserve import RecServeParams, RecServeRouter, confidence_from_logprobs


def meta(rid, family=2, c=None):
    m = {"req_id": rid, "req": rid, "family": family, "messages": [], "node": "E1"}
    if c is not None:
        m["race_u"] = c
    return m


class Listed(unittest.TestCase):
    def test_router_is_listed_and_built_from_router_params(self):
        self.assertIn("RecServe", routers.BUILTIN)
        p = RecServeParams(beta=0.5, window=300, score="perplexity", escalable_families=("f1", "f2"))
        r = routers.get_router("RecServe", p.to_router_params())
        self.assertEqual(r.rec_params, p)
        self.assertTrue(r.rebuild_per_block)
        self.assertFalse(r.wants_feedback)
        self.assertFalse(r.needs_after_edge)

    def test_the_queues_are_rebuilt_when_the_block_changes_only(self):
        a = routers.router_signature("RecServe", {"beta": 0.3}, 4)
        self.assertEqual(a, routers.router_signature("RecServe", {"beta": 0.3}, 4))
        self.assertNotEqual(a, routers.router_signature("RecServe", {"beta": 0.3}, 5))
        self.assertNotEqual(a, routers.router_signature("RecServe", {"beta": 0.1}, 4))

    def test_gateway_only_settings_are_ignored(self):
        r = routers.get_router("RecServe", {"beta": 0.3, "gateway": {"score": "logprob"}})
        self.assertEqual(r.scorer.kind, "perplexity")


class Scorers(unittest.TestCase):
    def test_perplexity_scorer_is_an_edge_pass_with_top1_log_probabilities(self):
        s = scoring.make_recserve_scorer("perplexity")
        self.assertEqual((s.mode, s.kind, s.top_k), ("edge_pass", "perplexity", 1))
        self.assertEqual(s.request_fields(), {"logprobs": True, "top_logprobs": 1})
        reply = {"token_top": [[-0.1], [-0.3], [-0.2]]}
        self.assertAlmostEqual(s.score(meta("a"), reply), confidence_from_logprobs([-0.1, -0.3, -0.2]))
        self.assertIsNone(s.score(meta("a"), {"token_top": []}))
        self.assertIsNone(s.score(meta("a"), {}))

    def test_first_token_scorer_is_the_prefill_pass_with_a_confidence_value(self):
        s = scoring.make_recserve_scorer("first_token")
        self.assertEqual((s.mode, s.kind), ("prefill", "first_token"))
        self.assertEqual(s.request_fields()["max_tokens"], 1)
        self.assertAlmostEqual(s.score(meta("a"), math.log(0.8)), 0.8)

    def test_unknown_score_is_refused(self):
        with self.assertRaises(ValueError):
            scoring.make_recserve_scorer("entropy")


class Decisions(unittest.TestCase):
    def test_perplexity_route_decides_after_the_edge_pass_like_the_package_router(self):
        r = routers.get_router("RecServe", RecServeParams(beta=0.3, escalable_families=("f1", "f2")).to_router_params())
        ref = RecServeRouter(RecServeParams(beta=0.3, escalable_families=("f1", "f2")))
        rng = random.Random(2)
        for i in range(300):
            c = rng.betavariate(6, 3)
            got = r.route(meta(f"r{i}", family=1 + i % 2, c=c), {})
            want = "cloud" if ref.after_edge(meta(f"r{i}", family=1 + i % 2), {"confidence": c}, {}) == "escalate" else "edge"
            self.assertEqual(got, want, i)
        self.assertEqual(r.state()["queues"]["f2"]["t"], 150)

    def test_first_token_route_decides_before_generation(self):
        r = routers.get_router("RecServe", RecServeParams(beta=0.3, score="first_token").to_router_params())
        out = [r.route(meta(f"r{i}", c=c), {}) for i, c in enumerate([0.9, 0.8, 0.85, 0.2])]
        self.assertEqual(out, ["edge", "cloud", "edge", "cloud"])

    def test_missing_confidence_serves_the_request_at_the_edge(self):
        for score in ("perplexity", "first_token"):
            r = routers.get_router("RecServe", RecServeParams(score=score).to_router_params())
            self.assertEqual(r.route(meta("x"), {}), "edge")
            self.assertEqual(r.decision_info()["reason"], "no_score")
            self.assertEqual(r.state()["n_no_score"], 1)

    def test_headline_scope_keeps_family_1_at_the_edge(self):
        r = routers.get_router("RecServe", RecServeParams(beta=0.5).to_router_params())
        for i, c in enumerate([0.4, 0.2, 0.3, 0.1, 0.05]):
            self.assertEqual(r.route(meta(f"a{i}", family=1, c=c), {}), "edge")
        self.assertEqual(r.state()["queues"]["f1"]["t"], 5)

    def test_decision_info_is_a_small_json_record(self):
        r = routers.get_router("RecServe", {"beta": 0.3})
        r.route(meta("a", c=0.3), {})
        r.route(meta("b", c=0.1), {})
        info = r.decision_info()
        self.assertEqual(set(info), {"c", "threshold", "n_hist", "step", "reason"})
        self.assertEqual((info["step"], info["n_hist"]), (2, 2))

    def test_cloud_share_is_beta_of_the_eligible_stream_in_expectation(self):
        scorer = scoring.make_recserve_scorer("perplexity")
        r = routers.get_router("RecServe", RecServeParams(beta=0.3, window=1000).to_router_params())
        rng = random.Random(9)
        n_cloud = n = 0
        for i in range(6000):
            mu = rng.uniform(0.02, 1.2)
            top = [[-abs(rng.gauss(mu, mu / 3))] for _ in range(40)]
            m = meta(f"r{i}")
            m["race_u"] = scorer.score(m, {"token_top": top})
            n += 1
            n_cloud += r.route(m, {}) == "cloud"
        self.assertAlmostEqual(n_cloud / n, 0.3, delta=0.02)


if __name__ == "__main__":
    unittest.main()
