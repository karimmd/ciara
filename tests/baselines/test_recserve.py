import json
import math
import random
import unittest

from ciara_baselines.base import Config, Obs
from ciara_baselines.factory import make_incumbent
from ciara_baselines.recserve import (
    ConfidenceQueue, RecServeIncumbent, RecServeNode, RecServeParams, RecServeRouter, confidence_from_logprobs,
    confidence_from_prefill_response, confidence_from_token_top, max_prob_confidence, perplexity,
)

try:
    import numpy as np
except Exception:
    np = None


def rp(**kw):
    kw.setdefault("escalable_families", ("f1", "f2"))
    return RecServeParams(**kw)


def reference_algorithm_1(scores, beta, k, min_history=2):
    hist, out = [], []
    for c in scores:
        hist.append(c)
        if len(hist) > k:
            hist.pop(0)
        s = sorted(hist)
        if len(s) < min_history:
            out.append((None, False))
            continue
        r = beta * (len(s) - 1)
        lo, hi = math.floor(r), math.ceil(r)
        t = s[lo] * (1 - (r - lo)) + s[hi] * (r - lo)
        out.append((t, c < t))
    return out


def req(family=2, rid="r1", **kw):
    return {"req_id": rid, "family": family, **kw}


class ConfidenceTests(unittest.TestCase):
    def test_perplexity_and_eq_12_by_hand(self):
        lps = [-math.log(2.0), -math.log(2.0)]
        self.assertAlmostEqual(perplexity(lps), 2.0)
        self.assertAlmostEqual(confidence_from_logprobs(lps), 1.0 / 3.0)
        lps = [-math.log(1.0), -math.log(4.0)]
        self.assertAlmostEqual(perplexity(lps), 2.0)

    def test_empty_reply_has_no_confidence_and_overflow_gives_zero(self):
        self.assertIsNone(confidence_from_logprobs([]))
        self.assertIsNone(confidence_from_token_top([]))
        self.assertEqual(confidence_from_logprobs([-1e6, -1e6]), 0.0)
        with self.assertRaises(ValueError):
            perplexity([])

    def test_confidence_is_monotone_in_the_log_likelihood(self):
        a = confidence_from_logprobs([-0.1] * 20)
        b = confidence_from_logprobs([-0.5] * 20)
        c = confidence_from_logprobs([-2.0] * 20)
        self.assertTrue(a > b > c > 0.0)
        self.assertLessEqual(a, 0.5)

    def test_token_top_uses_the_largest_logprob_of_each_token(self):
        top = [[-0.2, -1.5], [-0.4], [-0.1, -3.0, -4.0]]
        self.assertAlmostEqual(confidence_from_token_top(top), confidence_from_logprobs([-0.2, -0.4, -0.1]))

    def test_peak_probability_and_prefill_response(self):
        self.assertAlmostEqual(max_prob_confidence(math.log(0.8)), 0.8)
        self.assertEqual(max_prob_confidence(0.5), 1.0)
        resp = {"choices": [{"logprobs": {"content": [{"logprob": math.log(0.6),
                                                        "top_logprobs": [{"logprob": math.log(0.6)},
                                                                         {"logprob": math.log(0.3)}]}]}}]}
        self.assertAlmostEqual(confidence_from_prefill_response(resp), 0.6)


class QueueTests(unittest.TestCase):
    def test_fifo_eviction_eq_6(self):
        q = ConfidenceQueue(3)
        for v in (0.5, 0.1, 0.9, 0.3, 0.7):
            q.push(v)
        self.assertEqual(q.scores(), [0.9, 0.3, 0.7])
        self.assertEqual(len(q), 3)
        self.assertEqual(q._sorted, [0.3, 0.7, 0.9])

    def test_quantile_by_hand_eq_15(self):
        q = ConfidenceQueue(10)
        for v in (0.1, 0.4, 0.2, 0.9, 0.5):
            q.push(v)
        self.assertAlmostEqual(q.quantile(0.3), 0.24)
        self.assertAlmostEqual(q.quantile(0.5), 0.4)
        self.assertAlmostEqual(q.quantile(0.1), 0.1 * 0.6 + 0.2 * 0.4)

    def test_single_element_queue(self):
        q = ConfidenceQueue(10)
        q.push(0.42)
        self.assertEqual(q.quantile(0.3), 0.42)

    def test_duplicates_are_evicted_one_at_a_time(self):
        q = ConfidenceQueue(3)
        for v in (0.5, 0.5, 0.5, 0.2):
            q.push(v)
        self.assertEqual(q.scores(), [0.5, 0.5, 0.2])
        self.assertEqual(q._sorted, [0.2, 0.5, 0.5])

    @unittest.skipIf(np is None, "numpy not installed")
    def test_equals_numpy_percentile(self):
        rng = random.Random(3)
        q = ConfidenceQueue(50)
        hist = []
        for _ in range(300):
            v = rng.random()
            q.push(v)
            hist.append(v)
            hist = hist[-50:]
            for beta in (0.1, 0.3, 0.5, 0.77):
                self.assertAlmostEqual(q.quantile(beta), float(np.percentile(hist, beta * 100)), places=12)


class AlgorithmTests(unittest.TestCase):
    def test_matches_algorithm_1_on_a_random_stream(self):
        rng = random.Random(11)
        scores = [rng.betavariate(5, 2) for _ in range(1500)]
        for beta, k in ((0.3, 200), (0.1, 50), (0.5, 1000)):
            node = RecServeNode(rp(beta=beta, window=k))
            ref = reference_algorithm_1(scores, beta, k)
            for i, c in enumerate(scores):
                d = node.step(c)
                t, esc = ref[i]
                self.assertEqual(d.escalate, esc, (beta, k, i))
                if t is None:
                    self.assertIsNone(d.threshold)
                else:
                    self.assertAlmostEqual(d.threshold, t, places=12)

    def test_cold_start_serves_the_first_request(self):
        node = RecServeNode(rp())
        d1 = node.step(0.01)
        self.assertFalse(d1.escalate)
        self.assertIsNone(d1.threshold)
        d2 = node.step(0.5)
        self.assertIsNotNone(d2.threshold)

    def test_the_current_score_is_in_its_own_window(self):
        node = RecServeNode(rp(beta=0.3))
        node.step(0.9)
        d = node.step(0.1)
        self.assertAlmostEqual(d.threshold, 0.34)
        self.assertTrue(d.escalate)

    def test_serve_at_the_edge_when_confidence_equals_threshold(self):
        node = RecServeNode(rp(beta=0.5))
        for _ in range(5):
            d = node.step(0.7)
            self.assertFalse(d.escalate)
        self.assertEqual(node.n_escalated, 0)

    def test_threshold_follows_a_shift_of_the_distribution(self):
        rng = random.Random(5)
        node = RecServeNode(rp(beta=0.3, window=400))
        for _ in range(400):
            node.step(rng.uniform(0.0, 1.0))
        low = node.state()["threshold"]
        for _ in range(400):
            node.step(rng.uniform(0.5, 1.5))
        high = node.state()["threshold"]
        self.assertAlmostEqual(low, 0.3, delta=0.06)
        self.assertAlmostEqual(high, 0.8, delta=0.06)
        self.assertGreater(high, low + 0.4)

    def test_escalation_share_is_beta_in_expectation(self):
        for law, beta in ((lambda r: r.random(), 0.3), (lambda r: r.betavariate(8, 2), 0.1),
                          (lambda r: r.gauss(0.3, 0.05), 0.5)):
            rng = random.Random(21)
            node = RecServeNode(rp(beta=beta, window=2000))
            n = 20000
            for _ in range(n):
                node.step(law(rng))
            self.assertAlmostEqual(node.n_escalated / n, beta, delta=0.02)

    def test_parameter_checks(self):
        for bad in ({"beta": 0.0}, {"beta": 1.0}, {"window": 1}, {"min_history": 0}, {"score": "sep"}):
            with self.assertRaises(ValueError):
                RecServeParams(**bad)

    def test_router_params_round_trip(self):
        p = RecServeParams(beta=0.1, window=500, score="first_token", per_family=False, escalable_families=("f1", "f2"))
        d = json.loads(json.dumps(p.to_router_params()))
        self.assertEqual(RecServeParams.from_router_params(d), p)


class RouterTests(unittest.TestCase):
    def test_perplexity_score_decides_in_after_edge_only(self):
        r = RecServeRouter(rp(beta=0.3))
        replies = [0.5, 0.4, 0.45, 0.2]
        out = []
        for i, c in enumerate(replies):
            self.assertEqual(r.route(req(rid=f"a{i}", recserve_c=c), {}), "edge")
            out.append(r.after_edge(req(rid=f"a{i}"), {"confidence": c}, {}))
        self.assertEqual(out[-1], "escalate")
        self.assertEqual(r.nodes["f2"].t, 4)

    def test_first_token_score_decides_in_route_only(self):
        r = RecServeRouter(rp(beta=0.3, score="first_token"))
        outs = [r.route(req(rid=f"b{i}", recserve_c=c), {}) for i, c in enumerate([0.9, 0.8, 0.85, 0.3])]
        self.assertEqual(outs[-1], "cloud")
        self.assertEqual(r.after_edge(req(), {"confidence": 0.1}, {}), "accept")
        self.assertEqual(r.nodes["f2"].t, 4)

    def test_confidence_from_token_top_in_after_edge(self):
        r = RecServeRouter(rp(beta=0.5))
        good = {"token_top": [[-0.05]] * 10}
        bad = {"token_top": [[-1.5]] * 10}
        self.assertEqual(r.after_edge(req(rid="1"), good, {}), "accept")
        self.assertEqual(r.after_edge(req(rid="2"), bad, {}), "escalate")

    def test_missing_score_serves_at_the_edge_and_is_counted(self):
        r = RecServeRouter(rp())
        self.assertEqual(r.after_edge(req(), {"text": "x"}, {}), "accept")
        r2 = RecServeRouter(rp(score="first_token"))
        self.assertEqual(r2.route(req(), {}), "edge")
        self.assertEqual((r.n_no_score, r2.n_no_score), (1, 1))
        self.assertEqual(r.nodes, {})

    def test_headline_scope_escalates_family_2_only_but_keeps_both_queues(self):
        r = RecServeRouter(RecServeParams(beta=0.5))
        for i, c in enumerate([0.9, 0.1, 0.8, 0.05]):
            self.assertEqual(r.after_edge(req(family=1, rid=f"f1-{i}"), {"confidence": c}, {}), "accept")
        self.assertEqual(r.nodes["f1"].t, 4)
        self.assertEqual(r.nodes["f1"].n_escalated, 1 + 1)
        r.after_edge(req(family=2, rid="x"), {"confidence": 0.9}, {})
        self.assertEqual(r.after_edge(req(family=2, rid="y"), {"confidence": 0.0}, {}), "escalate")

    def test_per_family_and_pooled_queues(self):
        sep = RecServeRouter(rp(beta=0.5, per_family=True))
        pooled = RecServeRouter(rp(beta=0.5, per_family=False))
        for r in (sep, pooled):
            for i in range(4):
                r.after_edge(req(family=1, rid=f"p{i}"), {"confidence": 0.9}, {})
        self.assertEqual(sep.after_edge(req(family=2, rid="q"), {"confidence": 0.5}, {}), "accept")
        self.assertEqual(pooled.after_edge(req(family=2, rid="q"), {"confidence": 0.5}, {}), "escalate")
        self.assertEqual(set(sep.nodes), {"f1", "f2"})
        self.assertEqual(set(pooled.nodes), {"all"})

    def test_state_and_last_record(self):
        r = RecServeRouter(rp(beta=0.3))
        r.after_edge(req(rid="s1"), {"confidence": 0.5}, {})
        r.after_edge(req(rid="s2"), {"confidence": 0.1}, {})
        st = r.state()
        self.assertEqual(st["queues"]["f2"]["n_hist"], 2)
        self.assertEqual(st["queues"]["f2"]["n_escalated"], 1)
        self.assertEqual(r.last["req"], "s2")
        self.assertEqual(r.last["route"], "cloud")
        json.dumps(st)


class IncumbentTests(unittest.TestCase):
    BASE = Config("I1-S", "q35-4b-fp16", 8, False, base_arm="q35-4b-fp16|b8|k0")

    def test_decide_returns_the_recserve_router_on_the_base_configuration(self):
        inc = RecServeIncumbent(self.BASE, RecServeParams())
        cfg = inc.decide(Obs(epoch=3, block_id=9))
        d = cfg.to_dict()
        self.assertEqual((d["arm_id"], d["router"], d["profile"], d["batch_cap"]), ("RecServe", "RecServe", "q35-4b-fp16", 8))
        self.assertFalse(d["escalate_family2"])
        self.assertEqual(d["base_arm"], "q35-4b-fp16|b8|k0")
        self.assertEqual(RecServeParams.from_router_params(d["router_params"]), RecServeParams())
        json.dumps(d)

    def test_reset_empties_the_queues(self):
        inc = RecServeIncumbent(self.BASE, RecServeParams())
        inc.after_edge(req(), {"confidence": 0.4}, {})
        self.assertEqual(inc.diagnostics()["queues"]["f2"]["n_hist"], 1)
        inc.reset(block_id=4)
        self.assertEqual(inc.diagnostics()["queues"], {})

    def test_hooks_delegate_to_the_router(self):
        inc = RecServeIncumbent(self.BASE, rp(beta=0.5))
        self.assertEqual(inc.route(req(), {}), "edge")
        inc.after_edge(req(rid="a"), {"confidence": 0.9}, {})
        self.assertEqual(inc.after_edge(req(rid="b"), {"confidence": 0.1}, {}), "escalate")
        self.assertEqual(inc.diagnostics()["regime"], "RecServe")


class FactoryTests(unittest.TestCase):
    KW = {"base_profile": "q35-4b-fp16", "base_batch_cap": 8}

    def test_headline_and_all_variants(self):
        a = make_incumbent("RecServe", **self.KW)
        b = make_incumbent("RecServe-all", beta=0.5, **self.KW)
        da, db = a.decide({"epoch": 1, "arrival_rate": 0.3, "block_id": 2}), b.decide({"epoch": 1, "block_id": 2})
        self.assertEqual(da["router"], "RecServe")
        self.assertEqual(da["router_params"]["escalable_families"], ["f2"])
        self.assertEqual(da["router_params"]["beta"], 0.3)
        self.assertEqual(db["router_params"]["escalable_families"], ["f1", "f2"])
        self.assertEqual(db["router_params"]["beta"], 0.5)
        self.assertNotIn("arm_id", da)
        json.dumps(da)

    def test_unknown_keyword_raises(self):
        with self.assertRaises(ValueError):
            make_incumbent("RecServe", gamma=1.0, **self.KW)

    def test_base_must_be_admissible_or_selected(self):
        with self.assertRaises(ValueError):
            make_incumbent("RecServe", base_profile="q35-4b-fp16", base_batch_cap=16)
        with self.assertRaises(ValueError):
            make_incumbent("RecServe")


if __name__ == "__main__":
    unittest.main()
