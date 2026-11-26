import json
import math
import random
import unittest

from ciara_baselines.base import Config, Obs
from ciara_baselines.race import (
    RaceCalibrator, RaceIncumbent, RaceParams, RaceRouter, SepProbe, build_probe_request, canonical_form,
    probe_log_record, semantic_entropy,
)
from synth import basic_table

def rp(**kw):
    kw.setdefault("escalable_families", ("f1", "f2"))
    return RaceParams(**kw)


P = rp(alpha={"f1": 0.3, "f2": 0.3})


def reference_algorithm_1(u, wrong, alpha=0.30, tau0=0.5, eta0=0.1, rho=0.99, window=100, c_lo=0.8, c_hi=1.2,
                          eps=0.01, use_budget=True):
    n = len(u)
    tau, eta, b_rem = tau0, eta0, alpha * n
    win, decisions, taus = [], [], []
    for t in range(n):
        b_t = b_rem / (n - t)
        if use_budget and b_t < alpha / 2:
            tt = c_lo * tau
        elif use_budget and b_t > 1.5 * alpha:
            tt = min(1 - eps, c_hi * tau)
        else:
            tt = tau
        cloud = u[t] > tt
        decisions.append(cloud)
        if not cloud:
            if wrong[t]:
                b_rem -= 1
            win.append(1 if wrong[t] else 0)
            if len(win) > window:
                win.pop(0)
        if win:
            tau = min(1 - eps, max(eps, tau - eta * (sum(win) / len(win) - alpha)))
        eta *= rho
        taus.append(tau)
    return decisions, taus


class CalibratorTests(unittest.TestCase):
    def test_matches_algorithm_1_on_a_random_stream(self):
        rng = random.Random(7)
        n = 600
        u = [rng.random() for _ in range(n)]
        wrong = [rng.random() < 0.15 + 0.6 * x for x in u]
        for use_budget in (True, False):
            want_dec, want_tau = reference_algorithm_1(u, wrong, use_budget=use_budget)
            cal = RaceCalibrator(0.30, rp(alpha={"f1": 0.3}, use_budget=use_budget), horizon=n)
            got_dec, got_tau = [], []
            for t in range(n):
                d = cal.decide(u[t])
                got_dec.append(d.offload)
                if not d.offload:
                    cal.feedback(d.step, bool(wrong[t]))
                got_tau.append(cal.tau)
            self.assertEqual(got_dec, want_dec)
            for a, b in zip(got_tau, want_tau):
                self.assertAlmostEqual(a, b, places=12)

    def test_threshold_update_by_hand(self):
        cal = RaceCalibrator(0.3, P, horizon=100)
        d1 = cal.decide(0.4)
        self.assertFalse(d1.offload)
        self.assertEqual(d1.mode, "normal")
        self.assertAlmostEqual(d1.tau_tilde, 0.5)
        cal.feedback(d1.step, True)
        self.assertAlmostEqual(cal.tau, 0.43)
        self.assertAlmostEqual(cal.eta, 0.099)
        self.assertAlmostEqual(cal.b_rem, 0.3 * 100 - 1)
        d2 = cal.decide(0.6)
        self.assertTrue(d2.offload)
        self.assertAlmostEqual(cal.tau, 0.43 - 0.099 * 0.7)
        self.assertAlmostEqual(cal.eta, 0.09801)
        t2 = cal.tau
        d3 = cal.decide(0.1)
        self.assertFalse(d3.offload)
        cal.feedback(d3.step, False)
        self.assertAlmostEqual(cal.tau, t2 - 0.09801 * (0.5 - 0.3))
        self.assertEqual(cal.state()["window"], 2)

    def test_threshold_rises_when_window_error_is_below_alpha(self):
        cal = RaceCalibrator(0.3, P, horizon=100)
        for _ in range(5):
            d = cal.decide(0.0)
            cal.feedback(d.step, False)
        self.assertGreater(cal.tau, 0.5)

    def test_projection_onto_eps_interval(self):
        cal = RaceCalibrator(0.3, rp(alpha={"f1": 0.3}, eta0=10.0), horizon=100)
        d = cal.decide(0.0)
        cal.feedback(d.step, True)
        self.assertAlmostEqual(cal.tau, 0.01)
        cal2 = RaceCalibrator(0.3, rp(alpha={"f1": 0.3}, eta0=10.0), horizon=100)
        d = cal2.decide(0.0)
        cal2.feedback(d.step, False)
        self.assertAlmostEqual(cal2.tau, 0.99)

    def test_empty_window_skips_the_update_but_decays_eta(self):
        cal = RaceCalibrator(0.3, P, horizon=100)
        d = cal.decide(0.99)
        self.assertTrue(d.offload)
        self.assertAlmostEqual(cal.tau, 0.5)
        self.assertAlmostEqual(cal.eta, 0.099)

    def test_envelope_conservative_aggressive_and_neutral(self):
        cal = RaceCalibrator(0.3, P, horizon=10)
        cal.b_rem = 0.5
        tt, b, mode = cal.threshold()
        self.assertEqual(mode, "conservative")
        self.assertAlmostEqual(tt, 0.8 * 0.5)
        cal.b_rem = 10.0
        tt, b, mode = cal.threshold()
        self.assertEqual(mode, "aggressive")
        self.assertAlmostEqual(tt, min(0.99, 1.2 * 0.5))
        cal.tau = 0.9
        tt, _, _ = cal.threshold()
        self.assertAlmostEqual(tt, 0.99)
        cal.b_rem = 3.0
        cal.tau = 0.5
        self.assertEqual(cal.threshold()[2], "normal")

    def test_envelope_boundaries_are_open(self):
        cal = RaceCalibrator(0.5, rp(alpha={"f1": 0.5}), horizon=10)
        cal.b_rem = 2.5
        self.assertEqual(cal.threshold()[2], "normal")
        cal.b_rem = 7.5
        self.assertEqual(cal.threshold()[2], "normal")
        cal.b_rem = 2.4
        self.assertEqual(cal.threshold()[2], "conservative")
        cal.b_rem = 7.6
        self.assertEqual(cal.threshold()[2], "aggressive")

    def test_no_budget_variant_leaves_threshold_unadjusted(self):
        cal = RaceCalibrator(0.3, rp(alpha={"f1": 0.3}, use_budget=False), horizon=10)
        cal.b_rem = 0.0
        self.assertEqual(cal.threshold()[2], "normal")

    def test_budget_is_consumed_by_edge_errors_only(self):
        cal = RaceCalibrator(0.3, P, horizon=10)
        d = cal.decide(0.1)
        cal.feedback(d.step, True)
        self.assertAlmostEqual(cal.b_rem, 3.0 - 1.0)
        d = cal.decide(0.99)
        self.assertAlmostEqual(cal.b_rem, 2.0)
        self.assertAlmostEqual(cal.threshold()[1], 2.0 / (10 - 3 + 1))

    def test_envelope_is_off_after_the_horizon(self):
        cal = RaceCalibrator(0.3, P, horizon=2)
        for _ in range(2):
            d = cal.decide(0.99)
        tt, b, mode = cal.threshold()
        self.assertEqual(mode, "normal")

    def test_delayed_feedback_completes_steps_in_label_order(self):
        cal = RaceCalibrator(0.3, P, horizon=100)
        a = cal.decide(0.1)
        b = cal.decide(0.2)
        self.assertEqual(cal.state()["pending"], 2)
        self.assertTrue(cal.feedback(b.step, False))
        self.assertTrue(cal.feedback(a.step, True))
        self.assertFalse(cal.feedback(a.step, True))
        self.assertEqual(cal.state()["window"], 2)
        self.assertEqual(cal.state()["n_edge_err"], 1)

    def test_label_none_completes_a_step_without_touching_the_window(self):
        cal = RaceCalibrator(0.3, P, horizon=100)
        d = cal.decide(0.1)
        cal.feedback(d.step, None)
        self.assertEqual(cal.state()["window"], 0)
        self.assertAlmostEqual(cal.eta, 0.099)
        self.assertEqual(cal.state()["pending"], 0)

    def test_window_keeps_only_the_last_w_losses(self):
        cal = RaceCalibrator(0.3, rp(alpha={"f1": 0.3}, window=3), horizon=100)
        for err in (True, True, True, False, False, False):
            d = cal.decide(0.0)
            cal.feedback(d.step, err)
        self.assertEqual(cal.state()["window_error"], 0.0)


class ScoreTests(unittest.TestCase):
    def test_semantic_entropy_extremes_and_split(self):
        self.assertEqual(semantic_entropy(["a"] * 5), 0.0)
        self.assertAlmostEqual(semantic_entropy(["a", "b", "c", "d", "e"]), 1.0)
        want = -(0.6 * math.log(0.6) + 0.4 * math.log(0.4)) / math.log(5)
        self.assertAlmostEqual(semantic_entropy(["a", "a", "a", "b", "b"]), want)
        self.assertEqual(semantic_entropy(["only one"]), 0.0)

    def test_empty_reply_is_a_class_of_its_own(self):
        self.assertGreater(semantic_entropy(["a", "a", "a", "a", ""]), 0.0)

    def test_canonical_form_equates_formatting_and_separates_meaning(self):
        a = '```json\n{"name": "light.turn_on", "arguments": {"Entity": "Kitchen", "brightness": 50.0}}\n```'
        b = '{"arguments":{"entity":"kitchen","brightness":50},"name":"light.turn_on"}'
        c = '{"name": "light.turn_on", "arguments": {"entity": "kitchen", "brightness": 60}}'
        self.assertEqual(canonical_form(a), canonical_form(b))
        self.assertNotEqual(canonical_form(a), canonical_form(c))

    def test_canonical_form_takes_the_last_object_after_working(self):
        r1 = 'The mean is (1+2+3)/3 = 2, so {"mean": 2}'
        r2 = '<think>try {"x": 1}</think> Average: {"mean": 2.0}'
        self.assertEqual(canonical_form(r1), canonical_form(r2))

    def test_canonical_form_falls_back_to_normalized_text(self):
        self.assertEqual(canonical_form("  Yes,  it IS "), canonical_form("yes, it is"))

    def test_sep_probe_forward_pass(self):
        probe = SepProbe.one_hidden(w1=[[1.0, 0.0], [0.0, 1.0]], b1=[0.0, 0.0], w2=[1.0, 1.0], b2=-1.0)
        self.assertAlmostEqual(probe([1.0, 2.0]), 1 / (1 + math.exp(-2.0)))
        self.assertAlmostEqual(probe([-5.0, 1.0]), 1 / (1 + math.exp(0.0)))
        with self.assertRaises(ValueError):
            SepProbe.one_hidden([[1.0]], [0.0, 0.0], [1.0], 0.0)

    def test_sep_probe_with_two_hidden_layers_and_without_numpy(self):
        import ciara_baselines.race as race
        layers = [([[1.0, -1.0], [0.5, 0.5]], [0.0, 0.0]), ([[2.0, 1.0], [0.0, 1.0]], [0.0, -1.0]), ([1.0, 1.0], [0.0])]
        probe = SepProbe(layers)
        want = 1 / (1 + math.exp(-7.0))
        self.assertAlmostEqual(probe([3.0, 1.0]), want)
        saved = race._np
        race._np = None
        try:
            self.assertAlmostEqual(probe([3.0, 1.0]), want)
        finally:
            race._np = saved

    def test_probe_request_and_cost_record(self):
        meta = {"req_id": "r-1", "family": 1, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 512,
                "model": "q35-4b-fp16"}
        req = build_probe_request(meta, P)
        body = req["body"]
        self.assertEqual((body["n"], body["temperature"], body["max_tokens"], body["stream"]), (5, 0.7, 512, False))
        self.assertEqual(body["messages"], meta["messages"])
        self.assertEqual(body["seed"], build_probe_request(meta, P)["body"]["seed"])
        self.assertEqual(req["headers"]["X-Ciara-Kind"], "race_probe")
        self.assertEqual(req["headers"]["X-Ciara-Req"], "r-1.probe")
        capped = build_probe_request(meta, rp(alpha={"f1": 0.3}, probe_max_tokens=64))
        self.assertEqual(capped["body"]["max_tokens"], 64)
        rec = probe_log_record(meta, {"n": 5, "prompt_tokens": 2600, "completion_tokens": 150}, 1_000_000_000,
                               1_450_000_000, 0.4)
        self.assertAlmostEqual(rec["race_probe_s"], 0.45)
        self.assertEqual(rec["kind"], "race_probe")
        json.dumps(rec)


def meta(rid, fam=1, u=None):
    m = {"req_id": rid, "family": fam, "messages": []}
    if u is not None:
        m["race_u"] = u
    return m


class RouterTests(unittest.TestCase):
    def test_route_threshold_and_feedback_loop(self):
        labels = {"a": True, "b": False}
        r = RaceRouter(P, feedback_fn=lambda m, reply: labels[m["req_id"]], horizon=100)
        self.assertEqual(r.route(meta("a", u=0.4), {}), "edge")
        self.assertEqual(r.after_edge(meta("a"), "reply", {}), "accept")
        self.assertAlmostEqual(r.cal["f1"].tau, 0.43)
        self.assertEqual(r.route(meta("c", u=0.9), {}), "cloud")
        self.assertEqual(r.after_edge(meta("c"), "reply", {}), "accept")

    def test_missing_score_raises(self):
        r = RaceRouter(P, horizon=100)
        with self.assertRaises(KeyError):
            r.route(meta("x"), {})

    def test_scorer_supplies_the_score(self):
        r = RaceRouter(P, scorer=lambda m: 0.99, horizon=100)
        self.assertEqual(r.route(meta("x"), {}), "cloud")

    def test_one_calibrator_per_family(self):
        r = RaceRouter(P, feedback_fn=lambda m, rep: True, horizon=100)
        r.route(meta("a", fam=1, u=0.1), {})
        r.after_edge(meta("a", fam=1), "x", {})
        r.route(meta("b", fam=2, u=0.1), {})
        self.assertLess(r.cal["f1"].tau, 0.5)
        self.assertAlmostEqual(r.cal["f2"].tau, 0.5)
        self.assertEqual(sorted(r.cal), ["f1", "f2"])

    def test_non_escalable_family_is_served_at_the_edge_without_a_score(self):
        r = RaceRouter(rp(alpha={"f1": 0.3, "f2": 0.3}, escalable_families=("f2",)), horizon=100)
        self.assertEqual(r.route(meta("a", fam=1), {}), "edge")
        self.assertEqual(r.cal, {})

    def test_label_fraction_zero_leaves_the_window_empty(self):
        r = RaceRouter(rp(alpha={"f1": 0.3}, label_fraction=0.0), feedback_fn=lambda m, rep: True, horizon=100)
        r.route(meta("a", u=0.1), {})
        r.after_edge(meta("a"), "x", {})
        self.assertEqual(r.cal["f1"].state()["window"], 0)
        self.assertEqual(r.cal["f1"].state()["pending"], 0)

    def test_late_label_through_label(self):
        r = RaceRouter(P, horizon=100)
        r.route(meta("a", u=0.1), {})
        self.assertEqual(r.after_edge(meta("a"), "x", {}), "accept")
        self.assertEqual(r.cal["f1"].state()["pending"], 1)
        self.assertTrue(r.label("a", True))
        self.assertAlmostEqual(r.cal["f1"].tau, 0.43)
        self.assertFalse(r.label("a", True))

    def test_abandon_completes_the_step(self):
        r = RaceRouter(P, horizon=100)
        r.route(meta("a", u=0.1), {})
        r.abandon(meta("a"))
        self.assertEqual(r.cal["f1"].state()["pending"], 0)

    def test_pooled_calibrator_needs_equal_alphas(self):
        r = RaceRouter(rp(alpha={"f1": 0.2, "f2": 0.3}, per_family=False), horizon=100)
        with self.assertRaises(ValueError):
            r.route(meta("a", u=0.1), {})

    def test_alpha_is_required(self):
        with self.assertRaises(ValueError):
            RaceRouter(RaceParams(), horizon=10)


class IncumbentTests(unittest.TestCase):
    def setUp(self):
        self.base = Config("I1-S", "mid-int8", 8, True, base_arm="mid-int8|b8|k1")

    def test_decide_returns_base_with_race_router(self):
        inc = RaceIncumbent(self.base, P)
        c = inc.decide(Obs(epoch=1, block_id=4))
        d = c.to_dict()
        self.assertEqual((d["arm_id"], d["router"], d["profile"], d["batch_cap"]), ("I2", "I2", "mid-int8", 8))
        self.assertFalse(d["escalate_family2"])
        self.assertEqual(d["router_params"]["window"], 100)
        self.assertEqual(d["router_params"]["probe"]["n"], 5)
        self.assertEqual(d["base_arm"], "mid-int8|b8|k1")
        json.dumps(d)

    def test_reset_starts_fresh_calibrators_with_the_block_horizon(self):
        inc = RaceIncumbent(self.base, P, feedback_fn=lambda m, r: True)
        inc.route(meta("a", u=0.1), {})
        inc.after_edge(meta("a"), "x", {})
        self.assertLess(inc.router.cal["f1"].tau, 0.5)
        inc.reset(block_id=5, horizon=40)
        self.assertEqual(inc.router.cal, {})
        inc.route(meta("b", u=0.1), {})
        self.assertAlmostEqual(inc.router.cal["f1"].T, 40)
        self.assertAlmostEqual(inc.router.cal["f1"].b_rem, 40 * 0.3)

    def test_router_params_roundtrip(self):
        p = rp(alpha={"f1": 0.25, "f2": 0.55}, window=50, escalable_families=("f2",), label_fraction=0.5)
        q = RaceParams.from_router_params(json.loads(json.dumps(p.to_router_params())))
        self.assertEqual(p, q)

    def test_from_table_sets_alpha_from_the_accuracy_floor(self):
        t = basic_table()
        p = RaceParams.from_table(t, delta=0.05)
        self.assertAlmostEqual(p.alpha["f1"], 1 - (0.80 - 0.05))
        self.assertAlmostEqual(p.alpha["f2"], 1 - (0.50 - 0.05))


if __name__ == "__main__":
    unittest.main()
