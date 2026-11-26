import json
import unittest

from ciara_baselines.base import Config, Obs
from ciara_baselines.mixorsplit import MixOrSplitIncumbent, MixOrSplitRouter, MosParams
from ciara_baselines.profile import ProfileTable
from synth import basic_table, make_row

def mp(**kw):
    kw.setdefault("eligible_families", ("f1", "f2"))
    return MosParams(**kw)


BASE = Config("I1-S", "mid-int8", 8, False, base_arm="mid-int8|b8|k0")


def req(rid="r", fam=1, **kw):
    m = {"req_id": rid, "family": fam, "max_tokens": 512, "messages": []}
    m.update(kw)
    return m


def link(prompt, out, rtt, mbps=100.0):
    return rtt + 32.0 * (prompt + out) / (mbps * 1e6)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.table = basic_table()
        self.r = MixOrSplitRouter(self.table, BASE, mp())

    def modes(self, d):
        return {e.mode: e for e in d.evals}

    def test_predictors_match_the_profile(self):
        d = self.r.decide_mode(req(), {})
        m = self.modes(d)
        self.assertAlmostEqual(m["MIX"].pred.total_s, 1.0 + 30 * 0.01)
        self.assertAlmostEqual(m["MIX"].pred.ttft_s, 1.0)
        self.assertAlmostEqual(m["MIX"].pred.tpot_s, 0.01)
        self.assertAlmostEqual(m["CLOUD"].pred.total_s, 0.4 + 30 * 0.005 + link(2000, 31, 0.05), places=9)
        self.assertEqual(set(m), {"MIX", "CLOUD"})

    def test_minimum_predicted_latency_among_feasible_modes(self):
        self.assertEqual(self.r.decide_mode(req(), {}).route, "cloud")
        self.assertEqual(self.r.decide_mode(req(), {"backhaul_rtt_s": 2.0}).route, "edge")

    def test_edge_queue_shifts_the_choice_to_the_cloud(self):
        ns = {"backhaul_rtt_s": 0.8}
        self.assertEqual(self.r.decide_mode(req(), ns).route, "edge")
        ns = {"backhaul_rtt_s": 0.8, "inflight": 8, "queue_len": 8}
        d = self.r.decide_mode(req(), ns)
        self.assertAlmostEqual(self.modes(d)["MIX"].pred.total_s, 9 * 1.3 / 8 + 1.3)
        self.assertEqual(d.route, "cloud")

    def test_tpot_gate_removes_the_faster_mode(self):
        r = MixOrSplitRouter(self.table, BASE, mp(tpot_slo_s={"f1": 0.0075}))
        d = r.decide_mode(req(), {"backhaul_rtt_s": 0.8})
        self.assertFalse(self.modes(d)["MIX"].tpot_ok)
        self.assertEqual(d.route, "cloud")
        self.assertFalse(d.infeasible)

    def test_ttft_gate(self):
        r = MixOrSplitRouter(self.table, BASE, mp(ttft_slo_s={"f1": 0.6}))
        d = r.decide_mode(req(), {"backhaul_rtt_s": 0.1})
        self.assertFalse(self.modes(d)["MIX"].ttft_ok)
        self.assertEqual(d.route, "cloud")

    def test_deadline_gate_uses_elapsed_time(self):
        ns = {"backhaul_rtt_s": 0.8}
        self.assertEqual(self.r.decide_mode(req(deadline_ms=1320), ns).route, "edge")
        d = self.r.decide_mode(req(deadline_ms=1320, elapsed_s=0.1), ns)
        self.assertTrue(d.infeasible)

    def test_infeasible_set_falls_back_to_minimum_latency_and_is_flagged(self):
        ns = {"backhaul_rtt_s": 0.8}
        d = self.r.decide_mode(req(deadline_ms=500), ns)
        self.assertTrue(d.infeasible)
        self.assertEqual(d.route, "edge")
        self.assertEqual(self.r.decide_mode(req(deadline_ms=500), {"backhaul_rtt_s": 0.0}).route, "cloud")

    def test_reject_policy(self):
        r = MixOrSplitRouter(self.table, BASE, mp(infeasible_policy="reject"))
        d = r.decide_mode(req(deadline_ms=500), {})
        self.assertEqual((d.mode, d.route, d.infeasible), ("REJECT", "reject", True))

    def test_switching_engine_makes_mix_unavailable(self):
        d = self.r.decide_mode(req(), {"switching": True, "backhaul_rtt_s": 2.0})
        self.assertFalse(self.modes(d)["MIX"].available)
        self.assertEqual(d.route, "cloud")

    def test_nothing_available_queues_at_the_edge(self):
        r = MixOrSplitRouter(self.table, BASE, mp(eligible_families=("f2",)))
        d = r.decide_mode(req(fam=1), {"switching": True})
        self.assertEqual((d.route, d.infeasible), ("edge", True))

    def test_kv_slot_gate_makes_mix_unavailable(self):
        base = Config("I1-S", "big-fp16", 8, False)
        r = MixOrSplitRouter(self.table, base, mp())
        ok = r.decide_mode(req(prompt_tokens=2000, max_tokens=512), {"backhaul_rtt_s": 5.0})
        self.assertTrue(self.modes(ok)["MIX"].available)
        bad = r.decide_mode(req(prompt_tokens=2000, max_tokens=4000), {"backhaul_rtt_s": 5.0})
        self.assertFalse(self.modes(bad)["MIX"].available)
        self.assertEqual(bad.route, "cloud")

    def test_ineligible_family_stays_at_the_edge(self):
        r = MixOrSplitRouter(self.table, BASE, mp(eligible_families=("f1",)))
        self.assertEqual(r.decide_mode(req(fam=2), {}).route, "edge")
        self.assertFalse(self.modes(r.decide_mode(req(fam=2), {}))["CLOUD"].available)

    def test_output_length_comes_from_the_model_of_each_mode(self):
        d = self.r.decide_mode(req(fam=2), {})
        m = self.modes(d)
        self.assertAlmostEqual(m["MIX"].pred.total_s, 0.5 + 419 * 0.01)
        self.assertAlmostEqual(m["CLOUD"].pred.total_s, 0.2 + 9 * 0.005 + link(1000, 10, 0.05), places=9)
        cap = MixOrSplitRouter(self.table, BASE, mp(output_tokens="request_cap"))
        m2 = {e.mode: e for e in cap.decide_mode(req(fam=2, max_tokens=1000), {}).evals}
        self.assertAlmostEqual(m2["MIX"].pred.total_s, 0.5 + 999 * 0.01)

    def test_family_two_gate_uses_family_two_slo(self):
        r = MixOrSplitRouter(self.table, BASE, mp(tpot_slo_s={"f1": 1.0, "f2": 0.0075}))
        d = r.decide_mode(req(fam=2), {"backhaul_rtt_s": 9.0})
        self.assertFalse(self.modes(d)["MIX"].tpot_ok)


class TieBreakTests(unittest.TestCase):
    def setUp(self):
        edge = make_row("mid", t0=0.5, d=100.0)
        cloud = make_row("cl", site="cloud", t0=0.5, d=100.0, kv_tokens=200000, kv_gib=4.0, budget=8.8)
        cloud.families["f1"] = edge.families["f1"]
        self.table = ProfileTable(edge={"mid": edge}, cloud=cloud, reference="mid")
        self.base = Config("I1-S", "mid", 8, False)
        self.r = MixOrSplitRouter(self.table, self.base, mp(backhaul_mbps=1e18, backhaul_rtt_s=0.0))

    def test_equal_latency_goes_to_the_larger_guarded_residual_capacity(self):
        d = self.r.decide_mode(req(), {})
        m = {e.mode: e for e in d.evals}
        self.assertAlmostEqual(m["MIX"].pred.total_s, m["CLOUD"].pred.total_s, places=9)
        self.assertEqual((m["MIX"].residual, m["CLOUD"].residual), (8.0, 16.0))
        self.assertTrue(d.tie_break)
        self.assertEqual(d.route, "cloud")
        d2 = self.r.decide_mode(req(), {"cloud_inflight": 10})
        self.assertEqual(d2.route, "edge")

    def test_equal_residual_falls_to_the_fixed_mode_order(self):
        d = self.r.decide_mode(req(), {"cloud_inflight": 8})
        self.assertEqual(d.route, "edge")
        self.assertEqual(d.mode, "MIX")

    def test_guard_slots_reduce_the_residual(self):
        r = MixOrSplitRouter(self.table, self.base, mp(backhaul_mbps=1e18, backhaul_rtt_s=0.0, guard_slots=9))
        d = r.decide_mode(req(), {})
        m = {e.mode: e.residual for e in d.evals}
        self.assertEqual(m, {"MIX": -1.0, "CLOUD": 7.0})


class IncumbentTests(unittest.TestCase):
    def test_hooks_and_config(self):
        seen = []
        inc = MixOrSplitIncumbent(basic_table(), BASE, mp(), on_decision=lambda m, d: seen.append(d.route))
        self.assertEqual(inc.route(req(), {}), "cloud")
        self.assertEqual(inc.route(req(), {"backhaul_rtt_s": 2.0}), "edge")
        self.assertEqual(seen, ["cloud", "edge"])
        self.assertEqual(inc.after_edge(req(), "reply", {}), "accept")
        c = inc.decide(Obs(epoch=1, block_id=3))
        d = c.to_dict()
        self.assertEqual((d["arm_id"], d["router"], d["profile"], d["batch_cap"]), ("I3", "I3", "mid-int8", 8))
        self.assertEqual(d["router_params"]["modes"], ["MIX", "CLOUD"])
        json.dumps(d)
        json.dumps(inc.diagnostics())

    def test_router_params_roundtrip(self):
        p = mp(ttft_slo_s={"f1": 1.0}, tpot_slo_s={"f1": 0.075, "f2": 0.05}, eligible_families=("f2",),
                      infeasible_policy="reject")
        q = MosParams.from_router_params(json.loads(json.dumps(p.to_router_params())))
        self.assertEqual(p, q)

    def test_cloud_mode_needs_a_cloud_row(self):
        t = basic_table()
        t.cloud = None
        with self.assertRaises(ValueError):
            MixOrSplitRouter(t, BASE, mp())
        MixOrSplitRouter(t, BASE, mp(modes=("MIX",)))


if __name__ == "__main__":
    unittest.main()
