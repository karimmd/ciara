import unittest

from ciara_baselines.base import Obs
from ciara_baselines.i1 import I1E, I1S, I1T, I1Oracle, I1Params, I1Selector, select_base_config
from ciara_baselines.profile import OverheadTable, ProfileTable
from synth import basic_table, guard_table, make_row, mean_lat

TARGETS = {"f1": 2.0, "f2": 15.0}


def params(**kw):
    base = dict(latency_targets_s=TARGETS, profiles=("big-fp16", "mid-int8", "tiny-fp16"), design_load=0.5)
    base.update(kw)
    return I1Params(**base)


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.table = basic_table()
        self.sel = I1Selector(self.table, params())

    def select(self, lam, budget=10.45, sel=None):
        s = sel or self.sel
        return s.select(s.rates(lam, None), budget)

    def ev(self, selection, arm):
        return next(e for e in selection.evaluations if e.cand.arm == arm)

    def test_grid_respects_kv_admissibility(self):
        arms = [c.arm for c in self.sel.candidates()]
        self.assertEqual(len(arms), 10)
        self.assertNotIn("big-fp16|b16|k0", arms)
        self.assertIn("big-fp16|b8|k1", arms)

    def test_requires_latency_targets(self):
        with self.assertRaises(ValueError):
            I1Selector(self.table, I1Params(profiles=("big-fp16",)))

    def test_selects_minimum_latency_feasible_configuration(self):
        s = self.select(0.5)
        self.assertEqual(s.chosen.arm, "mid-int8|b8|k1")
        self.assertEqual(s.reason, "argmin")
        self.assertEqual(len(s.feasible), 4)
        link = 0.05 + 32320 / 100e6
        want = 0.5 * 1.3 + 0.5 * (0.245 + link)
        self.assertAlmostEqual(s.evaluation.mean_latency, want, places=9)

    def test_tie_between_batch_caps_goes_to_lower_memory(self):
        s = self.select(0.5)
        self.assertEqual(s.chosen.batch_cap, 8)
        self.assertLess(self.ev(s, "mid-int8|b8|k1").mem_gib, self.ev(s, "mid-int8|b16|k1").mem_gib)

    def test_accuracy_floor_excludes_small_model(self):
        s = self.select(0.5)
        e = self.ev(s, "tiny-fp16|b8|k0")
        self.assertFalse(e.acc_ok)
        self.assertIn("accuracy:f1", e.reasons)
        e1 = self.ev(s, "tiny-fp16|b8|k1")
        self.assertEqual([r for r in e1.reasons if r.startswith("accuracy")], ["accuracy:f1"])

    def test_floor_is_reference_minus_delta(self):
        strict = I1Selector(self.table, params(delta=0.01, latency_targets_s={"f1": 4.0, "f2": 30.0}))
        s = strict.select(strict.rates(0.5, None), 10.45)
        e = self.ev(s, "mid-int8|b8|k0")
        self.assertFalse(e.acc_ok)
        self.assertEqual(s.chosen.profile, "big-fp16")

    def test_latency_target_excludes_slow_profile(self):
        s = self.select(0.5)
        e = self.ev(s, "big-fp16|b8|k0")
        self.assertFalse(e.lat_ok)
        self.assertIn("latency:f1", e.reasons)
        self.assertAlmostEqual(e.p95_by_family["f1"], 2.0 + 59 / 50.0)

    def test_stability_filter_uses_sustained_throughput(self):
        s = self.select(15.0)
        self.assertEqual(s.chosen.arm, "mid-int8|b16|k1")
        self.assertEqual(len(s.feasible), 1)
        e8 = self.ev(s, "mid-int8|b8|k1")
        self.assertIn("throughput", e8.reasons)
        self.assertAlmostEqual(e8.throughput, (8 / 1.3) / 0.5, places=9)

    def test_infeasible_set_falls_back_to_cloud_configuration_with_lowest_p95(self):
        s = self.select(30.0)
        self.assertEqual(s.reason, "fallback_cloud")
        self.assertEqual(s.chosen.kappa, 1)
        self.assertEqual(s.chosen.arm, "tiny-fp16|b16|k1")
        self.assertEqual(s.feasible, [])

    def test_infeasible_set_without_cloud_falls_back_to_most_accurate_that_fits(self):
        sel = I1Selector(self.table, params(kappas=(0,)))
        s = sel.select(sel.rates(30.0, None), 10.45)
        self.assertEqual(s.reason, "fallback_accuracy")
        self.assertEqual(s.chosen.arm, "big-fp16|b8|k0")

    def test_memory_budget_filter(self):
        s = self.select(0.5, budget=9.6)
        self.assertIn("memory", self.ev(s, "mid-int8|b16|k1").reasons)
        self.assertTrue(self.ev(s, "mid-int8|b8|k1").mem_ok)
        self.assertEqual(s.chosen.arm, "mid-int8|b8|k1")
        s2 = self.select(0.5, budget=9.0)
        self.assertEqual(s2.reason, "fallback_min_memory")

    def test_unprofiled_family_blocks_edge_placement_only(self):
        t = basic_table()
        t.edge["mid-int8"] = make_row("mid-int8", t0=0.5, d=100.0, acc_f1=0.78, f2=None)
        sel = I1Selector(t, params())
        s = sel.select(sel.rates(0.5, None), 10.45)
        e0 = next(e for e in s.evaluations if e.cand.arm == "mid-int8|b8|k0")
        e1 = next(e for e in s.evaluations if e.cand.arm == "mid-int8|b8|k1")
        self.assertFalse(e0.feasible)
        self.assertTrue(any(r.startswith("unprofiled:f2") for r in e0.reasons))
        self.assertTrue(e1.feasible)

    def test_observed_family_split_overrides_the_mix(self):
        r = self.sel.rates(1.0, {"f1": 0.9, "f2": 0.1})
        self.assertEqual(r, {"f1": 0.9, "f2": 0.1})
        r2 = self.sel.rates(1.0, None)
        self.assertEqual(r2, {"f1": 0.5, "f2": 0.5})


class TieBreakTests(unittest.TestCase):
    def build(self, rows, **kw):
        t = ProfileTable(edge={r.ckpt: r for r in rows}, cloud=None, reference=rows[0].ckpt)
        p = I1Params(latency_targets_s={"f1": 20.0, "f2": 80.0}, profiles=tuple(r.ckpt for r in rows),
                     batch_caps=(8,), kappas=(0,), **kw)
        sel = I1Selector(t, p)
        return sel.select(sel.rates(0.1, None), 10.45)

    def test_lower_memory_wins_inside_the_one_percent_band(self):
        m1 = make_row("m1", t0=0.50, d=100.0, kv_gib=1.5)
        m2 = make_row("m2", t0=0.51, d=100.0, kv_gib=3.0)
        self.assertEqual(self.build([m1, m2]).chosen.profile, "m2")
        self.assertEqual(self.build([m1, m2], tie_margin=0.0).chosen.profile, "m1")

    def test_higher_accuracy_wins_when_memory_ties(self):
        m1 = make_row("m1", t0=0.70, d=100.0, acc_f1=0.80)
        m2 = make_row("m2", t0=0.505, d=100.0, acc_f1=0.78)
        m3 = make_row("m3", t0=0.505, d=100.0, acc_f1=0.79)
        self.assertEqual(self.build([m1, m2, m3]).chosen.profile, "m3")

    def test_grid_order_is_the_last_key(self):
        m1 = make_row("m1", t0=0.5, d=100.0)
        m2 = make_row("m2", t0=0.5, d=100.0)
        m3 = make_row("m3", t0=0.5, d=100.0)
        self.assertEqual(self.build([m1, m2, m3]).chosen.profile, "m1")
        self.assertEqual(self.build([m3, m2, m1]).chosen.profile, "m3")


class I1EGuardTests(unittest.TestCase):
    def make(self, overhead_s=20.0, **kw):
        t = guard_table(overhead_s)
        p = I1Params(latency_targets_s={"f1": 10.0, "f2": 40.0}, profiles=("ref", "slow", "fast"),
                     kappas=(0,), **kw)
        return t, I1E(t, p)

    def test_relative_gain_value(self):
        t, _ = self.make()
        slow = 0.5 * mean_lat(t.edge["slow"], "f1") + 0.5 * mean_lat(t.edge["slow"], "f2")
        fast = 0.5 * mean_lat(t.edge["fast"], "f1") + 0.5 * mean_lat(t.edge["fast"], "f2")
        self.assertAlmostEqual(slow, 5.9625)
        self.assertAlmostEqual(fast, 2.445)
        self.assertAlmostEqual((slow - fast) / slow * 30.0, 17.7, places=1)

    def test_guard_keeps_current_when_gain_times_epoch_is_below_switch_cost(self):
        _, inc = self.make(overhead_s=20.0)
        c1 = inc.decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=9.8))
        self.assertEqual((c1.profile, c1.batch_cap), ("slow", 8))
        c2 = inc.decide(Obs(epoch=2, arrival_rate=0.5, memory_budget_gib=10.45))
        self.assertEqual(c2.profile, "slow")
        self.assertEqual(inc.diagnostics()["guard"], "kept_by_guard")
        self.assertEqual(inc.diagnostics()["proposed"].split("|")[0], "fast")

    def test_guard_switches_when_gain_times_epoch_exceeds_switch_cost(self):
        _, inc = self.make(overhead_s=10.0)
        inc.decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=9.8))
        c2 = inc.decide(Obs(epoch=2, arrival_rate=0.5, memory_budget_gib=10.45))
        self.assertEqual(c2.profile, "fast")
        self.assertEqual(inc.diagnostics()["guard"], "switched")

    def test_zero_cost_between_pair_switches_on_any_gain(self):
        t, inc = self.make(overhead_s=1e9)
        t.overhead = OverheadTable(default_s=1e9, pairs={"slow>fast": 0.0})
        inc.decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=9.8))
        c2 = inc.decide(Obs(epoch=2, arrival_rate=0.5, memory_budget_gib=10.45))
        self.assertEqual(c2.profile, "fast")

    def test_current_configuration_that_fails_rule_3_is_replaced_without_the_guard(self):
        _, inc = self.make(overhead_s=1e9)
        c1 = inc.decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=10.45))
        self.assertEqual(c1.profile, "fast")
        c2 = inc.decide(Obs(epoch=2, arrival_rate=0.5, memory_budget_gib=9.8))
        self.assertEqual(c2.profile, "slow")
        self.assertEqual(inc.diagnostics()["guard"], "forced_current_infeasible")

    def test_memory_budget_change_triggers_evaluation_between_periodic_ones(self):
        _, inc = self.make(overhead_s=5.0, reselect_every_epochs=5)
        inc.decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=9.8))
        c2 = inc.decide(Obs(epoch=2, arrival_rate=0.5, memory_budget_gib=9.8))
        self.assertEqual(c2.profile, "slow")
        c3 = inc.decide(Obs(epoch=3, arrival_rate=0.5, memory_budget_gib=10.45))
        self.assertEqual(c3.profile, "fast")

    def test_no_evaluation_between_periods_without_a_trigger(self):
        _, inc = self.make(overhead_s=5.0, reselect_every_epochs=5)
        inc.decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=10.45))
        before = inc.diagnostics()["epoch"]
        inc.decide(Obs(epoch=2, arrival_rate=50.0, memory_budget_gib=10.45))
        self.assertEqual(inc.diagnostics()["epoch"], before)

    def test_information_boundary_telemetry_is_not_read(self):
        _, a = self.make()
        _, b = self.make()
        x = a.decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=10.45, telemetry={}))
        y = b.decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=10.45,
                         telemetry={"queue_len": 500, "p95_latency_s": 99.0, "gpu_util": 1.0}))
        self.assertEqual(x.to_dict(), y.to_dict())

    def test_reset_clears_the_configuration_in_force(self):
        _, inc = self.make(overhead_s=1e9)
        inc.decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=9.8))
        inc.reset(block_id=2)
        c = inc.decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=10.45, block_id=2))
        self.assertEqual(c.profile, "fast")
        self.assertEqual(c.block_id, 2)

    def test_config_object_fields(self):
        _, inc = self.make()
        c = inc.decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=10.45, block_id=17))
        d = c.to_dict()
        self.assertEqual(d["arm_id"], "I1-E")
        self.assertEqual(d["router"], "kappa")
        self.assertEqual(d["base_arm"], f"{c.profile}|b{c.batch_cap}|k0")
        self.assertFalse(d["escalate_family2"])


class VariantTests(unittest.TestCase):
    def test_static_selector_ignores_later_observations(self):
        t = guard_table()
        p = I1Params(latency_targets_s={"f1": 10.0, "f2": 40.0}, profiles=("ref", "slow", "fast"), kappas=(0,),
                     design_load=0.5)
        s = I1S(t, p)
        c1 = s.decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=10.45))
        c2 = s.decide(Obs(epoch=9, arrival_rate=100.0, memory_budget_gib=1.0, telemetry={"queue_len": 99}))
        self.assertEqual((c1.profile, c1.batch_cap), (c2.profile, c2.batch_cap))
        self.assertEqual(c1.arm_id, "I1-S")
        self.assertEqual(select_base_config(t, p).base_arm, c1.base_arm)

    def test_static_selector_needs_design_load(self):
        with self.assertRaises(ValueError):
            I1S(guard_table(), I1Params(latency_targets_s={"f1": 10.0, "f2": 40.0}, profiles=("ref",)))

    def test_throughput_variant_takes_maximum_sustained_throughput(self):
        t = guard_table()
        p = I1Params(latency_targets_s={"f1": 10.0, "f2": 40.0}, profiles=("ref", "slow", "fast"), kappas=(0,))
        e = I1E(t, p).decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=10.45))
        v = I1T(t, p).decide(Obs(epoch=1, arrival_rate=0.5, memory_budget_gib=10.45))
        self.assertEqual((e.profile, e.batch_cap), ("fast", 8))
        self.assertEqual((v.profile, v.batch_cap), ("fast", 16))
        self.assertEqual(v.arm_id, "I1-T")

    def test_oracle_uses_its_own_table(self):
        nominal = guard_table(overhead_s=5.0)
        co_located = guard_table(overhead_s=5.0)
        co_located.edge["fast"] = make_row("fast", t0=2.0, d=10.0, kv_tokens=1000000, kv_gib=0.5)
        p = I1Params(latency_targets_s={"f1": 100.0, "f2": 400.0}, profiles=("ref", "slow", "fast"), kappas=(0,))
        a = I1E(nominal, p).decide(Obs(epoch=1, arrival_rate=0.1, memory_budget_gib=10.45))
        b = I1Oracle(co_located, p).decide(Obs(epoch=1, arrival_rate=0.1, memory_budget_gib=10.45))
        self.assertEqual(a.profile, "fast")
        self.assertEqual(b.profile, "slow")
        self.assertEqual(b.arm_id, "I1-oracle")


if __name__ == "__main__":
    unittest.main()
