import json
import math
import os
import random
import tempfile
import unittest

from ciara_baselines.base import Obs
from ciara_baselines.cbandit import (
    DEFAULT_CONSTRAINTS, CBanditCore, CBanditIncumbent, CBanditParams, GatewayOutcomeSource, admissible_grid,
    epoch_outcomes, paper_gamma, rad,
)
from ciara_baselines.factory import make_incumbent
from ciara_baselines.i1 import I1Params, I1Selector
from ciara_baselines.lp import solve_distribution_lp
from ciara_baselines.profile import ProfileTable
from synth import SHIPPED_TABLE as REAL_TABLE, basic_table

SYNTH_PROFILES = ("big-fp16", "mid-int8", "tiny-fp16")


def synth_params(**kw):
    kw.setdefault("profiles", SYNTH_PROFILES)
    return CBanditParams(**kw)


Q6 = [0.5, 0.6, 0.7, 0.8, 0.9, 0.65]
EXC6 = [0.01, 0.03, 0.08, 0.04, 0.30, 0.02]
CLD6 = [0.0, 0.1, 0.2, 0.5, 0.0, 0.25]
BUDGETS6 = {"exc": 0.05, "cld": 0.30}


def run_stationary(T, gamma, eps, seed, noise=0.03):
    rng = random.Random(seed)
    core = CBanditCore(6, "q", BUDGETS6, gamma, eps, random.Random(seed + 1000))
    hist = []
    for _ in range(T):
        i, _p, _s = core.step()
        core.observe(i, {"q": Q6[i] + rng.gauss(0, noise), "exc": EXC6[i] + rng.gauss(0, 0.4 * noise),
                         "cld": CLD6[i] + rng.gauss(0, noise)})
        hist.append(i)
    return hist


def window_means(hist):
    n = len(hist)
    return (sum(Q6[i] for i in hist) / n, sum(EXC6[i] for i in hist) / n, sum(CLD6[i] for i in hist) / n)


class EstimateTests(unittest.TestCase):
    def test_rad_by_hand(self):
        self.assertAlmostEqual(rad(0.45, 10.0, 0.5), math.sqrt(0.5 * 0.45 / 10.0) + 0.05)
        self.assertAlmostEqual(rad(0.0, 4.0, 2.0), 0.5)

    def test_unplayed_arm_is_fully_optimistic(self):
        core = CBanditCore(3, "q", {"c": 0.1}, gamma=1.0)
        for i in range(3):
            self.assertEqual(core.ucb("q", i), 1.0)
            self.assertEqual(core.lcb("c", i), 0.0)

    def test_estimates_eq_25_and_26_by_hand(self):
        core = CBanditCore(2, "q", {"c": 0.1}, gamma=0.5)
        core.sum["q"][0], core.cnt["q"][0] = 4.5, 9
        v = 0.45
        r = math.sqrt(0.5 * v / 10.0) + 0.5 / 10.0
        self.assertAlmostEqual(core.mean("q", 0), v)
        self.assertAlmostEqual(core.ucb("q", 0), v + 2 * 0.2)
        self.assertAlmostEqual(core.lcb("q", 0), v - 2 * 0.2)
        self.assertAlmostEqual(r, 0.2)
        core.sum["c"][1], core.cnt["c"][1] = 0.2, 19
        self.assertEqual(core.lcb("c", 1), 0.0)
        core.sum["q"][1], core.cnt["q"][1] = 19.0, 19
        self.assertEqual(core.ucb("q", 1), 1.0)

    def test_paper_gamma(self):
        self.assertAlmostEqual(paper_gamma(18, 1200, 4, 0.05), math.log(18 * 1200 * 4 / 0.05))
        self.assertAlmostEqual(paper_gamma(18, 1200, 4, 0.05), 14.36, places=2)

    def test_observe_skips_missing_components_and_clips(self):
        core = CBanditCore(2, "q", {"a": 0.1, "b": 0.1}, gamma=1.0)
        n = core.observe(0, {"q": 1.4, "a": None, "b": float("nan")})
        self.assertEqual(n, 1)
        self.assertEqual(core.sum["q"][0], 1.0)
        self.assertEqual(core.cnt["a"][0], 0)
        self.assertEqual(core.cnt["b"][0], 0)
        self.assertEqual(core.observe(1, {}), 0)
        self.assertEqual(core.plays, [1, 0])


class LPStepTests(unittest.TestCase):
    def preload(self, core, means_q, means_c, n=10 ** 9):
        for i, (q, c) in enumerate(zip(means_q, means_c)):
            core.sum["q"][i], core.cnt["q"][i] = q * n, n
            core.sum["c"][i], core.cnt["c"][i] = c * n, n

    def test_the_lp_of_eq_12_by_hand(self):
        core = CBanditCore(2, "q", {"c": 0.05}, gamma=1.0, epsilon=0.0)
        self.preload(core, [0.9, 0.5], [0.06, 0.01])
        p, status = core.solve()
        self.assertEqual(status, "optimal")
        self.assertAlmostEqual(p[0], 0.8, places=3)

    def test_epsilon_shrinks_the_budgets(self):
        core = CBanditCore(2, "q", {"c": 0.05}, gamma=1.0, epsilon=0.2)
        self.preload(core, [0.9, 0.5], [0.06, 0.01])
        p, _ = core.solve()
        self.assertAlmostEqual(p[0], (0.04 - 0.01) / 0.05, places=3)

    def test_the_constraint_row_uses_the_lower_confidence_bound(self):
        core = CBanditCore(2, "q", {"c": 0.05}, gamma=1.0)
        core.sum["q"][0], core.cnt["q"][0] = 9.0, 10
        core.sum["c"][0], core.cnt["c"][0] = 0.8, 10
        core.sum["q"][1], core.cnt["q"][1] = 5.0, 10
        core.sum["c"][1], core.cnt["c"][1] = 0.1, 10
        self.assertEqual(core.lcb("c", 0), 0.0)
        p, status = core.solve()
        self.assertEqual(status, "optimal")
        self.assertAlmostEqual(p[0], 1.0)

    def test_fallback_is_the_smallest_largest_normalized_violation(self):
        core = CBanditCore(3, "q", {"a": 0.05, "b": 0.30}, gamma=1.0)
        n = 10 ** 9
        for i, (q, a, b) in enumerate([(0.9, 0.20, 0.10), (0.5, 0.10, 0.90), (0.7, 0.10, 0.60)]):
            core.sum["q"][i], core.cnt["q"][i] = q * n, n
            core.sum["a"][i], core.cnt["a"][i] = a * n, n
            core.sum["b"][i], core.cnt["b"][i] = b * n, n
        p, status = core.solve()
        self.assertEqual(status, "fallback")
        self.assertEqual(p, [0.0, 0.0, 1.0])
        self.assertEqual(core.last_status, "fallback")

    def test_sample_follows_the_distribution(self):
        core = CBanditCore(3, "q", {"c": 0.5}, gamma=1.0, rng=random.Random(4))
        p = [0.2, 0.5, 0.3]
        draws = [core.sample(p) for _ in range(20000)]
        for i in range(3):
            self.assertAlmostEqual(draws.count(i) / 20000, p[i], delta=0.01)
        self.assertEqual(core.sample([0.0, 0.0, 1.0]), 2)


class StationaryProblemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.opt = solve_distribution_lp(Q6, [EXC6, CLD6], [0.05, 0.30])

    def test_benchmark_mixture(self):
        self.assertAlmostEqual(self.opt.value, 0.7193, places=3)

    def test_budgets_are_respected_in_expectation_with_the_shrunken_budgets(self):
        for seed in (1, 2):
            hist = run_stationary(6000, gamma=1.0, eps=0.4, seed=seed)
            q, e, c = window_means(hist[3000:])
            self.assertLessEqual(e, 0.05, (seed, e))
            self.assertLessEqual(c, 0.30, (seed, c))
            self.assertGreaterEqual(q, 0.88 * self.opt.value, (seed, q))

    def test_the_overshoot_of_the_optimistic_lp_shrinks_with_the_horizon(self):
        short = [window_means(run_stationary(500, 1.0, 0.0, s))[1] for s in (1, 2)]
        long = [window_means(run_stationary(4000, 1.0, 0.0, s)[2000:])[1] for s in (1, 2)]
        self.assertGreater(sum(short) / 2, 0.05 * 1.5)
        self.assertLess(sum(long) / 2, sum(short) / 2 - 0.02)
        self.assertLess(sum(long) / 2, 0.05 * 1.5)

    def test_every_arm_is_explored_before_the_estimates_settle(self):
        hist = run_stationary(600, 1.0, 0.0, 5)
        self.assertEqual(set(hist), set(range(6)))


class GridTests(unittest.TestCase):
    def test_grid_equals_the_i1_candidate_grid_on_the_synthetic_table(self):
        table = basic_table()
        arms = admissible_grid(table, synth_params())
        sel = I1Selector(table, I1Params(latency_targets_s={"f1": 8.0, "f2": 30.0}, profiles=SYNTH_PROFILES))
        self.assertEqual([(a.profile, a.batch_cap, a.kappa) for a in arms], [c.key() for c in sel.candidates()])
        self.assertEqual(len(arms), 10)

    @unittest.skipUnless(REAL_TABLE.exists(), "profiles/edge_profile_table.json not found")
    def test_the_edge_set_has_18_arms_in_the_i1_order(self):
        table = ProfileTable.load(REAL_TABLE)
        arms = admissible_grid(table, CBanditParams())
        sel = I1Selector(table, I1Params(latency_targets_s={"f1": 8.0, "f2": 30.0}))
        self.assertEqual(len(arms), 18)
        self.assertEqual([(a.profile, a.batch_cap, a.kappa) for a in arms], [c.key() for c in sel.candidates()])

    def test_parameter_checks(self):
        for bad in ({"constraints": {}}, {"constraints": {"Y_D": 0.0}}, {"epsilon": 1.0}, {"gamma": 0.0},
                    {"reward": "Y_D"}):
            with self.assertRaises(ValueError):
                CBanditParams(**bad)
        self.assertEqual(CBanditParams().constraints, DEFAULT_CONSTRAINTS)


def arm_env(base_arm):
    prof, b, k = base_arm.split("|")
    size = {"big-fp16": 0.9, "mid-int8": 0.7, "tiny-fp16": 0.3}[prof]
    cloud = 1.0 if k == "k1" else 0.0
    return {"Y_Q": size, "Y_L_f1": 0.01 + 0.04 * (size > 0.8), "Y_L_f2": 0.02 + 0.2 * (size > 0.8) * (1 - cloud),
            "Y_D": 0.02, "Y_C": 0.4 * cloud}


def tel(outcomes=None, node="E1"):
    d = {"node": node}
    if outcomes is not None:
        d["outcomes"] = outcomes
    return d


class IncumbentTests(unittest.TestCase):
    def make(self, **kw):
        return CBanditIncumbent(basic_table(), synth_params(**kw))

    def drive(self, inc, epochs, node="E1", env=arm_env):
        out, prev = [], None
        for e in range(epochs):
            cfg = inc.decide(Obs(epoch=e, block_id=1, telemetry=tel(prev, node)))
            out.append(cfg)
            prev = env(cfg.base_arm)
        return out

    def test_decide_returns_the_sampled_configuration(self):
        inc = self.make()
        cfg = inc.decide(Obs(epoch=0, block_id=3, telemetry=tel()))
        d = cfg.to_dict()
        self.assertEqual((d["arm_id"], d["router"], d["block_id"]), ("CB", "kappa", 3))
        self.assertIn(d["profile"], SYNTH_PROFILES)
        self.assertEqual(d["escalate_family2"], d["base_arm"].endswith("k1"))
        json.dumps(d)

    def test_first_decision_credits_nothing_and_the_next_credits_the_pulled_arm(self):
        inc = self.make()
        c0 = inc.decide(Obs(epoch=0, telemetry=tel()))
        self.assertEqual(inc.core.plays, [0] * 10)
        outcomes = arm_env(c0.base_arm)
        inc.decide(Obs(epoch=1, telemetry=tel(outcomes)))
        idx = [a.arm for a in inc.arms].index(c0.base_arm)
        self.assertEqual(inc.core.plays[idx], 1)
        self.assertAlmostEqual(inc.core.sum["Y_Q"][idx], outcomes["Y_Q"])
        self.assertAlmostEqual(inc.core.sum["Y_C"][idx], outcomes["Y_C"])
        self.assertEqual(inc.stats["updates"], 1)

    def test_missing_feedback_is_counted_and_nothing_is_credited(self):
        inc = self.make()
        inc.decide(Obs(epoch=0, telemetry=tel()))
        inc.decide(Obs(epoch=1, telemetry=tel()))
        self.assertEqual(inc.stats["no_feedback"], 1)
        self.assertEqual(sum(inc.core.plays), 0)
        inc.decide(Obs(epoch=2, telemetry=tel({"Y_Q": None, "Y_C": None})))
        self.assertEqual(inc.stats["skipped_updates"], 1)

    def test_reset_forgets_the_last_arm_but_keeps_the_estimates(self):
        inc = self.make()
        self.drive(inc, 40)
        before = list(inc.core.plays)
        inc.reset(block_id=2)
        inc.decide(Obs(epoch=0, telemetry=tel(arm_env("big-fp16|b8|k0"))))
        self.assertEqual(inc.core.plays, before)
        inc2 = self.make(reset_each_block=True)
        self.drive(inc2, 40)
        inc2.reset(block_id=2)
        self.assertEqual(sum(inc2.core.plays), 0)

    def test_switching_every_epoch_is_visible_and_unguarded(self):
        inc = self.make()
        cfgs = self.drive(inc, 400)
        changes = sum(1 for a, b in zip(cfgs, cfgs[1:]) if a.base_arm != b.base_arm)
        profile_changes = sum(1 for a, b in zip(cfgs, cfgs[1:]) if a.profile != b.profile)
        self.assertGreater(changes, 50)
        self.assertGreater(profile_changes, 20)
        self.assertEqual(inc.stats["switches"], changes)
        self.assertEqual(inc.stats["epochs"], 400)
        self.assertLess(changes, 399 + 1)

    def test_decisions_are_deterministic_given_seed_and_node(self):
        a = [c.base_arm for c in self.drive(self.make(seed=5), 80)]
        b = [c.base_arm for c in self.drive(self.make(seed=5), 80)]
        c = [c.base_arm for c in self.drive(self.make(seed=5), 80, node="E2")]
        d = [c.base_arm for c in self.drive(self.make(seed=6), 80)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_the_policy_moves_to_the_feasible_high_quality_arms(self):
        inc = self.make(gamma=1.0)
        cfgs = self.drive(inc, 1500)
        late = [c.base_arm for c in cfgs[-300:]]
        share_tiny = sum(1 for a in late if a.startswith("tiny")) / 300.0
        share_big_edge = sum(1 for a in late if a.startswith("big") and a.endswith("k0")) / 300.0
        self.assertLess(share_tiny, 0.15)
        self.assertLess(share_big_edge, 0.35)
        pulls = inc.pulls()
        self.assertEqual(sum(pulls.values()), 1499)

    def test_diagnostics_are_json(self):
        inc = self.make()
        self.drive(inc, 5)
        d = inc.diagnostics()
        json.dumps(d)
        self.assertEqual(d["regime"], "CB")
        self.assertIn(d["lp_status"], ("optimal", "fallback"))
        self.assertAlmostEqual(sum(d["p"].values()), 1.0, places=4)


def rec(family=1, t0=0, dt=1.0, ok=True, served_by="edge", deadline_ms=5000, item="x", reply="r", cls="B"):
    return {"family": family, "t_arrive": t0, "t_done": t0 + int(dt * 1e9), "status": 200 if ok else 502,
            "error": None if ok else "boom", "served_by": served_by, "deadline_ms": deadline_ms, "item": item,
            "reply_text": reply, "finish_reason": "stop", "class": cls}


class OutcomeTests(unittest.TestCase):
    TARGETS = {"f1": 2.0, "f2": 10.0}

    def test_outcomes_by_hand(self):
        recs = [rec(1, dt=1.0), rec(1, dt=3.0), rec(2, dt=4.0, served_by="cloud"), rec(2, dt=12.0, deadline_ms=20000),
                rec(2, ok=False, deadline_ms=20000)]
        out = epoch_outcomes(recs, self.TARGETS, quality_fn=lambda r: 1.0 if r["family"] == 1 else 0.5)
        self.assertEqual((out["n"], out["n_f1"], out["n_f2"]), (5, 2, 3))
        self.assertAlmostEqual(out["Y_L_f1"], 0.5)
        self.assertAlmostEqual(out["Y_L_f2"], 2.0 / 3.0)
        self.assertAlmostEqual(out["Y_D"], 1.0 / 5.0)
        self.assertAlmostEqual(out["Y_C"], 1.0 / 5.0)
        self.assertAlmostEqual(out["Y_Q"], (1.0 + 1.0 + 0.5 + 0.5 + 0.0) / 5.0)

    def test_a_component_without_requests_is_none(self):
        out = epoch_outcomes([rec(1, dt=0.5)], self.TARGETS)
        self.assertIsNone(out["Y_L_f2"])
        self.assertIsNone(out["Y_Q"])
        self.assertEqual(out["Y_L_f1"], 0.0)
        empty = epoch_outcomes([], self.TARGETS)
        self.assertIsNone(empty["Y_D"])
        self.assertIsNone(empty["Y_C"])
        self.assertEqual(empty["n"], 0)

    def test_target_offset_removes_the_share_spent_outside_the_node(self):
        out = epoch_outcomes([rec(1, dt=1.5)], self.TARGETS, target_offset_s={"f1": 0.6})
        self.assertEqual(out["Y_L_f1"], 1.0)

    def test_energy_rows_are_added_when_a_function_is_given(self):
        out = epoch_outcomes([rec(1, cls="A"), rec(1, cls="A"), rec(1, cls="B")], self.TARGETS,
                             energy_fn=lambda r: 0.4 if r["class"] == "A" else 3.0)
        self.assertAlmostEqual(out["Y_E_A"], 0.4)
        self.assertAlmostEqual(out["Y_E_B"], 1.0)

    def test_feeding_epoch_outcomes_into_the_bandit(self):
        inc = CBanditIncumbent(basic_table(), synth_params())
        inc.decide(Obs(epoch=0, telemetry=tel()))
        out = epoch_outcomes([rec(1, dt=0.5), rec(2, dt=1.0)], self.TARGETS, quality_fn=lambda r: 0.8)
        inc.decide(Obs(epoch=1, telemetry=tel(out)))
        self.assertEqual(sum(inc.core.plays), 1)


class SourceTests(unittest.TestCase):
    TARGETS = {"f1": 2.0, "f2": 10.0}

    def test_reads_only_new_complete_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "edge-gw-1.jsonl")
            src = GatewayOutcomeSource(os.path.join(d, "edge-gw-{n}.jsonl"), self.TARGETS, lambda r: 1.0)
            self.assertEqual(src.read_records(path), [])
            with open(path, "w") as fh:
                fh.write(json.dumps(rec(1)) + "\n")
                fh.write(json.dumps(rec(2, dt=11.0)) + "\n")
                fh.write(json.dumps(rec(1))[:20])
            out = src(Obs(telemetry={"node": "E1", "node_index": 0}))
            self.assertEqual(out["n"], 2)
            self.assertEqual(out["Y_L_f2"], 1.0)
            with open(path, "a") as fh:
                fh.write(json.dumps(rec(1))[20:] + "\n")
                fh.write(json.dumps({"event": "no family here"}) + "\n")
            out = src(Obs(telemetry={"node": "E1", "node_index": 0}))
            self.assertEqual(out["n"], 1)
            self.assertEqual(src(Obs(telemetry={"node": "E1"}))["n"], 0)
            self.assertIsNone(src(Obs(telemetry={})))

    def test_path_uses_the_gateway_node_number(self):
        src = GatewayOutcomeSource("runs/r1/edge-gw-{n}.jsonl", self.TARGETS)
        self.assertEqual(src.path_for("E3"), "runs/r1/edge-gw-3.jsonl")
        self.assertEqual(src.path_for("E3", 4), "runs/r1/edge-gw-5.jsonl")
        self.assertEqual(GatewayOutcomeSource("x/{node}.jsonl", self.TARGETS).path_for("E3"), "x/E3.jsonl")

    def test_a_truncated_log_is_read_from_the_start(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gw.jsonl")
            src = GatewayOutcomeSource(path, self.TARGETS)
            with open(path, "w") as fh:
                fh.write("".join(json.dumps(rec(1)) + "\n" for _ in range(5)))
            self.assertEqual(len(src.read_records(path)), 5)
            with open(path, "w") as fh:
                fh.write(json.dumps(rec(1)) + "\n")
            self.assertEqual(len(src.read_records(path)), 1)

    def test_bandit_reads_its_source_when_the_harness_supplies_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gw-E1.jsonl")
            src = GatewayOutcomeSource(os.path.join(d, "gw-{node}.jsonl"), self.TARGETS, lambda r: 0.9)
            inc = CBanditIncumbent(basic_table(), synth_params(), src)
            inc.decide(Obs(epoch=0, telemetry=tel()))
            with open(path, "w") as fh:
                fh.write("".join(json.dumps(rec(1, dt=0.4)) + "\n" for _ in range(3)))
            inc.decide(Obs(epoch=1, telemetry=tel()))
            self.assertEqual(sum(inc.core.plays), 1)
            idx = next(i for i, n in enumerate(inc.core.plays) if n)
            self.assertAlmostEqual(inc.core.sum["Y_Q"][idx], 0.9)
            self.assertAlmostEqual(inc.core.sum["Y_L_f1"][idx], 0.0)


class FactoryTests(unittest.TestCase):
    def test_cb_through_the_controller_adapter(self):
        a = make_incumbent("CB", params={"epsilon": 0.1}, seed=3)
        a.begin_block({"node": "E1", "block_id": 4, "epoch": 0})
        d = a.decide({"epoch": 0, "block_id": 4, "node": "E1", "arrival_rate": 0.3, "memory_budget_mib": 11264})
        self.assertEqual(d["router"], "kappa")
        self.assertTrue(d["profile"].startswith("q35-"))
        self.assertIn(d["batch_cap"], (8, 16))
        for k in ("arm_id", "block_id", "base_arm"):
            self.assertNotIn(k, d)
        self.assertEqual(len(a.inner.arms), 18)
        self.assertEqual(a.inner.p.epsilon, 0.1)
        a.decide({"epoch": 1, "block_id": 4, "node": "E1",
                  "outcomes": {"Y_Q": 0.5, "Y_C": 0.0, "Y_D": 0.0, "Y_L_f1": 0.0, "Y_L_f2": None}})
        self.assertEqual(a.inner.stats["updates"], 1)
        json.dumps(a.diagnostics())

    def test_unknown_keyword_and_missing_targets(self):
        with self.assertRaises(ValueError):
            make_incumbent("CB", beta=0.3)
        with self.assertRaises(ValueError):
            make_incumbent("CB", outcome_log="/nonexistent/{node}.jsonl")

    def test_outcome_log_with_targets_builds_a_source(self):
        a = make_incumbent("CB", outcome_log="/nonexistent/edge-gw-{n}.jsonl", latency_targets_s={"f1": 8.0, "f2": 30.0},
                           quality_fn="tests.baselines.test_cbandit:_half")
        self.assertIsNotNone(a.inner.source)
        self.assertEqual(a.inner.source.quality_fn({}), 0.5)


def _half(record):
    return 0.5


if __name__ == "__main__":
    unittest.main()
