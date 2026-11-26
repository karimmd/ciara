import os
import unittest

from ciara_control.clock import FakeClock
from ciara_control.config import parse_config
from ciara_control.controller import EXIT_EXISTS, EXIT_OK, EXIT_PREFLIGHT, Controller
from ciara_control.design import make_design_law
from ciara_control.incumbents import Obs, merge_incumbent_config
from ciara_control.runstate import load_run_state
from ciara_control.util import read_jsonl
from helpers import Fleet, TmpDir, base_config, deep, make_controller


def run_all(cfg_dict, **kw):
    clock = FakeClock()
    fleet = Fleet(clock)
    ctl = make_controller(cfg_dict, clock, fleet, **kw)
    code = ctl.run()
    return ctl, fleet, code


def logs(run_dir):
    blocks = read_jsonl(os.path.join(run_dir, "blocks.jsonl"))
    return ([b for b in blocks if b["rec"] == "block"], [b for b in blocks if b["rec"] == "block_end"],
            read_jsonl(os.path.join(run_dir, "epochs.jsonl")), read_jsonl(os.path.join(run_dir, "events.jsonl")))


class FullRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TmpDir()
        cls.dir = cls.tmp.__enter__()
        cls.cfg_dict = base_config(cls.dir)
        cls.ctl, cls.fleet, cls.code = run_all(cls.cfg_dict)
        cls.blocks, cls.ends, cls.epochs, cls.events = logs(cls.dir)
        cls.cfg = parse_config(cls.cfg_dict)
        cls.law = make_design_law(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.__exit__()

    def test_finishes_done(self):
        self.assertEqual(self.code, EXIT_OK)
        with open(os.path.join(self.dir, "STATUS")) as fh:
            self.assertEqual(fh.read().strip(), "DONE")

    def test_one_block_record_per_node_and_block(self):
        keys = [(b["node"], b["block_id"]) for b in self.blocks]
        self.assertEqual(len(keys), 3 * 8)
        self.assertEqual(len(set(keys)), 3 * 8)

    def test_one_epoch_record_per_node_and_epoch(self):
        keys = [(e["node"], e["epoch"]) for e in self.epochs]
        self.assertEqual(len(keys), 3 * 32)
        self.assertEqual(len(set(keys)), 3 * 32)

    def test_block_record_fields_follow_spec(self):
        b = self.blocks[0]
        for key in ("node", "block_id", "epoch_start", "epoch_end", "arm_id", "config", "propensity", "context_id",
                    "design", "seed", "rng", "switch_time_s"):
            self.assertIn(key, b)
        self.assertEqual((b["epoch_start"], b["epoch_end"]), (b["block_id"] * 4, b["block_id"] * 4 + 4))
        self.assertEqual(b["design"], "screening")
        self.assertEqual(b["seed"], 20261016)
        self.assertEqual(set(b["rng"]), {"entropy", "draw", "arm_index"})

    def test_draws_equal_the_seeded_law_and_record_the_propensity(self):
        for b in self.blocks:
            d = self.law.draw(b["node_index"], b["block_id"], b["context_id"])
            self.assertEqual(b["arm_id"], d.arm_id)
            self.assertEqual(b["propensity"], d.propensity)
            self.assertEqual(b["rng"]["draw"], d.u)
            self.assertEqual(b["rng"]["entropy"], list(d.entropy))
            self.assertAlmostEqual(b["propensity"], 0.25)

    def test_nodes_draw_independently(self):
        by_block = {}
        for b in self.blocks:
            by_block.setdefault(b["block_id"], set()).add(b["rng"]["draw"])
        self.assertTrue(all(len(v) == 3 for v in by_block.values()))

    def test_burn_in_flags(self):
        for e in self.epochs:
            self.assertEqual(e["burn_in"], e["epoch_in_block"] < 2)
        b = self.blocks[0]
        self.assertEqual((b["measure_start_epoch"], b["measure_end_epoch"]),
                         (b["epoch_start"] + 2, b["epoch_end"]))

    def test_every_block_closed_complete(self):
        self.assertEqual(len(self.ends), 3 * 8)
        self.assertTrue(all(x["status"] == "complete" and x["epochs_logged"] == 4 for x in self.ends))

    def test_activate_only_on_profile_change_then_config(self):
        for nid, io in self.fleet.ios.items():
            prev_profile = None
            calls = [c for c in io.calls if c[0] != "prepare"]
            for i, (kind, arg) in enumerate(calls):
                if kind == "activate":
                    self.assertNotEqual(arg, prev_profile, "activate without a profile change")
                    prev_profile = arg
                    nxt = calls[i + 1]
                    self.assertEqual(nxt[0], "config")
                    self.assertEqual(nxt[1]["profile"], arg)
                else:
                    if arg["profile"] != prev_profile:
                        self.fail(f"{nid}: config with profile {arg['profile']} without activate")

    def test_static_arms_push_once_per_block(self):
        for nid, io in self.fleet.ios.items():
            by_block = {}
            for kind, arg in io.calls:
                if kind == "config" and not arg["arm_id"].startswith("I1"):
                    by_block[arg["block_id"]] = by_block.get(arg["block_id"], 0) + 1
            self.assertTrue(all(v == 1 for v in by_block.values()))

    def test_incumbent_decides_every_epoch_of_its_blocks(self):
        inc_blocks = [b for b in self.blocks if b["kind"] == "incumbent"]
        self.assertTrue(inc_blocks, "seed should assign the incumbent at least once")
        for b in inc_blocks:
            eps = [e for e in self.epochs if e["node"] == b["node"] and e["block_id"] == b["block_id"]]
            self.assertEqual(len(eps), 4)
            self.assertTrue(all(e["obs"] is not None and e["decide_ms"] is not None for e in eps))
        for node in self.ctl.nodes:
            inst = node.incumbents.get("I1-E")
            if inst is not None:
                n = sum(1 for b in inc_blocks if b["node"] == node.id)
                self.assertEqual(inst.calls, 4 * n)
                self.assertEqual(len(inst.blocks), n)

    def test_incumbent_obs_content(self):
        e = next(x for x in self.epochs if x["kind"] == "incumbent" and x["epoch_in_block"] == 2)
        obs = e["obs"]
        for key in ("arrival_rate", "memory_budget_mib", "gateway", "engine", "epoch", "node"):
            self.assertIn(key, obs)
        self.assertEqual(obs["memory_budget_mib"], 11264)
        for key in ("queue_len", "in_flight", "arrivals", "completed", "recent"):
            self.assertIn(key, obs["gateway"])
        self.assertIsNotNone(obs["arrival_rate"])
        self.assertIn("running", obs["engine"])

    def test_static_arm_epochs_have_no_obs_and_constant_config(self):
        for b in (x for x in self.blocks if x["kind"] == "static"):
            eps = [e for e in self.epochs if e["node"] == b["node"] and e["block_id"] == b["block_id"]]
            self.assertTrue(all(e["obs"] is None and e["config"] == b["config"] for e in eps))

    def test_switch_time_recorded_when_profile_changed(self):
        sw = [e for e in self.epochs if e["switched"]]
        self.assertTrue(sw)
        for e in sw:
            self.assertEqual(e["switch_time_s"], 0.5)
        blk = [b for b in self.blocks if b["profile_switched"]]
        self.assertTrue(all(b["switch_time_s"] == 0.5 for b in blk))

    def test_arrival_rate_and_engine_delta_in_epoch_telemetry(self):
        e = next(x for x in self.epochs if x["epoch"] == 5 and x["node"] == "E1")
        t = e["telemetry"]
        self.assertAlmostEqual(t["dt_s"], 30.0)
        self.assertAlmostEqual(t["arrival_rate"], t["arrivals"] / 30.0)
        self.assertIsNotNone(t["engine"]["requests_finished_delta"])

    def test_engine_counters_rebaselined_after_a_switch(self):
        for e in self.epochs:
            if e["switched"]:
                eng = e["telemetry"]["engine"]
                self.assertFalse(eng["counter_reset"])

    def test_cloud_notified_at_each_block_start(self):
        starts = {(nid, v["block_id"]) for n in self.fleet.cloud.notices for nid, v in n.items()}
        self.assertEqual(len(starts), 3 * 8)

    def test_manifest_and_plan(self):
        st = load_run_state(self.dir)
        self.assertEqual(st.manifest["design_hash"], self.cfg.design_hash)
        self.assertEqual(st.manifest["total_epochs"], 32)
        self.assertEqual(st.last_epoch, 31)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "PROGRESS")))

    def test_epoch_times_sit_on_the_grid(self):
        m = load_run_state(self.dir).manifest
        for e in self.epochs:
            self.assertEqual(e["t_start_mono_ns"], m["t0_mono_ns"] + e["epoch"] * 30_000_000_000)
            self.assertEqual(e["t_end_mono_ns"], m["t0_mono_ns"] + (e["epoch"] + 1) * 30_000_000_000)


class Designs(unittest.TestCase):
    def test_certification_draws_only_the_set_with_formula_propensities(self):
        with TmpDir() as d:
            cfg = base_config(d, n_blocks=40)
            cfg["design"] = {"label": "certification",
                             "arms": ["I1-E", "q35-4b-ih-w8a8|b16|k1", "q35-2b-fp16|b8|k1"],
                             "incumbent": "I1-E", "epsilon": 0.5}
            ctl, fleet, code = run_all(cfg)
            blocks, _, epochs, _ = logs(d)
            self.assertEqual(code, EXIT_OK)
            allowed = {"I1-E": 0.5 + 0.5 / 3, "q35-4b-ih-w8a8|b16|k1": 1 / 6, "q35-2b-fp16|b8|k1": 1 / 6}
            self.assertEqual({b["arm_id"] for b in blocks} - set(allowed), set())
            for b in blocks:
                self.assertAlmostEqual(b["propensity"], allowed[b["arm_id"]])
                self.assertEqual(b["design"], "certification")
                self.assertEqual(b["epsilon"], 0.5)
            share = sum(b["arm_id"] == "I1-E" for b in blocks) / len(blocks)
            self.assertGreater(share, 0.5)
            self.assertTrue(all(e["design"] == "certification" for e in epochs))

    def test_deployment_policy_with_context_schedule(self):
        with TmpDir() as d:
            cfg = base_config(d, n_blocks=6)
            cfg["design"] = {"label": "deployment", "policy": {
                "light": {"q35-2b-fp16|b8|k1": 1.0}, "heavy": {"q35-4b-fp16|b8|k0": 1.0}}}
            ctx = os.path.join(d, "ctx.json")
            import json
            with open(ctx, "w") as fh:
                json.dump({"default_context": "light", "segments": [
                    {"start_epoch": 12, "end_epoch": 24, "context_id": "heavy"}]}, fh)
            cfg["context"] = {"ref": ctx, "default": "light"}
            ctl, fleet, code = run_all(cfg)
            blocks, _, _, _ = logs(d)
            for b in blocks:
                want = "q35-4b-fp16|b8|k0" if 12 <= b["epoch_start"] < 24 else "q35-2b-fp16|b8|k1"
                self.assertEqual(b["arm_id"], want)
                self.assertEqual(b["context_id"], "heavy" if 12 <= b["epoch_start"] < 24 else "light")
                self.assertEqual(b["propensity"], 1.0)


class IncumbentBehaviour(unittest.TestCase):
    def one_node_cfg(self, d, script, **over):
        cfg = base_config(d, n_blocks=2, **over)
        cfg["nodes"] = cfg["nodes"][:1]
        cfg["arms"] = [{"id": "INC", "kind": "incumbent", "router": "I2", "router_params": {"theta": 0.3},
                        "incumbent": {"factory": "scripted", "kwargs": {"script": script}}}]
        cfg["design"] = {"label": "screening"}
        return cfg

    def test_profile_switches_mid_block_activate_then_config(self):
        with TmpDir() as d:
            script = [{"profile": "q35-4b-fp16", "batch_cap": 8}, {"profile": "q35-4b-fp16", "batch_cap": 16},
                      {"profile": "q35-2b-fp16", "batch_cap": 16}, {"profile": "q35-2b-fp16", "batch_cap": 8}]
            ctl, fleet, code = run_all(self.one_node_cfg(d, script))
            calls = [c for c in fleet.ios["E1"].calls if c[0] != "prepare"]
            kinds = [c[0] for c in calls]
            self.assertEqual(kinds[:6], ["activate", "config", "config", "activate", "config", "config"])
            self.assertEqual([c[1] for c in calls if c[0] == "activate"][:2], ["q35-4b-fp16", "q35-2b-fp16"])
            cfgs = [c[1] for c in calls if c[0] == "config"]
            self.assertTrue(all(c["router"] == "I2" and c["router_params"] == {"theta": 0.3} for c in cfgs))
            _, _, epochs, _ = logs(d)
            e2 = [e for e in epochs if e["block_id"] == 0][2]
            self.assertTrue(e2["switched"])
            self.assertEqual(e2["switch_time_s"], 0.5)

    def test_unchanged_decision_is_not_pushed_again(self):
        with TmpDir() as d:
            script = [{"profile": "q35-4b-fp16", "batch_cap": 8}]
            ctl, fleet, code = run_all(self.one_node_cfg(d, script))
            configs = [c for c in fleet.ios["E1"].calls if c[0] == "config"]
            self.assertEqual(len(configs), 2)

    def test_merge_missing_keys_keep_previous_values(self):
        cfg = parse_config(self.one_node_cfg("/x", [{"profile": "p", "batch_cap": 1}]))
        arm = cfg.arms["INC"]
        prev = {"arm_id": "INC", "profile": "q35-2b-fp16", "batch_cap": 8, "escalate_family2": False,
                "router": "I2", "block_id": 3}
        out = merge_incumbent_config({"batch_cap": 16}, arm, prev, 4)
        self.assertEqual((out["profile"], out["batch_cap"], out["escalate_family2"], out["block_id"]),
                         ("q35-2b-fp16", 16, False, 4))
        class Cfg:
            def __init__(self):
                self.profile, self.batch_cap = "x", 4
        out = merge_incumbent_config(Cfg(), arm, None, 1)
        self.assertEqual(out["profile"], "x")
        with self.assertRaises(ValueError):
            merge_incumbent_config({}, arm, None, 1)

    def test_cascade_incumbent_arm_keeps_its_router_over_the_decision(self):
        raw = self.one_node_cfg("/x", [{"profile": "p", "batch_cap": 1}])
        raw["arms"][0]["router"] = "kappaC"
        raw["arms"][0]["router_params"] = {"budget": 32}
        raw["arms"][0]["admission"] = "f1prio"
        arm = parse_config(raw).arms["INC"]
        out = merge_incumbent_config({"profile": "q35-2b-fp16", "batch_cap": 16, "escalate_family2": True, "router": "kappa"}, arm, None, 5)
        self.assertEqual((out["router"], out["router_params"], out["admission"]), ("kappaC", {"budget": 32}, "f1prio"))
        self.assertEqual((out["profile"], out["batch_cap"], out["escalate_family2"], out["block_id"]), ("q35-2b-fp16", 16, True, 5))
        prev = dict(out, escalate_family2=False, batch_cap=8)
        out2 = merge_incumbent_config({"batch_cap": 4, "router": "kappa"}, arm, prev, 6)
        self.assertEqual((out2["router"], out2["batch_cap"], out2["escalate_family2"]), ("kappaC", 4, False))
        plain = parse_config(self.one_node_cfg("/x", [{"profile": "p", "batch_cap": 1}])).arms["INC"]
        self.assertEqual(merge_incumbent_config({"profile": "p", "batch_cap": 2, "router": "I2"}, plain, None, 1)["router"], "I2")

    def test_obs_supports_attribute_access(self):
        o = Obs(a=1)
        self.assertEqual((o.a, o["a"]), (1, 1))
        with self.assertRaises(AttributeError):
            o.b

    def test_incumbent_error_is_logged_and_run_continues(self):
        with TmpDir() as d:
            cfg = self.one_node_cfg(d, [])
            cfg["arms"][0]["incumbent"] = {"factory": "broken"}
            ctl, fleet, code = run_all(cfg)
            self.assertEqual(code, EXIT_OK)
            _, _, epochs, events = logs(d)
            self.assertTrue(any("decide" in " ".join(e["errors"]) for e in epochs))
            self.assertTrue(any(ev["event"] == "incumbent_error" for ev in events))
            self.assertTrue(all(e["config_ok"] is False for e in epochs))

    def test_incumbent_error_abort_policy(self):
        with TmpDir() as d:
            cfg = self.one_node_cfg(d, [])
            cfg["arms"][0]["incumbent"] = {"factory": "broken"}
            cfg["control"]["on_incumbent_error"] = "abort"
            ctl, fleet, code = run_all(cfg)
            self.assertNotEqual(code, EXIT_OK)
            with open(os.path.join(d, "STATUS")) as fh:
                self.assertEqual(fh.read().strip(), "FAILED")

    def test_real_incumbent_package_loads_through_factory_path(self):
        with TmpDir() as d:
            cfg = self.one_node_cfg(d, [])
            cfg["arms"][0]["incumbent"] = {"factory": "ciara_control.incumbents:StubIncumbent",
                                           "kwargs": {"profile": "q35-2b-fp16"}}
            ctl, fleet, code = run_all(cfg)
            self.assertEqual(code, EXIT_OK)
            _, _, epochs, _ = logs(d)
            self.assertTrue(all(e["config"]["profile"] == "q35-2b-fp16" for e in epochs))


class Robustness(unittest.TestCase):
    def test_activate_failure_is_retried_next_epoch_and_flagged(self):
        with TmpDir() as d:
            cfg = base_config(d, n_blocks=2)
            cfg["nodes"] = cfg["nodes"][:1]
            cfg["arms"] = cfg["arms"][:1]
            clock = FakeClock()
            fleet = Fleet(clock)
            ctl = make_controller(cfg, clock, fleet)
            fleet.factory(ctl.cfg.nodes[0], ctl.cfg).fail["activate"] = 1
            ctl.run()
            _, ends, epochs, _ = logs(d)
            first = [e for e in epochs if e["block_id"] == 0]
            self.assertFalse(first[0]["config_ok"])
            self.assertTrue(any("activate" in x for x in first[0]["errors"]))
            self.assertTrue(first[1]["config_ok"])
            self.assertTrue(first[1]["switched"])
            self.assertEqual(ends[0]["actuation_failures"], 1)
            self.assertEqual(ends[0]["status"], "complete")

    def test_gateway_down_during_poll_does_not_stop_the_run(self):
        with TmpDir() as d:
            cfg = base_config(d, n_blocks=2)
            cfg["nodes"] = cfg["nodes"][:1]
            cfg["arms"] = cfg["arms"][:1]
            clock = FakeClock()
            fleet = Fleet(clock)
            ctl = make_controller(cfg, clock, fleet)
            io = fleet.factory(ctl.cfg.nodes[0], ctl.cfg)
            orig = io.gateway_state
            calls = {"n": 0}

            def flaky():
                calls["n"] += 1
                if calls["n"] in (4, 5):
                    raise OSError("gateway down")
                return orig()
            io.gateway_state = flaky
            self.assertEqual(ctl.run(), EXIT_OK)
            _, _, epochs, _ = logs(d)
            bad = [e for e in epochs if not e["telemetry"]["gateway_ok"]]
            self.assertTrue(bad)
            self.assertTrue(all(any("gateway_state" in x for x in e["errors"]) for e in bad))
            self.assertEqual(len(epochs), 8)

    def test_gateway_config_mismatch_triggers_repush(self):
        with TmpDir() as d:
            cfg = base_config(d, n_blocks=2)
            cfg["nodes"] = cfg["nodes"][:1]
            cfg["arms"] = cfg["arms"][:1]
            clock = FakeClock()
            fleet = Fleet(clock)
            ctl = make_controller(cfg, clock, fleet)
            io = fleet.factory(ctl.cfg.nodes[0], ctl.cfg)
            clk_state = {"hit": False}
            orig = io.gateway_state

            def drift():
                st = orig()
                if (ctl.t0_mono_ns and io.sim.cfg and not clk_state["hit"]
                        and clock.mono_ns() >= ctl.t0_mono_ns + 2 * 30_000_000_000):
                    clk_state["hit"] = True
                    io.sim.cfg = dict(io.sim.cfg, batch_cap=99)
                    st = orig()
                return st
            io.gateway_state = drift
            ctl.run()
            _, _, epochs, _ = logs(d)
            self.assertTrue(any(e["repushed"] for e in epochs))
            self.assertEqual(io.sim.cfg["batch_cap"], 8)

    def test_preflight_failure_stops_before_any_epoch(self):
        with TmpDir() as d:
            cfg = base_config(d)
            clock = FakeClock()
            fleet = Fleet(clock)
            ctl = make_controller(cfg, clock, fleet)
            fleet.factory(ctl.cfg.nodes[1], ctl.cfg).fail["prepare"] = 1
            self.assertEqual(ctl.run(), EXIT_PREFLIGHT)
            self.assertFalse(os.path.exists(os.path.join(d, "manifest.json")))

    def test_refuses_to_overwrite_an_existing_run(self):
        with TmpDir() as d:
            run_all(base_config(d, n_blocks=2))
            ctl, fleet, code = run_all(base_config(d, n_blocks=2))
            self.assertEqual(code, EXIT_EXISTS)

    def test_prepare_called_with_all_design_profiles(self):
        with TmpDir() as d:
            ctl, fleet, code = run_all(base_config(d, n_blocks=2))
            prep = [c for c in fleet.ios["E1"].calls if c[0] == "prepare"][0][1]
            self.assertEqual(sorted(prep), sorted(["q35-4b-fp16", "q35-4b-ih-w8a8", "q35-2b-fp16"]))

    def test_stop_request_ends_run_without_finishing(self):
        with TmpDir() as d:
            clock = FakeClock()
            fleet = Fleet(clock)
            cfg = base_config(d, n_blocks=20)
            ctl = make_controller(cfg, clock, fleet, fault_hook=lambda e: ctl.request_stop() if e == 9 else None)
            code = ctl.run()
            self.assertEqual(code, 130)
            blocks, ends, epochs, events = logs(d)
            with open(os.path.join(d, "STATUS")) as fh:
                self.assertEqual(fh.read().strip(), "STOPPED")
            self.assertEqual(max(e["epoch"] for e in epochs), 7)
            self.assertEqual({x["status"] for x in ends if x["block_id"] == 2}, {"interrupted"})


class Overrun(unittest.TestCase):
    def test_boundary_work_that_overruns_an_epoch_skips_the_missed_boundary_and_flags_the_epoch(self):
        with TmpDir() as d:
            cfg = base_config(d, n_blocks=2)
            cfg["nodes"] = cfg["nodes"][:1]
            cfg["arms"] = cfg["arms"][:1]
            clock = FakeClock()
            fleet = Fleet(clock)
            ctl = make_controller(cfg, clock, fleet)
            io = fleet.factory(ctl.cfg.nodes[0], ctl.cfg)
            orig = io.sim.activate

            def slow_activate(profile, switch_s=None):
                clock.advance(75 * 1_000_000_000)
                return orig(profile, switch_s)
            io.sim.activate = slow_activate
            self.assertEqual(ctl.run(), EXIT_OK)
            _, _, epochs, events = logs(d)
            missed = [e for e in events if e["event"] == "missed_boundary"]
            self.assertEqual([m["epoch"] for m in missed], [1])
            first = [e for e in epochs if e["epoch"] == 0][0]
            self.assertTrue(first["overrun"])
            self.assertGreater(first["end_late_ms"], 40_000)
            self.assertEqual(sorted(e["epoch"] for e in epochs), [0] + list(range(2, 8)))
            self.assertFalse(any(e["overrun"] for e in epochs if e["epoch"] >= 2))


class NodeErrors(unittest.TestCase):
    def test_error_in_one_node_is_isolated_and_logged(self):
        with TmpDir() as d:
            cfg = base_config(d, n_blocks=4)
            clock = FakeClock()
            fleet = Fleet(clock)
            ctl = make_controller(cfg, clock, fleet)
            real = ctl._node_step

            def flaky(node, e, t_b):
                if node.id == "E2" and e in (3, 4):
                    raise RuntimeError("bug in node step")
                return real(node, e, t_b)
            ctl._node_step = flaky
            self.assertEqual(ctl.run(), EXIT_OK)
            _, _, epochs, events = logs(d)
            errs = [e for e in events if e["event"] == "node_error"]
            self.assertEqual([(x["node"], x["epoch"]) for x in errs], [("E2", 3), ("E2", 4)])
            self.assertEqual(errs[1]["consecutive"], 2)
            self.assertEqual(len([e for e in epochs if e["node"] == "E1"]), 16)
            self.assertEqual(len([e for e in epochs if e["node"] == "E3"]), 16)
            self.assertEqual(sorted({e["epoch"] for e in epochs if e["node"] == "E2"}), [0, 1] + list(range(8, 16)))
            ends = [x for x in logs(d)[1] if x["node"] == "E2"]
            self.assertEqual([(x["block_id"], x["status"], x["epochs_logged"]) for x in ends[:3]],
                             [(0, "partial", 2), (2, "complete", 4), (3, "complete", 4)])

    def test_repeated_errors_in_one_node_fail_the_run(self):
        with TmpDir() as d:
            clock = FakeClock()
            fleet = Fleet(clock)
            ctl = make_controller(base_config(d, n_blocks=8), clock, fleet)
            real = ctl._node_step

            def broken(node, e, t_b):
                if node.id == "E3" and e >= 2:
                    raise RuntimeError("persistent bug")
                return real(node, e, t_b)
            ctl._node_step = broken
            self.assertNotEqual(ctl.run(), EXIT_OK)
            with open(os.path.join(d, "STATUS")) as fh:
                self.assertEqual(fh.read().strip(), "FAILED")
            self.assertIn("E3", ctl.abort_reason)


class CommandLine(unittest.TestCase):
    def write_cfg(self, d, **over):
        import json
        cfg = base_config(os.path.join(d, "run"), **over)
        path = os.path.join(d, "cfg.json")
        with open(path, "w") as fh:
            json.dump(cfg, fh)
        return path

    def test_validate_plan_and_dry_run(self):
        from ciara_control.controller import main
        with TmpDir() as d:
            path = self.write_cfg(d)
            self.assertEqual(main(["validate", "--config", path]), 0)
            out = os.path.join(d, "plan.json")
            self.assertEqual(main(["plan", "--config", path, "--out", out]), 0)
            import json
            with open(out) as fh:
                plan = json.load(fh)
            self.assertEqual(len(plan["draws"]), 3 * 8)
            self.assertEqual(main(["run", "--config", path, "--dry-run"]), 0)
            blocks, _, _, _ = logs(os.path.join(d, "run"))
            self.assertEqual({(b["node"], b["block_id"]): b["arm_id"] for b in blocks},
                             {(x["node"], x["block_id"]): x["arm_id"] for x in plan["draws"]})

    def test_invalid_config_exit_code(self):
        from ciara_control.controller import EXIT_CONFIG, main
        with TmpDir() as d:
            path = self.write_cfg(d, burn_in_epochs=9)
            self.assertEqual(main(["validate", "--config", path]), EXIT_CONFIG)

    def test_sigterm_stops_the_run_cleanly_and_it_is_resumable(self):
        import signal
        import subprocess
        import sys
        import time
        with TmpDir() as d:
            path = self.write_cfg(d, epoch_s=1, block_epochs=3, burn_in_epochs=1, n_blocks=60,
                                  control={"start_delay_s": 1, "resume_margin_s": 1})
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "src", "controller"),
                                                 env.get("PYTHONPATH", "")])
            p = subprocess.Popen([sys.executable, "-m", "ciara_control.controller", "run", "--config", path,
                                  "--dry-run", "--real-time"], env=env, stdout=subprocess.DEVNULL)
            run_dir = os.path.join(d, "run")
            deadline = time.time() + 30
            while time.time() < deadline and not os.path.exists(os.path.join(run_dir, "epochs.jsonl")):
                time.sleep(0.2)
            time.sleep(5.5)
            p.send_signal(signal.SIGTERM)
            self.assertEqual(p.wait(timeout=15), 130)
            with open(os.path.join(run_dir, "STATUS")) as fh:
                self.assertEqual(fh.read().strip(), "STOPPED")
            blocks, ends, epochs, events = logs(run_dir)
            self.assertTrue(epochs)
            self.assertTrue(any(x["status"] == "interrupted" for x in ends))
            self.assertTrue(any(e["event"] == "stop" for e in events))


class NodeIndependence(unittest.TestCase):
    def test_a_slow_node_does_not_delay_the_records_of_the_other_nodes(self):
        import time
        from ciara_control.clock import RealClock
        with TmpDir() as d:
            cfg = base_config(d, n_blocks=2, epoch_s=0.4, block_epochs=3, burn_in_epochs=1)
            cfg["nodes"] = cfg["nodes"][:2]
            cfg["arms"] = cfg["arms"][:1]
            cfg["control"] = {"start_delay_s": 0.5}
            clock = RealClock()
            fleet = Fleet(FakeClock())
            ctl = make_controller(cfg, clock, fleet)
            slow_io = fleet.factory(ctl.cfg.nodes[1], ctl.cfg)
            orig = slow_io.gateway_state
            calls = {"n": 0}

            def slow_state():
                calls["n"] += 1
                if calls["n"] == 3:
                    time.sleep(1.1)
                return orig()
            slow_io.gateway_state = slow_state
            self.assertEqual(ctl.run(), EXIT_OK)
            _, _, epochs, events = logs(d)
            e1 = sorted(e["epoch"] for e in epochs if e["node"] == "E1")
            e2 = sorted(e["epoch"] for e in epochs if e["node"] == "E2")
            self.assertEqual(e1, list(range(6)))
            self.assertLess(len(e2), 6)
            missed = [m for m in events if m["event"] == "missed_boundary"]
            self.assertTrue(missed and all(m["node"] == "E2" for m in missed))


class BalancedRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_design import balanced_cfg
        cls.tmp = TmpDir()
        cls.dir = cls.tmp.__enter__()
        d = balanced_cfg(["I1-E", "A|fifo", "A|f1prio", "B|fifo"])
        d["run_dir"], d["n_blocks"], d["block_epochs"], d["burn_in_epochs"] = cls.dir, 10, 4, 2
        d["arms"][0]["incumbent"] = {"factory": "stub", "kwargs": {"profile": "q35-4b-fp16"}}
        cls.ctl, cls.fleet, cls.code = run_all(d)
        cls.blocks, cls.ends, cls.epochs, cls.events = logs(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.__exit__()

    def test_finishes_and_every_round_holds_the_composition(self):
        self.assertEqual(self.code, EXIT_OK)
        by = {}
        for b in self.blocks:
            by.setdefault(b["block_id"], []).append(b["arm_id"])
        self.assertEqual(len(by), 10)
        for bid, arms in by.items():
            self.assertEqual(sorted(arms), sorted(["I1-E", "A|fifo", "A|f1prio", "B|fifo"]), bid)

    def test_block_records_carry_the_design_label_and_the_propensity(self):
        self.assertTrue(all(b["design"] == "balanced" and abs(b["propensity"] - 0.25) < 1e-12 for b in self.blocks))
        law = make_design_law(self.ctl.cfg)
        for b in self.blocks:
            d = law.draw(b["node_index"], b["block_id"], b["context_id"])
            self.assertEqual((d.arm_id, d.propensity, d.u), (b["arm_id"], b["propensity"], b["rng"]["draw"]))

    def test_admission_is_pushed_with_every_configuration(self):
        seen = {}
        for e in self.epochs:
            cfg = e.get("config") or {}
            if cfg.get("arm_id"):
                seen.setdefault(cfg["arm_id"], set()).add(cfg.get("admission"))
        self.assertEqual(seen["A|f1prio"], {"f1prio"})
        self.assertEqual(seen["A|fifo"], {"fifo"})
        self.assertEqual(seen["B|fifo"], {"fifo"})
        self.assertEqual(seen["I1-E"], {"fifo"})


if __name__ == "__main__":
    unittest.main()
