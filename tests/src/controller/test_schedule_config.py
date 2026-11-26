import json
import os
import unittest

from ciara_control.config import ConfigError, load_config, make_arm_id, parse_config
from ciara_control.schedule import BlockGrid, ContextSchedule
from helpers import TmpDir, base_config, deep


class GridBookkeeping(unittest.TestCase):
    def setUp(self):
        self.g = BlockGrid(epoch_ns=30_000_000_000, block_epochs=50, burn_in=25, n_blocks=42)

    def test_totals(self):
        self.assertEqual(self.g.total_epochs, 2100)

    def test_block_and_epoch_in_block(self):
        self.assertEqual(self.g.block_of(0), 0)
        self.assertEqual(self.g.block_of(49), 0)
        self.assertEqual(self.g.block_of(50), 1)
        self.assertEqual(self.g.epoch_in_block(51), 1)
        self.assertEqual((self.g.block_start(3), self.g.block_end(3)), (150, 200))

    def test_burn_in_is_first_b_epochs_of_each_block(self):
        flags = [self.g.is_burn_in(e) for e in range(100)]
        self.assertEqual(flags[:25], [True] * 25)
        self.assertEqual(flags[25:50], [False] * 25)
        self.assertEqual(flags[50:75], [True] * 25)
        self.assertEqual(sum(flags), 50)

    def test_measurement_range(self):
        self.assertEqual(self.g.measurement_range(2), (125, 150))

    def test_block_starts(self):
        self.assertEqual([e for e in range(0, 160) if self.g.is_block_start(e)], [0, 50, 100, 150])

    def test_boundaries_are_on_one_grid(self):
        t0 = 1_000
        self.assertEqual(self.g.boundary_ns(t0, 0), 1_000)
        self.assertEqual(self.g.boundary_ns(t0, 3), 1_000 + 90_000_000_000)

    def test_next_boundary_with_margin(self):
        t0, d = 0, self.g.epoch_ns
        self.assertEqual(self.g.next_boundary_epoch(t0, 0), 0)
        self.assertEqual(self.g.next_boundary_epoch(t0, 1), 1)
        self.assertEqual(self.g.next_boundary_epoch(t0, d), 1)
        self.assertEqual(self.g.next_boundary_epoch(t0, d, margin_ns=5_000_000_000), 2)
        self.assertEqual(self.g.next_boundary_epoch(-10 * d, 0), 10)
        self.assertEqual(self.g.next_boundary_epoch(10 * d, 0), 0)

    def test_zero_burn_in(self):
        g = BlockGrid(1, 3, 0, 2)
        self.assertFalse(any(g.is_burn_in(e) for e in range(6)))


class ContextScheduleTests(unittest.TestCase):
    def test_default_and_segments(self):
        cs = ContextSchedule("pooled", [{"start_epoch": 100, "end_epoch": 200, "context_id": "heavy"}])
        self.assertEqual(cs.context_at(99), "pooled")
        self.assertEqual(cs.context_at(100), "heavy")
        self.assertEqual(cs.context_at(199), "heavy")
        self.assertEqual(cs.context_at(200), "pooled")

    def test_misaligned(self):
        cs = ContextSchedule("p", [{"start_epoch": 0, "end_epoch": 50, "context_id": "a"},
                                   {"start_epoch": 50, "end_epoch": 75, "context_id": "b"}])
        self.assertEqual(cs.misaligned_segments(50), [(50, 75, "b")])

    def test_load_relative_to_config(self):
        with TmpDir() as d:
            with open(os.path.join(d, "ctx.json"), "w") as fh:
                json.dump({"default_context": "light",
                           "segments": [{"start_epoch": 4, "end_epoch": 8, "context_id": "heavy"}]}, fh)
            cs = ContextSchedule.load("ctx.json", "pooled", base_dir=d)
            self.assertEqual(cs.context_at(0), "light")
            self.assertEqual(cs.context_at(5), "heavy")


class ConfigValidation(unittest.TestCase):
    def good(self):
        return base_config("/x")

    def test_good_config_parses(self):
        cfg = parse_config(self.good())
        self.assertEqual(cfg.total_epochs, 32)
        self.assertEqual(list(cfg.arms)[0], make_arm_id("q35-4b-fp16", 8, False))
        self.assertEqual(cfg.arms["q35-4b-ih-w8a8|b16|k1"].config_object(17)["block_id"], 17)

    def test_spec_example_configuration_object(self):
        d = self.good()
        d["arms"][1]["profile"] = "q35-4b-ih-w8a8"
        cfg = parse_config(d)
        obj = cfg.arms["q35-4b-ih-w8a8|b16|k1"].config_object(17)
        self.assertEqual(obj, {"arm_id": "q35-4b-ih-w8a8|b16|k1", "profile": "q35-4b-ih-w8a8", "batch_cap": 16,
                               "escalate_family2": True, "router": "kappa", "block_id": 17, "admission": "fifo"})

    def test_all_problems_reported_at_once(self):
        d = self.good()
        d["block_epochs"] = 4
        d["burn_in_epochs"] = 4
        d["epoch_s"] = -1
        d["design"] = {"label": "nonsense"}
        with self.assertRaises(ConfigError) as cm:
            parse_config(d)
        self.assertGreaterEqual(len(cm.exception.problems), 3)

    def test_certification_requires_incumbent_in_set_and_eps(self):
        d = self.good()
        d["design"] = {"label": "certification", "arms": ["q35-2b-fp16|b8|k1"], "incumbent": "I1-E", "epsilon": 0.5}
        with self.assertRaises(ConfigError):
            parse_config(d)
        d["design"] = {"label": "certification", "arms": ["I1-E", "q35-2b-fp16|b8|k1"], "incumbent": "I1-E",
                       "epsilon": 0}
        with self.assertRaises(ConfigError):
            parse_config(d)
        d["design"]["epsilon"] = 0.5
        parse_config(d)

    def test_deployment_policy_must_sum_to_one(self):
        d = self.good()
        d["design"] = {"label": "deployment", "policy": {"I1-E": 0.5, "q35-2b-fp16|b8|k1": 0.4}}
        with self.assertRaises(ConfigError):
            parse_config(d)
        d["design"]["policy"]["q35-2b-fp16|b8|k1"] = 0.5
        parse_config(d)

    def test_unknown_arm_duplicate_arm_and_node(self):
        d = self.good()
        d["design"] = {"label": "screening", "arms": ["nope"]}
        with self.assertRaises(ConfigError):
            parse_config(d)
        d = self.good()
        d["arms"].append(deep(d["arms"][0]))
        with self.assertRaises(ConfigError):
            parse_config(d)
        d = self.good()
        d["nodes"].append({"id": "E1"})
        with self.assertRaises(ConfigError):
            parse_config(d)

    def test_incumbent_arm_needs_factory(self):
        d = self.good()
        d["arms"][3]["incumbent"] = {}
        with self.assertRaises(ConfigError):
            parse_config(d)

    def test_cloud_max_in_flight_bounded_by_spec(self):
        d = self.good()
        d["cloud"] = {"max_in_flight": 17}
        with self.assertRaises(ConfigError):
            parse_config(d)

    def test_design_hash_ignores_endpoints_but_not_seed(self):
        a = parse_config(self.good())
        d = self.good()
        d["nodes"][0]["gateway"] = "http://other:1"
        d["control"] = {"retries": 9}
        self.assertEqual(a.design_hash, parse_config(d).design_hash)
        d["seed"] += 1
        self.assertNotEqual(a.design_hash, parse_config(d).design_hash)

    def test_default_engine_manager_port(self):
        d = self.good()
        d["nodes"] = [{"id": "E3", "gateway": "http://g:1"}]
        self.assertEqual(parse_config(d).nodes[0].engine_manager, "http://127.0.0.1:9230")

    def test_yaml_and_json_load_the_same(self):
        d = self.good()
        with TmpDir() as t:
            pj, py = os.path.join(t, "c.json"), os.path.join(t, "c.yaml")
            with open(pj, "w") as fh:
                json.dump(d, fh)
            import yaml
            with open(py, "w") as fh:
                yaml.safe_dump(d, fh)
            self.assertEqual(load_config(pj).design_hash, load_config(py).design_hash)

    def test_shipped_example_configs_validate(self):
        here = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "configs", "controller")
        for name in sorted(os.listdir(here)):
            if name.endswith((".yaml", ".yml", ".json")) and not name.startswith("ctx-"):
                cfg = load_config(os.path.join(here, name))
                self.assertGreaterEqual(len(cfg.arms), 1, name)


if __name__ == "__main__":
    unittest.main()
