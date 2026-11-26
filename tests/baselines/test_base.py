import json
import unittest

from ciara_baselines.base import Config, Incumbent, Obs, arm_id, family_key


class BaseTests(unittest.TestCase):
    def test_arm_id_matches_spec_example(self):
        self.assertEqual(arm_id("q35-4b-ih-w8a8", 16, 1), "q35-4b-ih-w8a8|b16|k1")

    def test_family_key(self):
        for v in (1, "1", "f1", "F1"):
            self.assertEqual(family_key(v), "f1")
        for v in (2, "2", "f2"):
            self.assertEqual(family_key(v), "f2")
        with self.assertRaises(ValueError):
            family_key(3)

    def test_config_dict_follows_spec_section_5(self):
        c = Config(arm_id="I1-E", profile="q35-4b-ih-w8a8", batch_cap=16, escalate_family2=True, block_id=17,
                   base_arm="q35-4b-ih-w8a8|b16|k1")
        d = c.to_dict()
        for k in ("arm_id", "profile", "batch_cap", "escalate_family2", "router", "block_id"):
            self.assertIn(k, d)
        self.assertEqual(d["router"], "kappa")
        self.assertNotIn("router_params", d)
        json.loads(c.to_json())

    def test_router_params_only_for_routing_incumbents(self):
        c = Config("I2", "p", 8, False, router="I2", router_params={"window": 100})
        self.assertEqual(c.to_dict()["router_params"], {"window": 100})

    def test_default_hooks(self):
        inc = Incumbent()
        self.assertEqual(inc.route({}, {}), "edge")
        self.assertEqual(inc.after_edge({}, None, {}), "accept")
        with self.assertRaises(NotImplementedError):
            inc.decide(Obs())


if __name__ == "__main__":
    unittest.main()
