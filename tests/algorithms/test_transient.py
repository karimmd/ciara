import os
import unittest

from ciara import radius as R
from ciara.transient import TauTable

TABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "inputs", "campaign", "switch-transient.json")


class Tau(unittest.TestCase):
    def setUp(self):
        self.t = TauTable.load(TABLE)

    def test_values_of_the_switch_table(self):
        self.assertEqual(self.t.pair("q35-2b-fp16", "q35-4b-ih-w4a16"), 1)
        self.assertEqual(self.t.pair("q35-2b-fp16", "q35-4b-ih-w8a8"), 4)
        self.assertEqual(self.t.pair("q35-4b-fp16", "q35-4b-ih-w4a16"), 16)
        self.assertEqual(self.t.pair("q35-2b-fp16", "q35-4b-fp16"), 10)

    def test_proxies_and_same_profile(self):
        self.assertEqual(self.t.pair("q35-0.8b-fp16", "q35-4b-fp16"), self.t.pair("q35-2b-fp16", "q35-4b-fp16"))
        self.assertEqual(self.t.pair("q35-4b-fp16", "q35-0.8b-fp16"), self.t.pair("q35-4b-fp16", "q35-2b-fp16"))
        self.assertEqual(self.t.pair("q35-2b-fp16", "q35-0.8b-fp16"), self.t.max_sleep)
        self.assertEqual(self.t.pair("q35-4b-ih-w4a16", "q35-4b-ih-w4a16"), 1)

    def test_arm_tau_within_a_set(self):
        profiles = {"I1-E": ["q35-4b-ih-w4a16"], "a": ["q35-2b-fp16"], "b": ["q35-4b-fp16"]}
        self.assertEqual(self.t.arm_tau("a", profiles, ["I1-E", "a"]), 2)
        self.assertEqual(self.t.arm_tau("a", profiles, ["I1-E", "a", "b"]), 11)
        self.assertEqual(self.t.arm_tau("I1-E", profiles, ["I1-E", "a", "b"]), 16)
        self.assertEqual(self.t.arm_tau("I1-E", profiles, ["I1-E"]), 1)

    def test_set_tau_counts_pairs_between_other_members(self):
        profiles = {"I1-S": ["q35-4b-ih-w4a16"], "a": ["q35-4b-ih-w8a8"], "b": ["q35-2b-fp16"]}
        members = ["I1-S", "a", "b"]
        self.assertEqual(self.t.arm_tau("I1-S", profiles, members), 3)
        self.assertEqual(self.t.set_tau(profiles, members), 4)
        self.assertEqual(self.t.set_tau(profiles, ["I1-S"]), 1)
        self.assertEqual(self.t.set_tau({"I1-E": ["q35-4b-ih-w4a16", "q35-4b-fp16"]}, ["I1-E"]), 16)
        self.assertEqual(self.t.set_tau({**profiles, "f": ["q35-4b-fp16"]}, ["I1-S", "a", "f"]), 16)

    def test_pass_rule_values_with_these_taus(self):
        self.assertAlmostEqual(R.eta_dep(5, 1200, 2), 0.00626, places=5)
        self.assertGreater(R.eta_dep(5, 1200, 16), 0.017)
        self.assertGreater(R.beta_dm(5, 25, 50, 16), 0.02)


if __name__ == "__main__":
    unittest.main()
