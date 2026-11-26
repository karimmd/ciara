import contextlib
import importlib.util
import io
import json
import os
import unittest

from ciara import program as CT
from ciara.constants import BUDGET_KEYS, OUTCOMES

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUTS = os.path.join(REPO, "data", "inputs", "campaign")
DATA = os.environ.get("CIARA_CAMPAIGN_CSV") or os.path.join(REPO, "data", "campaign_node_blocks.csv")


def read(name):
    with open(os.path.join(INPUTS, name), encoding="utf-8") as fh:
        return json.load(fh)


def load_driver():
    spec = importlib.util.spec_from_file_location("certify_campaign", os.path.join(REPO, "scripts", "certify_campaign.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Inputs(unittest.TestCase):
    def test_the_inputs_agree_on_the_certified_set_and_the_design(self):
        sel, design, look = read("selection.json"), read("campaign-design.json"), read("look-day7.json")
        self.assertEqual(sel["S"], design["S"])
        self.assertEqual(sel["S"], look["arms"])
        self.assertEqual(sel["a0"], design["a0"])
        self.assertEqual(sorted(design["composition"]), sorted([sel["a0"], sel["a0"]] + [a for a in sel["S"] if a != sel["a0"]]))
        self.assertTrue(design["balanced"] and design["one_sided"])
        self.assertEqual((design["block_epochs"], design["burn_in_epochs"], design["l_dep_epochs"]), (50, 25, 1200))
        self.assertEqual(look["look_rounds"], 403)

    def test_the_stored_look_is_reproduced_by_the_program_from_its_own_radii(self):
        look = read("look-day7.json")
        members, a0 = look["arms"], look["a0"]
        tab = look["arms_table"]
        mu = {k: [tab[a]["outcomes"][k]["centre"] for a in members] for k in OUTCOMES}
        rr = {k: [tab[a]["outcomes"][k]["radius"] for a in members] for k in OUTCOMES}
        kw = CT.look_kwargs(look)
        zeta = kw.pop("zeta")
        self.assertTrue(kw["monotone"])
        out = CT.certified_policy(members, mu, rr, zeta, look["eta_dep"], a0, look["budgets"], one_sided_quality=True, **kw)
        self.assertEqual(out["lp_status"], "optimal")
        for a in members:
            self.assertAlmostEqual(out["pi"][a], look["pi"][a], places=9)
        self.assertAlmostEqual(out["quality_cost_bound"], look["quality_cost_bound"], places=9)
        for k in BUDGET_KEYS:
            self.assertAlmostEqual(out["upper_bounds"][k], look["upper_bounds"][k], places=9)
        self.assertEqual(out["branch"], look["branch"])


@unittest.skipUnless(os.path.exists(DATA), "data/campaign_node_blocks.csv is not in the repository (set CIARA_CAMPAIGN_CSV to its path)")
class Driver(unittest.TestCase):
    def test_check_passes_and_prints_the_two_looks(self):
        drv = load_driver()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = drv.main(["--data", DATA, "--check"])
        out = buf.getvalue()
        self.assertEqual(rc, 0, out)
        self.assertIn("Look at day 4: 230 rounds", out)
        self.assertIn("Look at day 7: 403 rounds", out)
        self.assertIn("RETAIN the incumbent I1-E", out)
        self.assertIn("DEPLOY the certified policy", out)
        self.assertIn("0.106  I1-E (incumbent)", out)
        self.assertIn("0.548  q35-4b-ih-w8a8|b8|c32|f1prio (priority arm)", out)
        self.assertIn("0.346  q35-4b-ih-w8a8|b8|c32|fifo (FIFO arm)", out)
        self.assertIn("<= 0.0842", out)
        self.assertIn("check passed", out)

    def test_a_changed_published_value_fails_the_check(self):
        drv = load_driver()
        drv.PUBLISHED_WEIGHTS = (0.2, 0.5, 0.3)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = drv.main(["--data", DATA, "--check"])
        self.assertEqual(rc, 1)
        self.assertIn("CHECK FAILED", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
