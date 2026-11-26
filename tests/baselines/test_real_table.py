import os
import tempfile
import unittest
from pathlib import Path

from ciara_baselines.base import Obs
from ciara_baselines.i1 import I1E, I1Params, I1S, I1Selector, select_base_config
from ciara_baselines.mixorsplit import MixOrSplitIncumbent, MosParams
from ciara_baselines.profile import ADMISSIBLE_PROFILES, ProfileTable, build_table, pilot_dir
from ciara_baselines.race import RaceIncumbent, RaceParams
from synth import SHIPPED_TABLE

PILOT = pilot_dir()
HAVE = (PILOT / "p25" / "results" / "p25-summary.json").exists() and (PILOT / "p3" / "results" / "p3").exists()
TARGETS = {"f1": 8.0, "f2": 30.0}


class RealTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = ProfileTable.load(SHIPPED_TABLE)

    def test_candidate_grid_is_the_admissible_set(self):
        sel = I1Selector(self.table, I1Params(latency_targets_s=TARGETS))
        arms = [c.arm for c in sel.candidates()]
        self.assertEqual(len(arms), 18)
        self.assertNotIn("q35-4b-fp16|b16|k0", arms)
        self.assertIn("q35-4b-fp16|b8|k1", arms)
        for p in ADMISSIBLE_PROFILES[:2] + ADMISSIBLE_PROFILES[3:]:
            self.assertIn(f"{p}|b16|k0", arms)

    def test_kv_capacity_and_memory(self):
        r = self.table.edge["q35-4b-fp16"]
        self.assertEqual(r.kv_seqs(), 8)
        self.assertLessEqual(r.mem_gib(8), 10.45 + 1e-9)
        self.assertGreater(r.mem_gib(16), 10.45)
        for p in ("q35-4b-ih-w8a8", "q35-4b-ih-w4a16"):
            self.assertLessEqual(self.table.edge[p].mem_gib(16), 10.45)

    def test_reference_and_quality_columns(self):
        self.assertEqual(self.table.reference, "q35-4b-fp16")
        self.assertAlmostEqual(self.table.reference_accuracy("f1"), 0.8170731707317073)
        self.assertAlmostEqual(self.table.reference_accuracy("f2"), 0.50)
        self.assertAlmostEqual(self.table.cloud.families["f1"].accuracy, 0.7024390243902439)
        self.assertAlmostEqual(self.table.cloud.families["f2"].accuracy, 0.59)
        self.assertNotIn("f2", self.table.edge["q35-4b-ih-w4a16"].families)

    def test_accuracy_floor_of_5_points_against_4b_fp16(self):
        sel = I1Selector(self.table, I1Params(latency_targets_s=TARGETS))
        s = sel.select(sel.rates(0.2, None), 10.45)
        by = {e.cand.arm: e for e in s.evaluations}
        for p in ("q35-0.8b-fp16", "q35-2b-fp16"):
            self.assertFalse(by[f"{p}|b8|k0"].acc_ok)
        for p in ("q35-4b-fp16", "q35-4b-ih-w8a8"):
            self.assertTrue(by[f"{p}|b8|k0"].acc_ok)
        self.assertTrue(by["q35-4b-ih-w4a16|b8|k1"].feasible)
        self.assertFalse(by["q35-4b-ih-w4a16|b8|k0"].profiled)

    def test_static_selection_is_a_4b_profile_and_deterministic(self):
        p = I1Params(latency_targets_s=TARGETS, design_load=0.3)
        a = I1S(self.table, p).base_config()
        b = I1S(self.table, p).base_config()
        self.assertEqual(a.base_arm, b.base_arm)
        self.assertTrue(a.profile.startswith("q35-4b"))

    def test_epoch_controller_over_a_load_ramp(self):
        inc = I1E(self.table, I1Params(latency_targets_s=TARGETS))
        seen = []
        for epoch, lam in enumerate([0.05, 0.4, 1.6, 3.2, 0.4], 1):
            c = inc.decide(Obs(epoch=epoch, arrival_rate=lam, memory_budget_gib=10.45))
            seen.append((c.base_arm, inc.diagnostics()["selection_reason"], inc.diagnostics()["guard"]))
        self.assertTrue(seen[0][0].startswith("q35-4b") and seen[0][0].endswith("|k1"))
        self.assertEqual(seen[0][2], "init")
        self.assertEqual(seen[3][1], "fallback_cloud")
        self.assertEqual(seen[3][2], "forced_current_infeasible")
        self.assertEqual(seen[4][1], "argmin")
        self.assertTrue(seen[4][0].startswith("q35-4b"))

    def test_json_roundtrip_preserves_the_table(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.json"
            self.table.save(path)
            u = ProfileTable.load(path)
        sel_a = I1Selector(self.table, I1Params(latency_targets_s=TARGETS))
        sel_b = I1Selector(u, I1Params(latency_targets_s=TARGETS))
        ra = sel_a.select(sel_a.rates(0.4, None), 10.45)
        rb = sel_b.select(sel_b.rates(0.4, None), 10.45)
        self.assertEqual(ra.chosen.arm, rb.chosen.arm)
        self.assertAlmostEqual(ra.evaluation.mean_latency, rb.evaluation.mean_latency)

    @unittest.skipUnless(HAVE, "profiling outputs not found (set CIARA_PILOT_DIR)")
    def test_prebuilt_table_file_matches_the_loader(self):
        built = build_table(klass=os.environ.get("CIARA_PILOT_CLASS", "edge"))
        u = self.table
        self.assertEqual(sorted(u.edge), sorted(built.edge))
        for k, r in built.edge.items():
            self.assertEqual(u.edge[k].cells, r.cells)
            self.assertEqual(u.edge[k].families, r.families)
        self.assertEqual(u.meta["sources_sha256"], built.meta["sources_sha256"])

    def test_race_alpha_from_the_reference(self):
        p = RaceParams.from_table(self.table, delta=0.05)
        self.assertAlmostEqual(p.alpha["f1"], 1 - (0.8170731707317073 - 0.05))
        self.assertAlmostEqual(p.alpha["f2"], 0.55)

    def test_routing_incumbents_run_on_the_i1_static_base(self):
        p = I1Params(latency_targets_s=TARGETS, design_load=0.3)
        base = select_base_config(self.table, p)
        i3 = MixOrSplitIncumbent(self.table, base, MosParams())
        r = i3.route({"req_id": "x", "family": 2, "max_tokens": 1024, "messages": []},
                     {"inflight": 0, "queue_len": 0})
        self.assertIn(r, ("edge", "cloud"))
        i2 = RaceIncumbent(base, RaceParams.from_table(self.table))
        self.assertEqual(i2.decide(Obs(block_id=1)).router, "I2")
        self.assertEqual(i2.route({"req_id": "y", "family": 2, "race_u": 0.99, "messages": []}, {}), "cloud")


if __name__ == "__main__":
    unittest.main()
