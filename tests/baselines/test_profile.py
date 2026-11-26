import json
import tempfile
import unittest
from pathlib import Path

from ciara_baselines.profile import (
    CONCS, FamilyCurve, LoadCurve, OverheadTable, ProfileTable, _interp, build_load_curve, describe_columns,
)
from synth import basic_table, make_row, mean_lat, p95_lat


class InterpTests(unittest.TestCase):
    def test_interp_linear_and_clamped(self):
        xs, ys = [512, 2048, 4096], [1.0, 3.0, 7.0]
        self.assertAlmostEqual(_interp(xs, ys, 1280), 2.0)
        self.assertAlmostEqual(_interp(xs, ys, 100), 1.0)
        self.assertAlmostEqual(_interp(xs, ys, 9999), 7.0)
        self.assertAlmostEqual(_interp(xs, ys, 3072), 5.0)


class RowTests(unittest.TestCase):
    def setUp(self):
        self.row = make_row("r", t0=1.0, d=50.0, kv_tokens=40960, kv_gib=1.5)

    def test_memory_and_admissibility(self):
        self.assertAlmostEqual(self.row.mem_gib(8), 10.45)
        self.assertTrue(self.row.admissible(8))
        self.assertFalse(self.row.admissible(16))
        self.assertEqual(self.row.kv_seqs(), 8)

    def test_curve_matches_closed_form(self):
        cv = self.row.curve("f1", 16)
        self.assertEqual(cv.conc, tuple(float(c) for c in CONCS))
        self.assertAlmostEqual(cv.mean_lat[0], mean_lat(self.row, "f1"))
        self.assertAlmostEqual(cv.p95_lat[0], p95_lat(self.row, "f1"))
        self.assertAlmostEqual(cv.mean_lat[0], 2.0 + 30 / 50.0)
        self.assertAlmostEqual(cv.p95_lat[0], 2.0 + 59 / 50.0)
        for m, c in zip(cv.mean_lat, cv.cycle):
            self.assertAlmostEqual(m, c)

    def test_batch_cap_keeps_cells_up_to_the_cap(self):
        self.assertEqual(self.row.curve("f1", 8).conc, (1.0, 4.0, 8.0))
        self.assertEqual(self.row.curve("f1", 16).conc, (1.0, 4.0, 8.0, 16.0))

    def test_interp_cell_between_grid_concurrencies(self):
        cell = self.row.interp_cell(1000.0, 2.5)
        self.assertAlmostEqual(cell.ttft, 1.0)
        self.assertAlmostEqual(cell.dec_rate, 50.0)


class LoadCurveTests(unittest.TestCase):
    def curve(self, cycle, mean=None):
        n = len(cycle)
        mean = mean or [1.0 + i for i in range(n)]
        return FamilyCurve(tuple(float(c) for c in CONCS[:n]), (0.5,) * n, (0.02,) * n, tuple(mean),
                           tuple(m + 1 for m in mean), tuple(cycle))

    def test_load_is_concurrency_over_cycle(self):
        lc = build_load_curve({"f1": self.curve([1.0, 2.0, 2.5, 4.0])}, {"f1": 1.0})
        self.assertEqual(lc.loads, (1.0, 2.0, 3.2, 4.0))
        self.assertAlmostEqual(lc.max_load, 4.0)

    def test_points_past_the_throughput_peak_are_dropped(self):
        lc = build_load_curve({"f1": self.curve([1.0, 2.0, 2.5, 8.0])}, {"f1": 1.0})
        self.assertEqual(lc.conc, (1.0, 4.0, 8.0))
        self.assertAlmostEqual(lc.max_load, 3.2)
        self.assertIsNone(lc.at(3.3))

    def test_interpolation_and_idle_value(self):
        lc = build_load_curve({"f1": self.curve([1.0, 2.0, 2.5, 4.0], mean=[1.0, 2.0, 3.0, 4.0])}, {"f1": 1.0})
        self.assertAlmostEqual(lc.at(0.0)["f1"]["mean"], 1.0)
        self.assertAlmostEqual(lc.at(1.0)["f1"]["mean"], 1.0)
        self.assertAlmostEqual(lc.at(1.5)["f1"]["mean"], 1.5)
        self.assertAlmostEqual(lc.at(4.0)["f1"]["mean"], 4.0)

    def test_mixture_cycle_is_share_weighted(self):
        a = self.curve([1.0, 2.0, 4.0, 8.0])
        b = self.curve([3.0, 6.0, 12.0, 24.0])
        lc = build_load_curve({"f1": a, "f2": b}, {"f1": 0.5, "f2": 0.5})
        self.assertAlmostEqual(lc.loads[0], 0.5)

    def test_zero_share_family_is_ignored(self):
        a = self.curve([1.0, 2.0, 4.0, 8.0])
        b = self.curve([3.0, 6.0, 12.0, 24.0])
        lc = build_load_curve({"f1": a, "f2": b}, {"f1": 1.0, "f2": 0.0})
        self.assertNotIn("f2", lc.mean_lat)


class TableTests(unittest.TestCase):
    def test_roundtrip_through_json(self):
        t = basic_table(overhead_s=7.0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.json"
            t.save(p)
            u = ProfileTable.load(p)
        self.assertEqual(sorted(u.edge), sorted(t.edge))
        self.assertAlmostEqual(u.overhead.default_s, 7.0)
        self.assertEqual(u.edge["mid-int8"].families["f2"].out_tokens, 420.0)
        self.assertAlmostEqual(u.edge["mid-int8"].mem_gib(8), t.edge["mid-int8"].mem_gib(8))
        json.dumps(u.to_dict())

    def test_overhead_table(self):
        o = OverheadTable(default_s=5.0, pairs={"a>b": 2.0})
        self.assertEqual(o.switch_time_s("a", "a"), 0.0)
        self.assertEqual(o.switch_time_s("a", "b"), 2.0)
        self.assertEqual(o.switch_time_s("b", "a"), 5.0)

    def test_columns_documented(self):
        self.assertIn("mem_gib(b)", describe_columns())


if __name__ == "__main__":
    unittest.main()
