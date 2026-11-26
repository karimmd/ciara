import random
import unittest

from ciara import targets as V


def make_requests(n=4000, seed=1, fail=0.0, drift=1.0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        fam = 1 if rng.random() < 0.54 else 2
        cls = rng.choice("ABC")
        base = rng.lognormvariate(1.0 if fam == 1 else 2.2, 0.6)
        ok = rng.random() >= fail
        rows.append({"epoch": 20 + (i * 200) // n, "family": fam, "class": cls, "ok": ok, "latency_s": base * (1 + (drift - 1) * i / n) if ok else None,
                     "energy_mj": {"A": 5000.0, "B": 3000.0, "C": 800.0}[cls] * rng.uniform(0.8, 1.2)})
    return rows


class Targets(unittest.TestCase):
    def test_p95_with_failures(self):
        self.assertAlmostEqual(V.p95_with_failures([float(i) for i in range(1, 101)]), 95.05, places=2)
        self.assertIsNone(V.p95_with_failures([1.0] * 90 + [None] * 10))
        self.assertIsNotNone(V.p95_with_failures([1.0] * 98 + [None] * 2))

    def test_values_and_margin(self):
        rows = make_requests()
        v = V.compute(rows, warm_epochs=20)
        self.assertTrue(v["valid"], v["problems"])
        for f in ("f1", "f2"):
            self.assertAlmostEqual(v["families"][f]["target_s"], 1.25 * v["families"][f]["p95_s"])
        self.assertAlmostEqual(v["classes"]["A"]["budget_J"], 1.25 * v["classes"]["A"]["mean_energy_J"])
        self.assertAlmostEqual(v["classes"]["A"]["mean_energy_J"], 5.0, delta=0.2)

    def test_warm_up_is_excluded(self):
        rows = make_requests(2000)
        for r in rows[:50]:
            r["epoch"] = 3
            r["latency_s"] = 1e4
        v = V.compute(rows, warm_epochs=20)
        self.assertLess(v["families"]["f1"]["p95_s"], 100)

    def test_failures_invalidate(self):
        v = V.compute(make_requests(fail=0.2))
        self.assertFalse(v["valid"])

    def test_nonstationary_run_is_invalid(self):
        v = V.compute(make_requests(drift=6.0))
        self.assertFalse(v["valid"])
        self.assertTrue(any("no stationary regime" in p for p in v["problems"]))

    def test_too_few_requests(self):
        v = V.compute(make_requests(n=500))
        self.assertFalse(v["valid"])


if __name__ == "__main__":
    unittest.main()
