import unittest

import numpy as np

from ciara import radius as R
from ref import round_mean_radius as REF


class WorkedValues(unittest.TestCase):
    def test_band_and_eta(self):
        self.assertAlmostEqual(R.beta_dm(5, 25, 50), 0.00148, places=5)
        self.assertAlmostEqual(R.beta_dm(5, 25, 50, tau=2), 0.00220, places=5)
        self.assertAlmostEqual(R.eta_dep(5, 1200, 0), 0.00460, places=5)
        self.assertAlmostEqual(R.eta_dep(5, 1200, 2), 0.00626, places=5)
        self.assertAlmostEqual(R.eta_dep(5, 400, 2), 0.01879, places=5)

    def test_level(self):
        self.assertAlmostEqual(R.level(3), 6.867, places=3)
        self.assertAlmostEqual(R.level(3, one_sided=True), 6.174, places=3)

    def test_planned_rounds(self):
        self.assertAlmostEqual(R.planned_rounds(1 / 6, 5, 403), 241.08, places=1)
        self.assertEqual(R.planned_rounds_balanced(403), 403.0)

    def test_oracle_radius_matches_worked_values(self):
        band = R.beta_dm(5, 25, 50, tau=2)
        u = R.planned_rounds(1 / 6, 5, 403)
        L = R.level(3)
        for m, want in ((0.005, 0.0311), (0.01, 0.0311), (0.02, 0.0313), (0.05, 0.0317)):
            s2 = m * (1 - m) / 250
            self.assertAlmostEqual(R.oracle_radius(s2, u, L, band), want, places=4)
            self.assertGreaterEqual(R.closed_radius(s2, u, L, band), R.oracle_radius(s2, u, L, band) - 1e-12)

    def test_closed_form_values(self):
        band = R.beta_dm(5, 25, 50, tau=2)
        u = R.planned_rounds(1 / 6, 5, 403)
        L = R.level(3)
        for m, want in ((0.005, 0.0318), (0.01, 0.0322), (0.02, 0.0328), (0.05, 0.0340)):
            self.assertAlmostEqual(R.closed_radius(m * (1 - m) / 250, u, L, band), want, places=4)

    def test_empty_arm_has_an_infinite_radius(self):
        self.assertEqual(R.closed_radius(0.01, 0, 6.0, 0.002), float("inf"))
        self.assertEqual(R.oracle_radius(0.01, 0, 6.0, 0.002), float("inf"))

    def test_empty_window_is_refused(self):
        with self.assertRaises(ValueError):
            R.beta_dm(5, 50, 50)


class RadiusMatchesReference(unittest.TestCase):
    def synthetic(self, seed, nb=120, N=5, K=3, nout=8):
        rng = np.random.default_rng(seed)
        q = np.array([2 / 3, 1 / 6, 1 / 6])
        Z = rng.choice(K, size=(nb, N), p=q)
        base = rng.uniform(0.02, 0.4, size=(K, nout))
        F = np.clip(base[Z] + rng.normal(0, 0.03, size=(nb, N, nout)), 0, 1)
        return Z, F, q

    def test_matches_reference_without_missing(self):
        for seed in (1, 2, 3):
            Z, F, q = self.synthetic(seed)
            N, K = 5, 3
            L = R.level(3)
            look_k = [30, 70, 120]
            k7 = 403
            t, l, b = 5.0, 50, 25
            nplan = k7 * (1 - (1 - q) ** N)
            band = np.full(K, R.beta_dm(t, b, l))
            out = R.c1r_intervals(Z, F, N, L, look_k, nplan, band, n_arms=K)
            for lo in (0, 4):
                ref_c, ref_r, ref_l = REF.c1_intervals(Z.T[None], F.transpose(1, 0, 2)[None, :, :, lo:lo + 4], N, l, b, t, L, look_k, k7, q, K=K)
                np.testing.assert_allclose(out["centre"][:, lo:lo + 4, :], ref_c[0], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(out["radius"][:, lo:lo + 4, :], ref_r[0], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(out["lam_sum"][:, lo:lo + 4, :], ref_l[0], rtol=1e-10, atol=1e-12)

    def test_missing_values_are_skipped(self):
        Z, F, q = self.synthetic(5)
        N, K = 5, 3
        L = R.level(3)
        nplan = 403 * (1 - (1 - q) ** N)
        band = np.full(K, 0.002)
        full = R.c1r_intervals(Z, F, N, L, [120], nplan, band, n_arms=K)
        F2 = F.copy()
        F2[:, :, 2] = np.nan
        part = R.c1r_intervals(Z, F2, N, L, [120], nplan, band, n_arms=K)
        self.assertTrue(np.all(np.isinf(part["radius"][0, 2])))
        np.testing.assert_allclose(part["radius"][0, 3], full["radius"][0, 3])

    def test_radius_shrinks_with_rounds(self):
        Z, F, q = self.synthetic(7, nb=400)
        L = R.level(3)
        nplan = 403 * (1 - (1 - q) ** 5)
        out = R.c1r_intervals(Z, F, 5, L, [50, 100, 400], nplan, np.full(3, 0.002), n_arms=3)
        r = out["radius"][:, 1, 1]
        self.assertTrue(r[0] > r[1] > r[2])


if __name__ == "__main__":
    unittest.main()
