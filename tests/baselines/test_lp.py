import itertools
import random
import unittest

from ciara_baselines.lp import solve_distribution_lp


def _solve_square(M, y):
    n = len(M)
    A = [row[:] + [y[i]] for i, row in enumerate(M)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[piv][c]) < 1e-10:
            return None
        A[c], A[piv] = A[piv], A[c]
        for r in range(n):
            if r != c:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [A[i][n] / A[i][i] for i in range(n)]


def brute_force(c, A, b):
    n, d = len(c), len(A)
    cons = [([1.0 if j == i else 0.0 for j in range(n)], 0.0) for i in range(n)] + [(A[i], b[i]) for i in range(d)]
    best = None
    for tight in itertools.combinations(range(n + d), n - 1):
        p = _solve_square([[1.0] * n] + [cons[t][0] for t in tight], [1.0] + [cons[t][1] for t in tight])
        if p is None or min(p) < -1e-9:
            continue
        if any(sum(a * x for a, x in zip(A[i], p)) > b[i] + 1e-9 for i in range(d)):
            continue
        v = sum(ci * x for ci, x in zip(c, p))
        if best is None or v > best:
            best = v
    return best


class LPTests(unittest.TestCase):
    def test_hand_solved_two_arm_mixture(self):
        r = solve_distribution_lp([0.9, 0.5], [[0.06, 0.01]], [0.05])
        self.assertEqual(r.status, "optimal")
        self.assertAlmostEqual(r.p[0], 0.8, places=9)
        self.assertAlmostEqual(r.value, 0.9 * 0.8 + 0.5 * 0.2, places=9)

    def test_slack_constraint_gives_the_best_arm(self):
        r = solve_distribution_lp([0.2, 0.7, 0.4], [[0.1, 0.1, 0.1]], [0.5])
        self.assertEqual(r.p, [0.0, 1.0, 0.0])

    def test_infeasible_when_every_arm_exceeds_a_budget(self):
        r = solve_distribution_lp([0.5, 0.6], [[0.2, 0.3]], [0.1])
        self.assertEqual(r.status, "infeasible")
        self.assertIsNone(r.p)

    def test_negative_right_hand_side_is_infeasible_for_nonnegative_rows(self):
        self.assertEqual(solve_distribution_lp([1.0, 1.0], [[0.5, 0.5]], [-0.1]).status, "infeasible")

    def test_matches_vertex_enumeration_on_random_programs(self):
        rng = random.Random(7)
        n_opt = n_inf = 0
        for trial in range(400):
            n = rng.randint(2, 6)
            d = rng.randint(1, 4)
            c = [rng.random() for _ in range(n)]
            A = [[rng.random() for _ in range(n)] for _ in range(d)]
            b = [rng.uniform(0.1, 0.9) for _ in range(d)]
            if trial % 7 == 0:
                b[0] = -rng.random() * 0.1
            res = solve_distribution_lp(c, A, b)
            ref = brute_force(c, A, b)
            if ref is None:
                n_inf += 1
                self.assertEqual(res.status, "infeasible", trial)
                continue
            n_opt += 1
            self.assertEqual(res.status, "optimal", trial)
            self.assertAlmostEqual(res.value, ref, places=7, msg=str(trial))
            self.assertAlmostEqual(sum(res.p), 1.0, places=9)
            self.assertGreaterEqual(min(res.p), -1e-12)
            for i in range(d):
                self.assertLessEqual(sum(a * x for a, x in zip(A[i], res.p)), b[i] + 1e-7)
        self.assertGreater(n_opt, 100)
        self.assertGreater(n_inf, 10)

    def test_degenerate_ties_are_deterministic(self):
        a = solve_distribution_lp([1.0] * 5, [[0.0] * 5], [0.1])
        b = solve_distribution_lp([1.0] * 5, [[0.0] * 5], [0.1])
        self.assertEqual(a.p, b.p)
        self.assertAlmostEqual(a.value, 1.0)

    def test_shape_errors(self):
        with self.assertRaises(ValueError):
            solve_distribution_lp([1.0, 2.0], [[1.0]], [0.5])
        with self.assertRaises(ValueError):
            solve_distribution_lp([], [], [])


if __name__ == "__main__":
    unittest.main()
