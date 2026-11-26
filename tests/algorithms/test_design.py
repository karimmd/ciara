import math
import unittest

import numpy as np

from ciara.design import BalancedLaw, BlockGrid, DesignLaw, certification_propensities

CHI2_P001 = {2: 13.816, 4: 18.467, 20: 45.315}


def screening_law(n_arms=21, seed=7) -> DesignLaw:
    ids = tuple(f"a{i}" for i in range(n_arms))
    return DesignLaw("screening", ids, seed, {}, [1.0 / n_arms] * n_arms)


def cert_law(eps=0.5, seed=7) -> DesignLaw:
    ids = ("a0", "a1", "a2")
    return DesignLaw("certification", ids, seed, {}, certification_propensities(3, 0, eps), epsilon=eps,
                     incumbent="a0")


class PropensityFormula(unittest.TestCase):
    def test_certification_propensities_of_the_safe_design_law(self):
        for eps in (0.1, 0.5, 1.0):
            for n in (2, 3, 5):
                for a0 in range(n):
                    p = certification_propensities(n, a0, eps)
                    self.assertAlmostEqual(sum(p), 1.0, places=12)
                    for a in range(n):
                        want = (1 - eps) * (1 if a == a0 else 0) + eps / n
                        self.assertAlmostEqual(p[a], want, places=12)
                    self.assertAlmostEqual(min(p), eps / n, places=12)

    def test_eps_half_three_arms(self):
        p = certification_propensities(3, 0, 0.5)
        self.assertAlmostEqual(p[0], 0.5 + 0.5 / 3)
        self.assertAlmostEqual(p[1], 1 / 6)
        self.assertAlmostEqual(p[2], 1 / 6)


class EmpiricalPropensity(unittest.TestCase):
    N_NODES, N_BLOCKS = 5, 8000

    def freq(self, law):
        n = len(law.arm_ids)
        counts = [0] * n
        total = 0
        for node in range(1, self.N_NODES + 1):
            for blk in range(self.N_BLOCKS):
                d = law.draw(node, blk, "pooled")
                counts[d.arm_index] += 1
                total += 1
        return counts, total

    def test_certification_draw_frequencies_match_e(self):
        law = cert_law()
        counts, total = self.freq(law)
        for a, p in enumerate(law.propensities("pooled")):
            se = math.sqrt(p * (1 - p) / total)
            self.assertLess(abs(counts[a] / total - p), 4.5 * se, f"arm {a}")

    def test_screening_draw_frequencies_uniform_21_arms(self):
        law = screening_law(21)
        counts, total = self.freq(law)
        exp = total / 21
        chi2 = sum((c - exp) ** 2 / exp for c in counts)
        self.assertLess(chi2, CHI2_P001[20])

    def test_recorded_propensity_is_e_of_drawn_arm(self):
        law = cert_law(0.3)
        vec = law.propensities("pooled")
        for node in (1, 2, 3):
            for blk in range(300):
                d = law.draw(node, blk, "pooled")
                self.assertEqual(d.propensity, vec[d.arm_index])
                self.assertEqual(law.arm_ids[d.arm_index], d.arm_id)

    def test_horvitz_thompson_identity(self):
        law = cert_law()
        total = 40000
        est = [0.0, 0.0, 0.0]
        for i in range(total):
            d = law.draw(1 + i % 5, i // 5, "pooled")
            est[d.arm_index] += 1.0 / d.propensity
        for a in range(3):
            self.assertLess(abs(est[a] / total - 1.0), 0.03)

    def test_zero_mass_arms_never_drawn(self):
        law = DesignLaw("deployment", ("a", "b", "c"), 5, {}, [0.0, 0.7, 0.3])
        seen = {law.draw(1, b, "pooled").arm_id for b in range(3000)}
        self.assertEqual(seen, {"b", "c"})

    def test_per_context_policy(self):
        law = DesignLaw("deployment", ("a", "b"), 5, {"x1": [1.0, 0.0], "x2": [0.0, 1.0]}, [0.5, 0.5])
        self.assertTrue(all(law.draw(1, b, "x1").arm_id == "a" for b in range(50)))
        self.assertTrue(all(law.draw(1, b, "x2").arm_id == "b" for b in range(50)))


class Reproducibility(unittest.TestCase):
    def test_same_seed_same_draws_in_any_call_order(self):
        a, b = screening_law(seed=11), screening_law(seed=11)
        keys = [(n, k) for n in range(1, 6) for k in range(60)]
        fwd = {key: a.draw(key[0], key[1], "pooled").u for key in keys}
        rev = {key: b.draw(key[0], key[1], "pooled").u for key in reversed(keys)}
        self.assertEqual(fwd, rev)

    def test_different_seed_different_draws(self):
        a, b = screening_law(seed=11), screening_law(seed=12)
        same = sum(a.draw(1, k, "p").arm_id == b.draw(1, k, "p").arm_id for k in range(400))
        self.assertLess(same, 60)

    def test_logged_entropy_and_draw_recompute_the_arm(self):
        law = cert_law(seed=99)
        d = law.draw(3, 41, "pooled")
        self.assertEqual(d.entropy, (99, 3, 41))
        u = float(np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(d.entropy)))).random())
        self.assertEqual(u, d.u)
        cum = np.cumsum(law.propensities("pooled"))
        self.assertEqual(law.arm_ids[int(np.searchsorted(cum, u, side="right"))], d.arm_id)

    def test_draw_does_not_depend_on_context_except_through_vector(self):
        law = screening_law()
        self.assertEqual(law.draw(1, 5, "x").u, law.draw(1, 5, "y").u)


class Independence(unittest.TestCase):
    N = 20000

    def table(self, law, node_a, node_b):
        t = np.zeros((3, 3))
        for blk in range(self.N):
            t[law.draw(node_a, blk, "p").arm_index, law.draw(node_b, blk, "p").arm_index] += 1
        return t

    def test_arms_independent_across_nodes(self):
        law = cert_law()
        t = self.table(law, 1, 2)
        row, col = t.sum(1, keepdims=True), t.sum(0, keepdims=True)
        exp = row * col / t.sum()
        chi2 = ((t - exp) ** 2 / exp).sum()
        self.assertLess(chi2, CHI2_P001[4])

    def test_all_node_pairs_uncorrelated(self):
        law = screening_law()
        us = np.array([[law.draw(n, b, "p").u for b in range(self.N)] for n in range(1, 6)])
        c = np.corrcoef(us)
        off = c[~np.eye(5, dtype=bool)]
        self.assertLess(np.abs(off).max(), 4.5 / math.sqrt(self.N))

    def test_independent_across_blocks(self):
        law = screening_law()
        u = np.array([law.draw(1, b, "p").u for b in range(self.N)])
        for lag in (1, 2, 5):
            r = np.corrcoef(u[:-lag], u[lag:])[0, 1]
            self.assertLess(abs(r), 4.5 / math.sqrt(self.N))

    def test_node_streams_do_not_collide(self):
        law = screening_law()
        a = [law.draw(1, b, "p").u for b in range(200)]
        b = [law.draw(2, b, "p").u for b in range(200)]
        self.assertEqual(len(set(a) & set(b)), 0)


class BalancedRounds(unittest.TestCase):
    COMP_CAMPAIGN = ["I1-E", "I1-E", "A|fifo", "B|fifo"]
    COMP_PROBE = ["I1-E", "A|fifo", "A|f1prio", "B|fifo"]

    def law(self, comp):
        return BalancedLaw(20261901, tuple(comp), (1, 2, 3, 4))

    def test_every_round_has_exactly_the_composition(self):
        for comp in (self.COMP_CAMPAIGN, self.COMP_PROBE):
            law = self.law(comp)
            for b in range(300):
                arms = sorted(law.draw(n, b, "pooled").arm_id for n in (1, 2, 3, 4))
                self.assertEqual(arms, sorted(comp))

    def test_propensities_are_counts_over_nodes_and_arms_in_first_appearance_order(self):
        law = self.law(self.COMP_CAMPAIGN)
        self.assertEqual(law.arm_ids, ("I1-E", "A|fifo", "B|fifo"))
        self.assertEqual(law.propensities("pooled"), [0.5, 0.25, 0.25])
        self.assertEqual(law.draw(1, 0, "pooled").propensity, law.propensity_of(law.draw(1, 0, "pooled").arm_id, "pooled"))
        self.assertEqual(self.law(self.COMP_PROBE).propensities("pooled"), [0.25] * 4)

    def test_reproducible_whatever_the_order_of_calls(self):
        a, b = self.law(self.COMP_PROBE), self.law(self.COMP_PROBE)
        fwd = [[a.draw(n, k, "x").arm_id for n in (1, 2, 3, 4)] for k in range(50)]
        rev = [[b.draw(n, k, "x").arm_id for n in (4, 3, 2, 1)][::-1] for k in reversed(range(50))][::-1]
        self.assertEqual(fwd, rev)

    def test_permutations_uniform_and_independent_across_rounds(self):
        law = self.law(self.COMP_PROBE)
        n = 24000
        cnt = {}
        seq = []
        for b in range(n):
            key = tuple(law.draw(k, b, "x").arm_id for k in (1, 2, 3, 4))
            cnt[key] = cnt.get(key, 0) + 1
            seq.append(key)
        self.assertEqual(len(cnt), 24)
        exp = n / 24
        chi2 = sum((v - exp) ** 2 / exp for v in cnt.values())
        self.assertLess(chi2, 52.62)
        keys = sorted(cnt)
        ix = {k: i for i, k in enumerate(keys)}
        t = np.zeros((24, 24))
        for x, y in zip(seq[:-1], seq[1:]):
            t[ix[x], ix[y]] += 1
        e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
        chi2 = ((t - e) ** 2 / e).sum()
        self.assertLess(chi2, 23 * 23 + 6 * math.sqrt(2 * 23 * 23))

    def test_marginal_per_node_is_the_propensity(self):
        law = self.law(self.COMP_CAMPAIGN)
        for node in (1, 2, 3, 4):
            arms = [law.draw(node, b, "x").arm_id for b in range(8000)]
            for a, p in zip(law.arm_ids, law.propensities("x")):
                self.assertAlmostEqual(arms.count(a) / 8000, p, delta=0.02)

    def test_composition_must_match_the_nodes(self):
        with self.assertRaises(ValueError):
            BalancedLaw(1, ("I1-E", "A|fifo", "B|fifo"), (1, 2, 3, 4))


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


if __name__ == "__main__":
    unittest.main()
