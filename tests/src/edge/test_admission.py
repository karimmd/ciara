from __future__ import annotations

import os
import random
import sys
import unittest


from ciara_edge.admission import F1PRIO, FIFO, AdmissionQueue


def fill(q, fams):
    return [q.push(object(), f) for f in fams]


def drain(q, mode):
    out = []
    while len(q):
        out.append(q.pop(mode))
    return out


class Order(unittest.TestCase):
    def test_fifo_is_arrival_order_whatever_the_families(self):
        q = AdmissionQueue()
        ws = fill(q, [2, 1, 2, 1, None, 2])
        got = drain(q, FIFO)
        self.assertEqual([w.arrive_seq for w in got], [w.arrive_seq for w in ws])
        self.assertTrue(all(w.overtaken == 0 for w in got))

    def test_f1prio_family1_ahead_of_every_family2_fifo_within_a_family(self):
        q = AdmissionQueue()
        ws = fill(q, [2, 2, 1, 2, 1, 1, 2])
        got = drain(q, F1PRIO)
        self.assertEqual([ws.index(w) for w in got], [2, 4, 5, 0, 1, 3, 6])

    def test_overtaken_counts_earlier_arrivals_passed_over(self):
        q = AdmissionQueue()
        fill(q, [2, 2, 1, 2, 1])
        got = drain(q, F1PRIO)
        self.assertEqual([w.overtaken for w in got], [2, 3, 0, 0, 0])
        self.assertEqual([w.admit_seq for w in got], [1, 2, 3, 4, 5])
        self.assertEqual([w.queue_len_admit for w in got], [5, 4, 3, 2, 1])
        self.assertTrue(all(w.mode == F1PRIO for w in got))

    def test_unknown_family_counts_as_family_2(self):
        q = AdmissionQueue()
        ws = fill(q, [None, 2, 1])
        self.assertEqual([ws.index(w) for w in drain(q, F1PRIO)], [2, 0, 1])

    def test_only_family2_queued_is_fifo_under_f1prio(self):
        q = AdmissionQueue()
        ws = fill(q, [2, 2, 2])
        self.assertEqual([ws.index(w) for w in drain(q, F1PRIO)], [0, 1, 2])

    def test_a_late_family2_does_not_take_a_slot_ahead_of_an_earlier_family1_and_mode_can_change_between_pops(self):
        q = AdmissionQueue()
        ws = fill(q, [2, 1])
        self.assertIs(q.pop(FIFO), ws[0])
        q.push(object(), 2)
        self.assertIs(q.pop(F1PRIO), ws[1])

    def test_remove_and_empty(self):
        q = AdmissionQueue()
        a, b = fill(q, [2, 1])
        self.assertTrue(q.remove(a))
        self.assertFalse(q.remove(a))
        self.assertEqual(len(q), 1)
        self.assertIs(q.pop(F1PRIO), b)
        self.assertIsNone(q.pop(FIFO))

    def test_unknown_mode_rejected(self):
        q = AdmissionQueue()
        fill(q, [1])
        with self.assertRaises(ValueError):
            q.pop("lifo")

    def test_property_f1prio_never_admits_family2_while_family1_waits_and_keeps_order(self):
        rng = random.Random(7)
        for _ in range(200):
            q = AdmissionQueue()
            ws = []
            live = []
            order = []
            for step in range(rng.randint(1, 40)):
                if live and rng.random() < 0.4:
                    w = q.pop(F1PRIO)
                    live.remove(w)
                    if w.family != 1:
                        self.assertFalse(any(x.family == 1 for x in live))
                    for fam in (1, 2):
                        same = [x.arrive_seq for x in order if x.family == fam]
                        self.assertEqual(same, sorted(same))
                    order.append(w)
                else:
                    w = q.push(object(), rng.choice([1, 2]))
                    live.append(w)
            self.assertEqual(len(q), len(live))


if __name__ == "__main__":
    unittest.main()
