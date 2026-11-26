from __future__ import annotations

import os
import sys
import unittest


from ciara_edge import routers
from ciara_edge.routers import request_half


def meta(rid, family=2, cls="A"):
    return {"req_id": rid, "req": rid, "family": family, "class": cls, "node": "E1"}


def cfg(router, **kw):
    c = {"arm_id": "x", "profile": "q35-4b-ih-w8a8", "batch_cap": 16, "escalate_family2": True, "router": router}
    c.update(kw)
    return c


class Listed(unittest.TestCase):
    def test_routers_are_listed(self):
        for n in ("kappa", "kappaA", "kappaH", "I2", "I3", "RecServe"):
            self.assertIn(n, routers.BUILTIN)
        self.assertFalse(routers.get_router("kappaA").needs_after_edge)
        self.assertEqual(routers.router_signature("kappaH", {}, 3), routers.router_signature("kappaH", {}, 4))

    def test_default_kappa_unchanged(self):
        r = routers.get_router("kappa")
        r.configure(cfg("kappa", escalate_family2=True))
        self.assertEqual(r.route(meta("a", 2, "C"), {}), "cloud")
        self.assertEqual(r.route(meta("a", 1, "A"), {}), "edge")
        r.configure(cfg("kappa", escalate_family2=False))
        self.assertEqual(r.route(meta("a", 2, "A"), {}), "edge")


class ClassA(unittest.TestCase):
    def test_only_family2_of_class_A(self):
        r = routers.get_router("kappaA")
        r.configure(cfg("kappaA"))
        self.assertEqual(r.route(meta("n1A001-00001", 2, "A"), {}), "cloud")
        self.assertEqual(r.route(meta("n1B001-00001", 2, "B"), {}), "edge")
        self.assertEqual(r.route(meta("n1C001-00001", 2, "C"), {}), "edge")
        self.assertEqual(r.route(meta("n1A001-00002", 1, "A"), {}), "edge")
        self.assertEqual(r.route(meta("n1A001-00003", None, "A"), {}), "edge")


class Half(unittest.TestCase):
    ids = [f"n{n}{c}{d:03d}-{k:05d}" for n in range(1, 5) for c in "ABC" for d in range(20) for k in range(10)]

    def test_same_request_same_way_in_every_arm_and_process(self):
        a = routers.get_router("kappaH")
        b = routers.get_router("kappaH")
        a.configure(cfg("kappaH", arm_id="p|b8|kH", batch_cap=8, block_id=1))
        b.configure(cfg("kappaH", arm_id="q|b16|kH", batch_cap=16, block_id=7))
        for rid in self.ids[:300]:
            self.assertEqual(a.route(meta(rid), {"queue_len": 0}), b.route(meta(rid), {"queue_len": 50}))
        self.assertEqual(request_half("n1A007-00012"), request_half("n1A007-00012"))

    def test_about_half_of_family_two_and_nothing_else(self):
        r = routers.get_router("kappaH")
        r.configure(cfg("kappaH"))
        f2 = [r.route(meta(i, 2, "B"), {}) == "cloud" for i in self.ids]
        share = sum(f2) / len(f2)
        self.assertGreater(share, 0.45)
        self.assertLess(share, 0.55)
        self.assertFalse(any(r.route(meta(i, 1, "A"), {}) == "cloud" for i in self.ids))
        self.assertFalse(any(r.route(meta(i, 3, "A"), {}) == "cloud" for i in self.ids))

    def test_class_does_not_matter(self):
        r = routers.get_router("kappaH")
        r.configure(cfg("kappaH"))
        for i in self.ids[:200]:
            self.assertEqual(r.route(meta(i, 2, "A"), {}), r.route(meta(i, 2, "C"), {}))

    def test_falls_back_to_req_when_req_id_is_missing(self):
        r = routers.get_router("kappaH")
        r.configure(cfg("kappaH"))
        for i in self.ids[:50]:
            self.assertEqual(r.route({"req": i, "family": 2}, {}), r.route(meta(i), {}))

    def test_known_values(self):
        import hashlib
        for rid in ("n1A007-00012", "n4C019-00009"):
            self.assertEqual(request_half(rid), hashlib.sha256(rid.encode()).digest()[0] & 1)


if __name__ == "__main__":
    unittest.main()
