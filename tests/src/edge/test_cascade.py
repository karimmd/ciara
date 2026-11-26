from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


from ciara_edge import routers

try:
    import aiohttp
    from ciara_edge import gateway as gw
    import fake_engine
except ImportError:
    aiohttp = None


_REPO = str(Path(__file__).resolve().parents[3])
PILOT = os.environ.get("CIARA_PILOT_ROOT") or os.path.join(_REPO, "workload")
_DATA = os.environ.get("EVAL_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(PILOT)), "datasets")
PILOT_OK = os.path.isfile(os.path.join(PILOT, "eval", "suites", "iot.py")) and os.path.isfile(os.path.join(_DATA, "iot_toolcall", "test.jsonl"))
NEEDS_PILOT = unittest.skipUnless(PILOT_OK, "the tool-call suite is not built (build it with datasets/iot_toolcall/build/freeze_iot.py); the frozen suites are read from datasets/ through workload/eval/")


def meta(rid="n1A001-00001", family=2, cls="A", max_tokens=1024):
    return {"req_id": rid, "req": rid, "family": family, "class": cls, "max_tokens": max_tokens}


def reply(finish="stop", status=200, error=None):
    return {"text": "x", "finish_reason": finish, "status": status, "error": error, "usage": {"completion_tokens": 3}}


class RouterRule(unittest.TestCase):
    def router(self, **params):
        r = routers.get_router("kappaC", params)
        r.configure({"arm_id": "p|b8|c32|fifo", "escalate_family2": True, "router": "kappaC", "batch_cap": 8})
        return r

    def test_listed_and_default_budget(self):
        self.assertIn("kappaC", routers.BUILTIN)
        self.assertEqual(routers.get_router("kappaC").budget, 32)
        self.assertFalse(routers.get_router("kappaC").needs_after_edge)

    def test_bad_budget_rejected(self):
        for b in (0, -1, "32", 3.5, True):
            with self.assertRaises(ValueError):
                routers.get_router("kappaC", {"budget": b})

    def test_every_request_starts_at_the_edge(self):
        r = self.router()
        for fam in (1, 2, None):
            self.assertEqual(r.route(meta(family=fam), {}), "edge")

    def test_only_family_two_is_buffered_and_capped(self):
        r = self.router()
        self.assertTrue(r.wants_after_edge(meta(family=2)))
        self.assertFalse(r.wants_after_edge(meta(family=1)))
        self.assertFalse(r.wants_after_edge(meta(family=None)))
        self.assertEqual(r.edge_max_tokens(meta(family=2)), 32)
        self.assertIsNone(r.edge_max_tokens(meta(family=1)))
        self.assertEqual(self.router(budget=16).edge_max_tokens(meta(family=2)), 16)
        self.assertEqual(r.edge_max_tokens(meta(family=2, max_tokens=20)), 20)
        self.assertEqual(r.edge_max_tokens(meta(family=2, max_tokens=None)), 32)

    def test_decision_follows_the_finish_reason(self):
        r = self.router()
        self.assertEqual(r.after_edge(meta(), reply("stop"), {}), "accept")
        self.assertEqual(r.decision_info()["reason"], "complete")
        self.assertEqual(r.after_edge(meta(), reply("length"), {}), "escalate")
        self.assertEqual(r.decision_info(), {"budget": 32, "reason": "length"})
        self.assertEqual(r.after_edge(meta(), reply("stop", status=500), {}), "escalate")
        self.assertEqual(r.decision_info()["reason"], "edge_error")
        self.assertEqual(r.after_edge(meta(), reply("stop", status=None, error="ConnectionError"), {}), "escalate")

    def test_the_flag_of_the_configuration_object_does_not_change_the_rule(self):
        r = routers.get_router("kappaC", {"budget": 32})
        for flag in (True, False):
            r.configure({"escalate_family2": flag, "router": "kappaC"})
            self.assertEqual(r.route(meta(family=2), {}), "edge")
            self.assertTrue(r.wants_after_edge(meta(family=2)))
            self.assertEqual(r.edge_max_tokens(meta(family=2)), 32)

    def test_other_routers_keep_their_defaults(self):
        for n in ("kappa", "kappaA", "kappaH", "cascade-test"):
            r = routers.get_router(n)
            self.assertIsNone(r.edge_max_tokens(meta()))
        self.assertTrue(routers.get_router("cascade-test").wants_after_edge(meta(family=1)))
        self.assertFalse(routers.get_router("kappa").wants_after_edge(meta()))

    def test_signature_independent_of_block(self):
        self.assertEqual(routers.router_signature("kappaC", {"budget": 32}, 1), routers.router_signature("kappaC", {"budget": 32}, 9))
        self.assertNotEqual(routers.router_signature("kappaC", {"budget": 32}, 1), routers.router_signature("kappaC", {"budget": 16}, 1))


class A:
    def __init__(self, tokens=12, tok_ms=5, ttft_ms=10, natural_short=None, natural_long=200):
        self.tokens, self.tok_ms, self.ttft_ms = tokens, tok_ms, ttft_ms
        self.sleep_s = self.wake_s = 0.0
        self.natural_short, self.natural_long = natural_short, natural_long


@unittest.skipIf(aiohttp is None, "aiohttp is not installed")
@NEEDS_PILOT
class GatewayCascade(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp import web
        self._eval_before = {k for k in sys.modules if k == "eval" or k.startswith("eval.")}
        self.addCleanup(self._restore_eval_modules)
        self.tmp = tempfile.mkdtemp(prefix="cascade-")
        self.edge_app = fake_engine.make_app(A(natural_short=10, natural_long=200))
        self.cloud_app = fake_engine.make_app(A(tokens=7))
        self.runners = []
        for app, port in ((self.edge_app, 19301), (self.cloud_app, 19302)):
            r = web.AppRunner(app)
            await r.setup()
            await web.TCPSite(r, "127.0.0.1", port).start()
            self.runners.append(r)
            self.addAsyncCleanup(r.cleanup)
        args = gw.parse_args(["--node", "1", "--run-id", "cascade", "--log-dir", self.tmp, "--bind", "127.0.0.1", "--data-port", "19300", "--ctl-port", "19310",
                              "--engine-url", "http://127.0.0.1:19301", "--cloud-url", "http://127.0.0.1:19302", "--cloud-model", "cloudm", "--compact-classes", "A,B", "--pilot-root", PILOT])
        self.g = gw.Gateway(args)
        await self.g.start()
        self.s = aiohttp.ClientSession()
        self.ctl = "http://127.0.0.1:19310"
        self.data = "http://127.0.0.1:19300/v1/chat/completions"

    def _restore_eval_modules(self):
        for k in [k for k in sys.modules if (k == "eval" or k.startswith("eval.")) and k not in self._eval_before]:
            del sys.modules[k]

    async def asyncTearDown(self):
        await self.s.close()
        await self.g.stop()

    async def cfg(self, **kw):
        c = {"arm_id": "p|b4|c32|fifo", "profile": "p1", "batch_cap": 4, "escalate_family2": True, "router": "kappaC", "router_params": {"budget": 32}, "block_id": 1}
        c.update(kw)
        async with self.s.post(self.ctl + "/config", json=c) as r:
            self.assertEqual(r.status, 200, await r.text())

    async def req(self, rid, fam=2, long=False, stream=True, cls="C", mt=1024):
        h = {"X-Ciara-Req": rid, "X-Ciara-Device": "d1", "X-Ciara-Class": cls, "X-Ciara-Family": str(fam), "X-Ciara-Item": "it", "X-Ciara-Deadline-Ms": "50000"}
        body = {"model": "x", "messages": [{"role": "user", "content": "LONGREPLY" if long else "short"}], "stream": stream, "max_tokens": mt}
        async with self.s.post(self.data, json=body, headers=h) as r:
            return r.status, await r.text()

    def rec(self, rid):
        with open(self.g.log_path) as fh:
            return [json.loads(l) for l in fh if f'"req":"{rid}"' in l][-1]

    async def stats(self, port):
        async with self.s.get(f"http://127.0.0.1:{port}/stats") as r:
            return await r.json()

    async def test_short_reply_stays_at_the_edge(self):
        await self.cfg()
        st, tx = await self.req("s1")
        r = self.rec("s1")
        self.assertEqual(st, 200)
        self.assertEqual((r["served_by"], r["escalated_after_edge"], r["cascade_reason"], r["cascade_budget"]), ("edge", False, "complete", 32))
        self.assertEqual((r["finish_reason"], r["completion_tokens"], r["edge_attempt_tokens"]), ("stop", 10, 10))
        self.assertGreater(r["edge_attempt_ms"], 0)
        self.assertEqual((await self.stats(19301))["max_tokens_seen"][-1], 32)
        self.assertEqual((await self.stats(19302)).get("total", 0), 0)
        self.assertIn("p1:2:9", tx)

    async def test_long_reply_goes_to_the_cloud_with_the_original_cap(self):
        await self.cfg()
        st, tx = await self.req("l1", long=True)
        r = self.rec("l1")
        self.assertEqual(st, 200)
        self.assertEqual((r["served_by"], r["escalated_after_edge"], r["cascade_reason"]), ("cloud", True, "length"))
        self.assertEqual((r["edge_attempt_tokens"], r["edge_finish_reason"]), (32, "length"))
        self.assertEqual(r["routed"], "edge")
        cs = await self.stats(19302)
        self.assertEqual(cs["total"], 1)
        self.assertEqual(cs["max_tokens_seen"], [1024])
        self.assertIn("cloudm", tx)
        self.assertNotIn("p1:2:", tx)
        self.assertEqual(r["completion_tokens"], 7)
        self.assertGreaterEqual((r["t_done"] - r["t_arrive"]) / 1e6, r["edge_attempt_ms"])
        self.assertEqual(r["edge_attempt"]["usage"]["completion_tokens"], 32)

    async def test_family_one_is_unchanged(self):
        await self.cfg()
        st, tx = await self.req("f1", fam=1, long=True)
        r = self.rec("f1")
        self.assertEqual((st, r["served_by"], r["escalated_after_edge"]), (200, "edge", False))
        self.assertNotIn("cascade_budget", r)
        self.assertEqual((await self.stats(19301))["max_tokens_seen"][-1], 1024)
        self.assertEqual(r["completion_tokens"], 200)

    async def test_compact_class_and_nonstream(self):
        await self.cfg()
        for rid, cls, stream, long in (("a1", "A", True, True), ("b1", "B", False, True), ("c1", "C", False, True), ("a2", "A", True, False)):
            st, tx = await self.req(rid, cls=cls, stream=stream, long=long)
            r = self.rec(rid)
            self.assertEqual(st, 200, rid)
            self.assertEqual(r["escalated_after_edge"], long, rid)
            self.assertEqual(r["served_by"], "cloud" if long else "edge", rid)

    async def test_switch_between_blocks_without_restart(self):
        await self.cfg(router="kappaC")
        await self.req("w1", long=True)
        await self.cfg(router="kappa", escalate_family2=False, router_params={}, arm_id="p|b4|k0", block_id=2)
        st, _ = await self.req("w2", long=True)
        r = self.rec("w2")
        self.assertEqual((st, r["served_by"], r["escalated_after_edge"]), (200, "edge", False))
        self.assertNotIn("cascade_budget", r)
        await self.cfg(router="kappaC", block_id=3)
        await self.req("w3", long=True)
        self.assertTrue(self.rec("w3")["escalated_after_edge"])
        self.assertEqual(self.rec("w3")["block_id"], 3)

    async def test_admission_modes_combine(self):
        for mode in ("fifo", "f1prio"):
            await self.cfg(admission=mode, batch_cap=1)
            res = await asyncio.gather(*[self.req(f"{mode}{i}", fam=2 if i % 2 == 0 else 1, long=(i % 4 == 0)) for i in range(6)])
            self.assertTrue(all(s == 200 for s, _ in res))
            recs = [self.rec(f"{mode}{i}") for i in range(6)]
            self.assertTrue(all(r["admission"] == mode for r in recs))
            self.assertEqual([r["escalated_after_edge"] for r in recs], [True, False, False, False, True, False])

    async def test_flag_of_the_configuration_object_does_not_change_the_rule(self):
        await self.cfg(escalate_family2=False)
        await self.req("g1", long=True)
        r = self.rec("g1")
        self.assertEqual((r["routed"], r["served_by"], r["escalated_after_edge"], r["cascade_reason"]), ("edge", "cloud", True, "length"))

    async def test_edge_error_escalates(self):
        await self.cfg()
        self.g.engine_url = "http://127.0.0.1:19999"
        st, tx = await self.req("e1")
        r = self.rec("e1")
        self.assertEqual((st, r["served_by"], r["cascade_reason"]), (200, "cloud", "edge_error"))
        self.assertTrue(r["escalated_after_edge"])


if __name__ == "__main__":
    unittest.main()
