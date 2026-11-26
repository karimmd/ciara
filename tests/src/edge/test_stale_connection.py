from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


try:
    import aiohttp
    from aiohttp import web
    from ciara_edge import gateway as gw
    import fake_engine
except ImportError:
    aiohttp = None

_REPO = str(Path(__file__).resolve().parents[3])
PILOT = os.environ.get("CIARA_PILOT_ROOT") or os.path.join(_REPO, "workload")
_DATA = os.environ.get("EVAL_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(PILOT)), "datasets")
PILOT_OK = os.path.isfile(os.path.join(PILOT, "eval", "suites", "iot.py")) and os.path.isfile(os.path.join(_DATA, "iot_toolcall", "test.jsonl"))
NEEDS_PILOT = unittest.skipUnless(PILOT_OK, "the tool-call suite is not built (build it with datasets/iot_toolcall/build/freeze_iot.py); the frozen suites are read from datasets/ through workload/eval/")


@unittest.skipIf(aiohttp is None, "aiohttp is not installed")
class Rule(unittest.TestCase):
    def test_edge_server_disconnected_is_retried_at_any_time(self):
        self.assertTrue(gw.stale_connection_error("ServerDisconnectedError: Server disconnected", 0.01, "edge"))
        self.assertTrue(gw.stale_connection_error("ServerDisconnectedError: Server disconnected", 30.0, "edge"))

    def test_cloud_server_disconnected_and_reset_by_peer_are_retried_within_the_limit(self):
        self.assertTrue(gw.stale_connection_error("ServerDisconnectedError: Server disconnected", 0.01, "cloud"))
        self.assertTrue(gw.stale_connection_error("ClientOSError: [Errno 104] Connection reset by peer", 0.01, "edge"))
        self.assertTrue(gw.stale_connection_error("ClientConnectionResetError: Connection lost", 0.5, "cloud"))

    def test_late_failures_and_other_errors_are_not_retried(self):
        self.assertFalse(gw.stale_connection_error("ClientOSError: [Errno 104] Connection reset by peer", gw.STALE_RETRY_S + 0.1, "edge"))
        self.assertFalse(gw.stale_connection_error("ServerDisconnectedError: Server disconnected", gw.STALE_RETRY_S + 0.1, "cloud"))
        self.assertFalse(gw.stale_connection_error("ClientConnectorError: Cannot connect to host", 0.01, "edge"))
        self.assertFalse(gw.stale_connection_error("TimeoutError", 0.01, "edge"))
        self.assertFalse(gw.stale_connection_error(None, 0.01, "edge"))


class A:
    def __init__(self):
        self.tokens, self.tok_ms, self.ttft_ms = 6, 2, 5
        self.sleep_s = self.wake_s = 0.0
        self.natural_short, self.natural_long = None, 200


@unittest.skipIf(aiohttp is None, "aiohttp is not installed")
@NEEDS_PILOT
class GatewayRetry(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._eval_before = {k for k in sys.modules if k == "eval" or k.startswith("eval.")}
        self.addCleanup(self._restore_eval_modules)
        self.tmp = tempfile.mkdtemp(prefix="stale-")
        self.drops = {"edge": 0, "cloud": 0}
        self.seen = {"edge": 0, "cloud": 0}
        self.runners = []
        for name, port in (("edge", 19401), ("cloud", 19402)):
            app = fake_engine.make_app(A())
            app.middlewares.append(self._dropper(name))
            r = web.AppRunner(app)
            await r.setup()
            await web.TCPSite(r, "127.0.0.1", port).start()
            self.runners.append(r)
            self.addAsyncCleanup(r.cleanup)
        args = gw.parse_args(["--node", "1", "--run-id", "stale", "--log-dir", self.tmp, "--bind", "127.0.0.1", "--data-port", "19400", "--ctl-port", "19410",
                              "--engine-url", "http://127.0.0.1:19401", "--cloud-url", "http://127.0.0.1:19402", "--cloud-model", "cloudm", "--pilot-root", PILOT])
        self.g = gw.Gateway(args)
        await self.g.start()

    def _restore_eval_modules(self):
        for k in [k for k in sys.modules if (k == "eval" or k.startswith("eval.")) and k not in self._eval_before]:
            del sys.modules[k]

    def _dropper(self, name):
        @web.middleware
        async def drop(request, handler):
            if request.path.endswith("/chat/completions"):
                self.seen[name] += 1
                if self.drops[name] > 0:
                    self.drops[name] -= 1
                    request.transport.close()
                    return web.Response(status=500)
            return await handler(request)
        return drop

    async def asyncTearDown(self):
        await self.g.stop()

    async def call(self, tier):
        rec = {}
        body = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": False, "max_tokens": 8}
        summary = await self.g.call_upstream(tier, body, {}, rec, None, buffer_only=True)
        return summary, rec

    async def test_edge_connection_closed_once_is_retried(self):
        self.drops["edge"] = 1
        summary, rec = await self.call("edge")
        self.assertEqual((summary["status"], summary["error"], rec.get("upstream_retry"), self.seen["edge"]), (200, None, 1, 2))
        self.assertEqual(self.g.counters["upstream_retry"], 1)

    async def test_cloud_connection_closed_once_is_retried(self):
        self.drops["cloud"] = 1
        summary, rec = await self.call("cloud")
        self.assertEqual((summary["status"], summary["error"], rec.get("upstream_retry"), self.seen["cloud"]), (200, None, 1, 2))

    async def test_a_second_failure_is_reported_not_retried_again(self):
        self.drops["cloud"] = 2
        summary, rec = await self.call("cloud")
        self.assertIsNone(summary["status"])
        self.assertTrue(summary["error"].startswith("ServerDisconnectedError"), summary["error"])
        self.assertEqual((rec.get("upstream_retry"), self.seen["cloud"]), (1, 2))

    async def test_no_failure_no_retry(self):
        summary, rec = await self.call("edge")
        self.assertEqual((summary["status"], rec.get("upstream_retry"), self.seen["edge"]), (200, None, 1))


if __name__ == "__main__":
    unittest.main()
