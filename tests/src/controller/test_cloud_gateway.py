import asyncio
import json
import os
import unittest

import aiohttp
from aiohttp import web

from ciara_control.cloud_gateway import CloudGateway
from ciara_control.gwcore import FifoAdmission, RecentWindow, SseSplitter, ciara_headers, parse_sse_event
from ciara_control.mocks.mock_vllm import MockVllm
from ciara_control.util import read_jsonl
from helpers import TmpDir

TOKEN = "run-token-123"


async def start(app: web.Application):
    runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def body(i=0, stream=True, usage=False, **extra):
    d = {"model": "m", "messages": [{"role": "user", "content": f"hello {i}"}], "stream": stream,
         "max_tokens": 512, "temperature": 0, "seed": 0, "mock_id": i,
         "chat_template_kwargs": {"enable_thinking": False}}
    if usage:
        d["stream_options"] = {"include_usage": True}
    d.update(extra)
    return d


def hdrs(i=0, node="E1", **extra):
    h = {"X-Ciara-Req": f"r{i}", "X-Ciara-Device": f"dev{i % 7}", "X-Ciara-Class": "B", "X-Ciara-Family": "1",
         "X-Ciara-Item": f"item{i}", "X-Ciara-Deadline-Ms": "5000", "X-Ciara-Node": node}
    h.update(extra)
    return h


class GatewayCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = TmpDir()
        self.dir = self.tmp.__enter__()
        self.engine = MockVllm(ttft_s=0.02, tok_s=0.002, tokens=12, token=TOKEN)
        self.eng_runner, self.eng_url = await start(self.engine.make_app())
        self.log_path = os.path.join(self.dir, "cloud-gw.jsonl")
        self.gw = CloudGateway(self.eng_url, self.log_path, token=TOKEN, max_in_flight=16)
        self.gw_runner, self.gw_url = await start(self.gw.make_app())
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))

    async def asyncTearDown(self):
        await self.http.close()
        await self.gw_runner.cleanup()
        await self.eng_runner.cleanup()
        self.tmp.__exit__()

    async def post(self, i=0, node="E1", read=True, **kw):
        h = kw.pop("headers", None) or hdrs(i, node)
        async with self.http.post(self.gw_url + "/v1/chat/completions", json=body(i, **kw), headers=h) as r:
            data = await r.read() if read else None
            return r.status, data

    def records(self):
        return read_jsonl(self.log_path)


class Streaming(GatewayCase):
    async def test_stream_passes_through_and_usage_event_hidden_when_not_requested(self):
        status, data = await self.post(1, usage=False)
        self.assertEqual(status, 200)
        events = SseSplitter().feed(data)
        kinds = [parse_sse_event(e) for e in events]
        self.assertEqual(kinds[-1][0], "done")
        self.assertFalse(any(k == "json" and o.get("usage") for k, o in kinds))
        text = "".join(o["choices"][0]["delta"].get("content", "") for k, o in kinds if k == "json")
        self.assertTrue(text.startswith("tok0 tok1"))
        self.assertTrue(self.engine.last_body["stream_options"]["include_usage"])

    async def test_usage_event_kept_when_client_requested_it(self):
        status, data = await self.post(2, usage=True)
        kinds = [parse_sse_event(e) for e in SseSplitter().feed(data)]
        self.assertTrue(any(k == "json" and o.get("usage") for k, o in kinds))

    async def test_log_record_fields(self):
        await self.post(3, node="E4")
        await asyncio.sleep(0.2)
        rec = self.records()[0]
        self.assertEqual((rec["req_id"], rec["device"], rec["class"], rec["family"], rec["item"], rec["deadline_ms"]),
                         ("r3", "dev3", "B", 1, "item3", 5000))
        self.assertEqual((rec["node"], rec["served_by"], rec["http_status"]), ("E4", "cloud", 200))
        self.assertIsNone(rec["error"])
        self.assertLessEqual(rec["t_arrive"], rec["t_admit"])
        self.assertLess(rec["t_admit"], rec["t_first_token"])
        self.assertLess(rec["t_first_token"], rec["t_done"])
        self.assertGreater(rec["wall_done_ns"], rec["wall_arrive_ns"])
        self.assertEqual(rec["completion_tokens"], 12)
        self.assertGreater(rec["prompt_tokens"], 0)
        self.assertTrue(rec["reply_text"].startswith("tok0 tok1 tok2"))
        for key in ("queue_len_at_arrival", "in_flight_at_arrival", "arm_id", "block_id", "escalated_after_edge",
                    "bytes_down", "component"):
            self.assertIn(key, rec)

    async def test_non_streaming_request(self):
        status, data = await self.post(4, stream=False)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["usage"]["completion_tokens"], 12)
        await asyncio.sleep(0.1)
        rec = self.records()[0]
        self.assertEqual((rec["completion_tokens"], rec["http_status"]), (12, 200))

    async def test_bearer_token_is_attached_upstream(self):
        status, _ = await self.post(5)
        self.assertEqual(status, 200)
        self.assertEqual(self.engine.auth_failures, 0)

    async def test_wrong_token_surfaces_as_upstream_401(self):
        self.gw.token = "wrong"
        status, _ = await self.post(6)
        self.assertEqual(status, 401)
        await asyncio.sleep(0.1)
        rec = self.records()[0]
        self.assertEqual((rec["http_status"], rec["error"]), (401, "upstream_http_401"))


class FifoAndBound(GatewayCase):
    async def test_at_most_16_in_flight_and_fifo_admission(self):
        n = 70
        tasks = []
        for i in range(n):
            tasks.append(asyncio.create_task(self.post(i, node=f"E{1 + i % 5}", read=True,
                                                       **{"mock_ttft_s": 0.15})))
            await asyncio.sleep(0.001)
        results = await asyncio.gather(*tasks)
        self.assertTrue(all(s == 200 for s, _ in results))
        self.assertEqual(self.engine.max_concurrent, 16)
        self.assertEqual(self.engine.total, n)
        await asyncio.sleep(0.2)
        recs = sorted(self.records(), key=lambda r: r["t_arrive"])
        self.assertEqual(len(recs), n)
        admits = [r["t_admit"] for r in recs]
        self.assertEqual(admits, sorted(admits), "admission order differs from arrival order")
        rank = {int(r["req_id"][1:]): k for k, r in enumerate(recs)}
        seen = [rank[i] for i in self.engine.arrival_order]
        self.assertLessEqual(max(abs(pos - r) for pos, r in enumerate(seen)), 16)
        self.assertGreater(max(r["queue_len_at_arrival"] for r in recs), 20)
        self.assertEqual({r["node"] for r in recs}, {f"E{k}" for k in range(1, 6)})

    async def test_limit_can_be_lowered_through_config_but_not_above_16(self):
        async with self.http.post(self.gw_url + "/config", json={"max_in_flight": 4}) as r:
            self.assertEqual(r.status, 200)
        await asyncio.gather(*(self.post(i) for i in range(20)))
        self.assertEqual(self.engine.max_concurrent, 4)
        async with self.http.post(self.gw_url + "/config", json={"max_in_flight": 17}) as r:
            self.assertEqual(r.status, 400)
        self.assertEqual(self.gw.adm.limit, 4)

    async def test_client_disconnect_while_queued_frees_the_slot_and_is_not_forwarded(self):
        self.gw.adm.set_limit(1)
        slow = asyncio.create_task(self.post(0, **{"mock_ttft_s": 0.6}))
        await asyncio.sleep(0.1)
        ghost_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.2))
        try:
            with self.assertRaises(asyncio.TimeoutError):
                async with ghost_session.post(self.gw_url + "/v1/chat/completions", json=body(1),
                                              headers=hdrs(1)) as r:
                    await r.read()
        finally:
            await ghost_session.close()
        third = asyncio.create_task(self.post(2))
        await asyncio.gather(slow, third)
        await asyncio.sleep(0.3)
        self.assertEqual(self.engine.total, 2)
        self.assertEqual(self.gw.adm.in_flight, 0)
        self.assertEqual(self.gw.adm.queue_len, 0)
        recs = {r["req_id"]: r for r in self.records()}
        self.assertEqual(recs["r1"]["http_status"], 499)


class Errors(GatewayCase):
    async def test_upstream_500_is_passed_back_and_logged(self):
        status, _ = await self.post(7, mock_fail="http500")
        self.assertEqual(status, 500)
        await asyncio.sleep(0.1)
        rec = self.records()[0]
        self.assertEqual((rec["http_status"], rec["error"]), (500, "upstream_http_500"))
        self.assertEqual(self.gw.adm.in_flight, 0)

    async def test_upstream_closing_mid_stream_is_logged_and_gateway_survives(self):
        try:
            await self.post(8, mock_fail="drop")
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(0.2)
        rec = self.records()[0]
        self.assertIsNotNone(rec["error"])
        self.assertEqual(self.gw.adm.in_flight, 0)
        status, _ = await self.post(9)
        self.assertEqual(status, 200)

    async def test_unreachable_upstream_returns_502(self):
        await self.eng_runner.cleanup()
        status, data = await self.post(10)
        self.assertEqual(status, 502)
        await asyncio.sleep(0.1)
        self.assertEqual(self.records()[0]["http_status"], 502)
        self.assertEqual(self.gw.adm.in_flight, 0)
        self.eng_runner = web.AppRunner(web.Application())
        await self.eng_runner.setup()


class ControlApi(GatewayCase):
    async def test_state_reports_queue_inflight_counters_and_recent_latency(self):
        self.gw.adm.set_limit(2)
        tasks = [asyncio.create_task(self.post(i, **{"mock_ttft_s": 0.3})) for i in range(6)]
        await asyncio.sleep(0.1)
        async with self.http.get(self.gw_url + "/state") as r:
            st = await r.json()
        self.assertEqual((st["in_flight"], st["queue_len"]), (2, 4))
        self.assertEqual(st["counters"]["arrived"], 6)
        await asyncio.gather(*tasks)
        async with self.http.get(self.gw_url + "/state") as r:
            st = await r.json()
        self.assertEqual(st["counters"]["completed"], 6)
        self.assertEqual(st["recent"]["n"], 6)
        self.assertGreater(st["recent"]["p95_ms"], 100)
        self.assertEqual(st["config"]["max_in_flight"], 2)

    async def test_state_is_readable_by_the_controller_parser(self):
        from ciara_control.telemetry import parse_gateway_state
        await self.post(1)
        async with self.http.get(self.gw_url + "/state") as r:
            st = parse_gateway_state(await r.json())
        self.assertEqual(st["counters"]["arrived"], 1)
        self.assertEqual(st["counters"]["completed"], 1)
        self.assertEqual(st["queue_len"], 0)
        self.assertIsNotNone(st["recent"])

    async def test_controller_push_labels_requests_with_arm_and_block(self):
        async with self.http.post(self.gw_url + "/config", json={"nodes": {"E2": {"arm_id": "A|b8|k0", "block_id": 5}}}) as r:
            self.assertEqual(r.status, 200)
        await self.post(1, node="E2")
        await self.post(2, node="E3")
        await self.post(3, node="E2", headers=hdrs(3, "E2", **{"X-Ciara-Arm": "OTHER", "X-Ciara-Block": "9"}))
        await asyncio.sleep(0.2)
        by = {r["req_id"]: r for r in self.records()}
        self.assertEqual((by["r1"]["arm_id"], by["r1"]["block_id"]), ("A|b8|k0", 5))
        self.assertEqual((by["r2"]["arm_id"], by["r2"]["block_id"]), (None, None))
        self.assertEqual((by["r3"]["arm_id"], by["r3"]["block_id"]), ("OTHER", 9))

    async def test_node_from_peer_address_when_header_is_absent(self):
        self.gw.node_map = {"127.0.0.1": "E5"}
        h = hdrs(1)
        del h["X-Ciara-Node"]
        await self.post(1, headers=h)
        await asyncio.sleep(0.1)
        self.assertEqual(self.records()[0]["node"], "E5")
        self.gw.node_map = {}
        await self.post(2, headers=h)
        await asyncio.sleep(0.1)
        self.assertEqual(self.records()[1]["node"], "unknown")

    async def test_escalated_header_is_logged(self):
        await self.post(1, headers=hdrs(1, **{"X-Ciara-Escalated": "1"}))
        await asyncio.sleep(0.1)
        self.assertTrue(self.records()[0]["escalated_after_edge"])


class CoreUnits(unittest.IsolatedAsyncioTestCase):
    async def test_fifo_order_and_release(self):
        adm = FifoAdmission(2)
        order = []

        async def worker(i):
            await adm.acquire()
            order.append(i)
            await asyncio.sleep(0.01)
            adm.release()

        await asyncio.gather(*(worker(i) for i in range(10)))
        self.assertEqual(order, list(range(10)))
        self.assertEqual((adm.in_flight, adm.queue_len), (0, 0))

    async def test_cancellation_of_a_granted_waiter_does_not_leak_a_slot(self):
        adm = FifoAdmission(1)
        await adm.acquire()
        t = asyncio.create_task(adm.acquire())
        await asyncio.sleep(0)
        adm.release()
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        self.assertEqual(adm.in_flight, 0)
        await adm.acquire()
        self.assertEqual(adm.in_flight, 1)

    async def test_cancellation_while_waiting_removes_waiter(self):
        adm = FifoAdmission(1)
        await adm.acquire()
        t = asyncio.create_task(adm.acquire())
        await asyncio.sleep(0)
        self.assertEqual(adm.queue_len, 1)
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        self.assertEqual(adm.queue_len, 0)
        adm.release()
        self.assertEqual(adm.in_flight, 0)

    def test_sse_splitter_handles_split_chunks_and_crlf(self):
        sp = SseSplitter()
        out = sp.feed(b'data: {"a":1}\r\n\r\ndata: {"b"')
        out += sp.feed(b':2}\n\ndata: [DONE]\n\n')
        self.assertEqual([parse_sse_event(e) for e in out],
                         [("json", {"a": 1}), ("json", {"b": 2}), ("done", None)])

    def test_recent_window_quantiles(self):
        w = RecentWindow(10.0)
        for i in range(1, 101):
            w.add(i * 10**6, float(i), float(i) / 2)
        s = w.stats(100 * 10**6)
        self.assertEqual((s["n"], s["p50_ms"], s["p95_ms"]), (100, 50.0, 95.0))
        self.assertEqual(w.stats(10**12)["n"], 0)

    def test_header_normalization(self):
        h = ciara_headers({"x-ciara-req": "a", "X-Ciara-Family": "2", "X-Ciara-Deadline-Ms": "1500",
                           "X-Ciara-Origin": "E2", "Other": "z", "X-Ciara-Escalated": "true"})
        self.assertEqual(h, {"req_id": "a", "family": 2, "deadline_ms": 1500, "node": "E2",
                             "escalated_after_edge": True})


class StaleConnection(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = TmpDir()
        self.dir = self.tmp.__enter__()
        self.drops = 0
        self.seen = 0
        engine = MockVllm(ttft_s=0.01, tok_s=0.001, tokens=6, token=TOKEN)
        app = engine.make_app()

        @web.middleware
        async def drop(request, handler):
            if request.path.endswith("/chat/completions"):
                self.seen += 1
                if self.drops > 0:
                    self.drops -= 1
                    request.transport.close()
                    return web.Response(status=500)
            return await handler(request)

        app.middlewares.append(drop)
        self.eng_runner, self.eng_url = await start(app)
        self.gw = CloudGateway(self.eng_url, os.path.join(self.dir, "cloud-gw.jsonl"), token=TOKEN, max_in_flight=16)
        self.gw_runner, self.gw_url = await start(self.gw.make_app())
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))

    async def asyncTearDown(self):
        await self.http.close()
        await self.gw_runner.cleanup()
        await self.eng_runner.cleanup()
        self.tmp.__exit__()

    async def post(self, stream):
        async with self.http.post(self.gw_url + "/v1/chat/completions", json=body(1, stream=stream), headers=hdrs(1)) as r:
            return r.status, await r.read()

    def last(self):
        return read_jsonl(os.path.join(self.dir, "cloud-gw.jsonl"))[-1]

    async def test_closed_connection_is_retried_non_streaming(self):
        self.drops = 1
        status, _ = await self.post(False)
        await asyncio.sleep(0.1)
        r = self.last()
        self.assertEqual((status, r["http_status"], r["error"], r.get("upstream_retry"), self.seen), (200, 200, None, 1, 2))
        self.assertEqual(self.gw.counters["upstream_retry"], 1)

    async def test_closed_connection_is_retried_streaming(self):
        self.drops = 1
        status, data = await self.post(True)
        await asyncio.sleep(0.1)
        r = self.last()
        self.assertEqual((status, r["http_status"], r["error"], r.get("upstream_retry"), self.seen), (200, 200, None, 1, 2))
        self.assertIn(b"data:", data)

    async def test_two_failures_answer_502_after_one_retry(self):
        self.drops = 2
        status, _ = await self.post(False)
        await asyncio.sleep(0.1)
        r = self.last()
        self.assertEqual((status, r["http_status"], r.get("upstream_retry"), self.seen), (502, 502, 1, 2))
        self.assertTrue(r["error"].startswith("upstream_error: ServerDisconnectedError"), r["error"])

    async def test_no_failure_no_retry(self):
        status, _ = await self.post(False)
        await asyncio.sleep(0.1)
        self.assertEqual((status, self.last().get("upstream_retry"), self.seen, self.gw.counters["upstream_retry"]), (200, None, 1, 0))


if __name__ == "__main__":
    unittest.main()
