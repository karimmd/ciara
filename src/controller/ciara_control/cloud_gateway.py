from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Optional

import aiohttp
from aiohttp import web

from .gwcore import (STALE_RETRY_S, FifoAdmission, JsonlWriter, RecentWindow, SseSplitter, ciara_headers,
                     parse_sse_event, stale_connection_error)

HARD_MAX_IN_FLIGHT = 16


class CloudGateway:
    def __init__(self, upstream: str, log_path: str, token: Optional[str] = None,
                 max_in_flight: int = HARD_MAX_IN_FLIGHT, node_map: Optional[dict] = None,
                 request_timeout_s: float = 900.0, recent_window_s: float = 60.0):
        self.upstream = upstream.rstrip("/")
        self.url = self.upstream + "/v1/chat/completions"
        self.token = token
        self.adm = FifoAdmission(min(max_in_flight, HARD_MAX_IN_FLIGHT))
        self.node_map = dict(node_map or {})
        self.timeout_s = request_timeout_s
        self.log = JsonlWriter(log_path)
        self.recent = RecentWindow(recent_window_s)
        self.pushed: dict[str, dict] = {}
        self.counters = {"arrived": 0, "admitted": 0, "completed": 0, "errors": 0, "upstream_retry": 0}
        self.by_node: dict[str, dict] = {}
        self.session: Optional[aiohttp.ClientSession] = None


    def make_app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/chat/completions", self.handle_chat)
        app.router.add_get("/state", self.handle_state)
        app.router.add_post("/config", self.handle_config)
        app.router.add_get("/healthz", self.handle_health)
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _startup(self, app: web.Application) -> None:
        conn = aiohttp.TCPConnector(limit=0, keepalive_timeout=30)
        self.session = aiohttp.ClientSession(
            connector=conn, timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=self.timeout_s))

    async def _cleanup(self, app: web.Application) -> None:
        if self.session is not None:
            await self.session.close()
        self.log.close()


    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "component": "cloud-gw"})

    def _config_in_force(self) -> dict:
        return {"max_in_flight": self.adm.limit, "upstream": self.upstream, "nodes": self.pushed}

    async def handle_state(self, request: web.Request) -> web.Response:
        now = time.monotonic_ns()
        return web.json_response({
            "component": "cloud-gw", "queue_len": self.adm.queue_len, "in_flight": self.adm.in_flight,
            "config": self._config_in_force(),
            "counters": {**self.counters, "by_node": self.by_node},
            "recent": self.recent.stats(now), "t_mono_ns": now, "wall_ns": time.time_ns(),
        })

    async def handle_config(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "body is not JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be an object"}, status=400)
        if "max_in_flight" in body:
            n = body["max_in_flight"]
            if isinstance(n, bool) or not isinstance(n, int) or not (1 <= n <= HARD_MAX_IN_FLIGHT):
                return web.json_response({"error": f"max_in_flight must be an integer in 1..{HARD_MAX_IN_FLIGHT}"},
                                         status=400)
            self.adm.set_limit(n)
        nodes = body.get("nodes")
        if isinstance(nodes, dict):
            for nid, info in nodes.items():
                if isinstance(info, dict):
                    self.pushed[str(nid)] = {k: info.get(k) for k in ("arm_id", "block_id", "epoch")}
        return web.json_response({"ok": True, "config": self._config_in_force()})


    def _node_of(self, request: web.Request, h: dict) -> str:
        if h.get("node"):
            node = str(h["node"])
            return f"E{node}" if node.isdigit() else node
        peer = request.remote
        return self.node_map.get(peer or "", "unknown")

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        t_arrive = time.monotonic_ns()
        wall_arrive = time.time_ns()
        raw = await request.read()
        h = ciara_headers(request.headers)
        node = self._node_of(request, h)
        pushed = self.pushed.get(node, {})
        rec: dict = {
            "rec": "request", "component": "cloud-gw", "req_id": h.get("req_id"), "device": h.get("device"),
            "class": h.get("class"), "family": h.get("family"), "item": h.get("item"),
            "deadline_ms": h.get("deadline_ms"), "node": node,
            "arm_id": h.get("arm_id", pushed.get("arm_id")),
            "block_id": h.get("block_id", pushed.get("block_id")),
            "t_arrive": t_arrive, "wall_arrive_ns": wall_arrive, "t_admit": None, "t_first_token": None,
            "t_done": None, "wall_done_ns": None, "served_by": "cloud",
            "escalated_after_edge": h.get("escalated_after_edge"),
            "queue_len_at_arrival": self.adm.queue_len, "in_flight_at_arrival": self.adm.in_flight,
            "prompt_tokens": None, "completion_tokens": None, "http_status": None, "bytes_down": 0,
            "error": None, "reply_text": None,
        }
        nstat = self.by_node.setdefault(node, {"arrived": 0, "completed": 0, "errors": 0})
        self.counters["arrived"] += 1
        nstat["arrived"] += 1
        parts: list[str] = []
        admitted = False
        try:
            payload = None
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            stream = isinstance(payload, dict) and bool(payload.get("stream"))
            inject = False
            body = raw
            if stream:
                so = payload.get("stream_options") or {}
                if not so.get("include_usage"):
                    payload["stream_options"] = {**so, "include_usage": True}
                    inject = True
                    body = json.dumps(payload).encode("utf-8")
            await self.adm.acquire()
            admitted = True
            rec["t_admit"] = time.monotonic_ns()
            self.counters["admitted"] += 1
            if request.transport is None or request.transport.is_closing():
                rec["http_status"], rec["error"] = 499, "client_disconnected_in_queue"
                return web.Response(status=499)
            resp = await self._forward(request, body, stream, inject, rec, parts)
            return resp
        except asyncio.CancelledError:
            rec["http_status"] = rec["http_status"] or 499
            rec["error"] = rec["error"] or "client_disconnected"
            raise
        except Exception as exc:
            rec["error"] = rec["error"] or f"{type(exc).__name__}: {exc}"
            if rec["http_status"] is None:
                rec["http_status"] = 502
            return web.json_response({"error": rec["error"]}, status=rec["http_status"])
        finally:
            if admitted:
                self.adm.release()
            rec["t_done"] = time.monotonic_ns()
            rec["wall_done_ns"] = time.time_ns()
            rec["reply_text"] = "".join(parts) if parts else rec["reply_text"]
            ok = rec["error"] is None and rec["http_status"] == 200
            if ok:
                self.counters["completed"] += 1
                nstat["completed"] += 1
                ttft = (rec["t_first_token"] - t_arrive) / 1e6 if rec["t_first_token"] else None
                self.recent.add(rec["t_done"], (rec["t_done"] - t_arrive) / 1e6, ttft)
            else:
                self.counters["errors"] += 1
                nstat["errors"] += 1
            self.log.write(rec)

    async def _forward(self, request: web.Request, body: bytes, stream: bool, inject: bool,
                       rec: dict, parts: list) -> web.StreamResponse:
        t0 = time.monotonic()
        resp = await self._forward_once(request, body, stream, inject, rec, parts)
        if stale_connection_error(rec.get("error"), rec.get("http_status")) and not rec["bytes_down"] and rec["t_first_token"] is None and time.monotonic() - t0 < STALE_RETRY_S:
            rec["upstream_retry"] = 1
            self.counters["upstream_retry"] += 1
            rec["error"], rec["http_status"] = None, None
            resp = await self._forward_once(request, body, stream, inject, rec, parts)
        return resp

    async def _forward_once(self, request: web.Request, body: bytes, stream: bool, inject: bool,
                            rec: dict, parts: list) -> web.StreamResponse:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        assert self.session is not None
        try:
            async with self.session.post(self.url, data=body, headers=headers) as up:
                rec["http_status"] = up.status
                if not stream or up.status != 200:
                    data = await up.read()
                    rec["bytes_down"] = len(data)
                    self._read_plain_reply(data, rec, parts)
                    if up.status != 200:
                        rec["error"] = f"upstream_http_{up.status}"
                    return web.Response(status=up.status, body=data,
                                        content_type=up.content_type or "application/json")
                resp = web.StreamResponse(status=200, headers={
                    "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                await resp.prepare(request)
                split = SseSplitter()
                try:
                    async for chunk in up.content.iter_any():
                        for ev in split.feed(chunk):
                            out = self._on_event(ev, rec, parts, inject)
                            if out is not None:
                                rec["bytes_down"] += len(out)
                                await resp.write(out)
                    for ev in split.flush():
                        out = self._on_event(ev, rec, parts, inject)
                        if out is not None:
                            rec["bytes_down"] += len(out)
                            await resp.write(out)
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    rec["error"] = f"upstream_stream_error: {type(exc).__name__}: {exc}"
                    if request.transport is not None:
                        request.transport.close()
                    return resp
                except ConnectionResetError:
                    rec["error"] = "client_disconnected"
                    rec["http_status"] = 499
                    return resp
                await resp.write_eof()
                return resp
        except aiohttp.ClientConnectorError as exc:
            rec["error"], rec["http_status"] = f"upstream_unreachable: {exc}", 502
            return web.json_response({"error": rec["error"]}, status=502)
        except asyncio.TimeoutError:
            rec["error"], rec["http_status"] = "upstream_timeout", 504
            return web.json_response({"error": rec["error"]}, status=504)
        except aiohttp.ClientError as exc:
            rec["error"], rec["http_status"] = f"upstream_error: {type(exc).__name__}: {exc}", 502
            return web.json_response({"error": rec["error"]}, status=502)

    def _read_plain_reply(self, data: bytes, rec: dict, parts: list) -> None:
        try:
            obj = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(obj, dict):
            return
        usage = obj.get("usage")
        if isinstance(usage, dict):
            rec["prompt_tokens"] = usage.get("prompt_tokens")
            rec["completion_tokens"] = usage.get("completion_tokens")
        ch = obj.get("choices")
        if isinstance(ch, list) and ch and isinstance(ch[0], dict):
            msg = ch[0].get("message") or {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                parts.append(msg["content"])

    def _on_event(self, ev: bytes, rec: dict, parts: list, inject: bool) -> Optional[bytes]:
        kind, obj = parse_sse_event(ev)
        if kind == "json" and isinstance(obj, dict):
            choices = obj.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta") or {}
                text = delta.get("content") if isinstance(delta, dict) else None
                reasoning = delta.get("reasoning_content") if isinstance(delta, dict) else None
                if (text or reasoning) and rec["t_first_token"] is None:
                    rec["t_first_token"] = time.monotonic_ns()
                if text:
                    parts.append(text)
            usage = obj.get("usage")
            if isinstance(usage, dict):
                rec["prompt_tokens"] = usage.get("prompt_tokens")
                rec["completion_tokens"] = usage.get("completion_tokens")
                if inject and not choices:
                    return None
        return ev + b"\n\n"


def _parse_node_map(spec: str) -> dict:
    out: dict[str, str] = {}
    for part in filter(None, (s.strip() for s in spec.split(","))):
        node, ip = part.split("=", 1)
        out[ip.strip()] = node.strip()
    return out


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="ciara_control.cloud_gateway", description="CIARA cloud gateway")
    ap.add_argument("--listen", default="0.0.0.0:8001", help="host:port for data and control API")
    ap.add_argument("--upstream", required=True, help="cloud engine base URL, e.g. http://CLOUD_HOST:8100")
    ap.add_argument("--log", required=True, help="cloud-gw.jsonl path")
    ap.add_argument("--max-in-flight", type=int, default=HARD_MAX_IN_FLIGHT)
    ap.add_argument("--token-env", default="CIARA_TOKEN", help="environment variable holding the engine token")
    ap.add_argument("--node-map", default="", help="E1=ip,E2=ip fallback for requests without X-Ciara-Node")
    ap.add_argument("--request-timeout-s", type=float, default=900.0)
    args = ap.parse_args(argv)
    host, port = args.listen.rsplit(":", 1)
    gw = CloudGateway(args.upstream, args.log, token=os.environ.get(args.token_env) or None,
                      max_in_flight=args.max_in_flight, node_map=_parse_node_map(args.node_map),
                      request_timeout_s=args.request_timeout_s)
    web.run_app(gw.make_app(), host=host, port=int(port), access_log=None, handler_cancellation=True,
                print=lambda *_: None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
