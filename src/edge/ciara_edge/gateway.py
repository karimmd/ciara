from __future__ import annotations

import argparse
import asyncio
import collections
import inspect
import json
import math
import os
import signal
import sys
import time
from typing import Any, Dict, Optional

import aiohttp
from aiohttp import web

try:
    from .compose import COMPOSE_PATH, ComposeError, Composer, ContextStore
    from .admission import MODES as ADMISSION_MODES, AdmissionQueue
    from .routers import RouterContext, get_router, router_signature
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ciara_edge.compose import COMPOSE_PATH, ComposeError, Composer, ContextStore
    from ciara_edge.admission import MODES as ADMISSION_MODES, AdmissionQueue
    from ciara_edge.routers import RouterContext, get_router, router_signature

HDR = {"req": "X-Ciara-Req", "device": "X-Ciara-Device", "class": "X-Ciara-Class", "family": "X-Ciara-Family",
       "item": "X-Ciara-Item", "deadline_ms": "X-Ciara-Deadline-Ms"}
DEFAULT_CONFIG = {"arm_id": None, "profile": None, "batch_cap": 16, "escalate_family2": False, "router": "kappa",
                  "block_id": None, "admission": "fifo"}
NON_CONFIG_KEYS = {"engine_url", "run_id"}


def now_ns() -> int:
    return time.monotonic_ns()


STALE_RETRY_S = 2.0
_STALE_RESET = ("ClientOSError: [Errno 104]", "ClientConnectionResetError")


def stale_connection_error(error: Optional[str], elapsed_s: float, tier: str = "edge") -> bool:
    err = error or ""
    if err.startswith("ServerDisconnectedError") and tier == "edge":
        return True
    return elapsed_s < STALE_RETRY_S and (err.startswith("ServerDisconnectedError") or err.startswith(_STALE_RESET))


class GatewayError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


class RecentWindow:
    def __init__(self, window_s: float = 60.0, maxlen: int = 20000):
        self.window_ns = int(window_s * 1e9)
        self._items: collections.deque = collections.deque(maxlen=maxlen)

    def add(self, t_done_ns: int, latency_ms: float, ttft_ms: Optional[float]) -> None:
        self._items.append((t_done_ns, latency_ms, ttft_ms))

    @staticmethod
    def _q(vals: list, q: float) -> Optional[float]:
        if not vals:
            return None
        return vals[max(0, min(len(vals) - 1, math.ceil(q * len(vals)) - 1))]

    def stats(self, now: int) -> dict:
        lo = now - self.window_ns
        lat = sorted(x[1] for x in self._items if x[0] >= lo)
        ttft = sorted(x[2] for x in self._items if x[0] >= lo and x[2] is not None)
        n = len(lat)
        return {"window_s": self.window_ns / 1e9, "n": n, "mean_ms": (sum(lat) / n) if n else None,
                "p50_ms": self._q(lat, 0.5), "p95_ms": self._q(lat, 0.95), "ttft_p95_ms": self._q(ttft, 0.95)}


class SSEParser:
    def __init__(self, top_k: int = 0):
        self.buf = b""
        self.text: list = []
        self.tools: Dict[int, Dict[str, str]] = {}
        self.usage: Optional[dict] = None
        self.finish: Optional[str] = None
        self.t_first_token: Optional[int] = None
        self.t_first_chunk: Optional[int] = None
        self.done = False
        self.events = 0
        self.top_k = top_k
        self.token_top: list = []
        self.clean: list = []

    def feed(self, data: bytes, t: int) -> None:
        self.buf += data
        while True:
            i = self.buf.find(b"\n")
            if i < 0:
                break
            line, self.buf = self.buf[:i].strip(), self.buf[i + 1:]
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                self.done = True
                if self.top_k:
                    self.clean.append(b"data: [DONE]\n\n")
                continue
            try:
                ev = json.loads(payload)
            except ValueError:
                continue
            self.events += 1
            if self.t_first_chunk is None:
                self.t_first_chunk = t
            if ev.get("usage"):
                self.usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or ""
                rpiece = d.get("reasoning_content") or d.get("reasoning") or ""
                tcs = d.get("tool_calls") or []
                if (piece or rpiece or tcs) and self.t_first_token is None:
                    self.t_first_token = t
                if piece:
                    self.text.append(piece)
                for tc in tcs:
                    slot = self.tools.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                if ch.get("finish_reason"):
                    self.finish = ch["finish_reason"]
                if self.top_k:
                    lp = ch.pop("logprobs", None)
                    if lp:
                        for tok in lp.get("content") or []:
                            tops = sorted((x["logprob"] for x in tok.get("top_logprobs") or []), reverse=True)
                            if not tops:
                                tops = [tok["logprob"]]
                            self.token_top.append(tops[: self.top_k])
            if self.top_k:
                self.clean.append(b"data: " + json.dumps(ev, separators=(",", ":")).encode() + b"\n\n")

    def result(self) -> dict:
        return {"text": "".join(self.text), "tool_calls": [self.tools[k] for k in sorted(self.tools)] or None,
                "finish_reason": self.finish, "usage": self.usage}


def strip_json_logprobs(raw: bytes, top_k: int) -> tuple:
    try:
        js = json.loads(raw)
    except ValueError:
        return raw, []
    token_top = []
    for ch in js.get("choices") or []:
        lp = ch.pop("logprobs", None)
        if lp and not token_top:
            for tok in lp.get("content") or []:
                tops = sorted((x["logprob"] for x in tok.get("top_logprobs") or []), reverse=True) or [tok["logprob"]]
                token_top.append(tops[:top_k])
    return json.dumps(js, separators=(",", ":")).encode(), token_top


class Gateway:
    def __init__(self, args):
        self.args = args
        self.node = args.node
        self.run_id = args.run_id
        self.config: Dict[str, Any] = dict(DEFAULT_CONFIG)
        self.config_version = 0
        self.router_ctx = RouterContext(args.profile_table or None, args.feedback_fn or None)
        self.router = get_router("kappa", {}, self.router_ctx)
        self.router.configure(self.config)
        self.router_sig = router_signature("kappa", {}, None)
        self.engine_url = args.engine_url
        self.engine_map: Dict[str, str] = json.loads(args.engine_map) if args.engine_map else {}
        self.cloud_url = args.cloud_url
        self.cloud_model = args.cloud_model
        self.token = os.environ.get("CIARA_TOKEN", "")
        self.cloud_token = os.environ.get("CLOUD_TOKEN", self.token)
        self.compact_classes = {c.strip().upper() for c in (args.compact_classes or "").split(",") if c.strip()}
        self.compactor = None
        if self.compact_classes:
            from .compact import Compactor
            self.compactor = Compactor(args.pilot_root or None)
        self.composer = None
        self.compose_verify = bool(args.compose_verify)
        self.store_info: Dict[str, Any] = {}
        if args.context_store:
            t0 = time.perf_counter()
            store = ContextStore.load(args.context_store)
            self.composer = Composer(store)
            self.store_info = {"path": args.context_store, "bytes": os.path.getsize(args.context_store),
                               "load_s": round(time.perf_counter() - t0, 4), **store.summary()}
        self.waiters = AdmissionQueue()
        self.inflight = 0
        self.paused = False
        self.pause_since: Optional[int] = None
        self.pause_total_ns = 0
        self.pauses = 0
        self._pause_gen = 0
        self.drain_event = asyncio.Event()
        self.drain_event.set()
        self.counters = collections.Counter({k: 0 for k in ("arrived", "admitted", "completed", "escalated", "errors",
                                                            "routed_edge", "routed_cloud", "served_edge", "served_cloud",
                                                            "escalated_after_edge", "failed", "client_disconnected", "compact_replies", "compact_fallback",
                                                            "upstream_retry", "admitted_ahead", "compose_requests", "compose_ok", "compose_mismatch",
                                                            "compose_error")})
        self.recent = RecentWindow(args.recent_window_s)
        self.max_inflight_seen = 0
        self.max_queue_seen = 0
        self.edge_fail_streak = 0
        self.cloud_state: Dict[str, Any] = {}
        self.cloud_poll_ok = True
        self.cloud_poll_supported = True
        self.cloud_rtt_s: Optional[float] = None
        self.started = now_ns()
        self.session: Optional[aiohttp.ClientSession] = None
        self.logf = self.evf = self.probef = None
        self.log_path = None
        self._tasks: list = []
        self.open_logs(self.run_id)

    def open_logs(self, run_id: str) -> None:
        for f in (self.logf, self.evf, self.probef):
            if f:
                f.close()
        d = os.path.join(self.args.log_dir, run_id)
        os.makedirs(d, exist_ok=True)
        self.log_path = os.path.join(d, f"edge-gw-{self.node}.jsonl")
        self.logf = open(self.log_path, "a", buffering=1)
        self.evf = open(os.path.join(d, f"edge-gw-{self.node}.events.jsonl"), "a", buffering=1)
        self.probef = open(os.path.join(d, f"edge-gw-{self.node}.probes.jsonl"), "a", buffering=1)
        self.run_id = run_id

    def event(self, kind: str, **kw) -> None:
        rec = {"event": kind, "node": self.node, "t": now_ns(), "t_wall": time.time_ns(), **kw}
        self.evf.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")

    def node_state(self) -> dict:
        st = {"queue_len": len(self.waiters), "inflight": self.inflight, "in_flight": self.inflight,
              "batch_cap": self.config["batch_cap"], "profile": self.config.get("profile"), "paused": self.paused,
              "switching": self.paused, "health_ok": self.edge_fail_streak < 3, "config": self.config,
              "counters": dict(self.counters), "now_ns": now_ns(), "engine_url": self.engine_url}
        if self.cloud_state:
            st["cloud_queue_len"] = self.cloud_state.get("queue_len", 0)
            st["cloud_inflight"] = self.cloud_state.get("in_flight", 0)
        if self.cloud_poll_supported and self.cloud_state:
            st["cloud_ok"] = self.cloud_poll_ok
        if self.cloud_rtt_s is not None:
            st["backhaul_rtt_s"] = self.cloud_rtt_s
        if self.args.backhaul_mbps:
            st["backhaul_mbps"] = self.args.backhaul_mbps
        return st

    def pump(self) -> None:
        cap = int(self.config["batch_cap"])
        while len(self.waiters) and not self.paused and self.inflight < cap:
            w = self.waiters.pop(self.config.get("admission") or "fifo")
            if w.fut.cancelled():
                continue
            self.inflight += 1
            self.drain_event.clear()
            self.max_inflight_seen = max(self.max_inflight_seen, self.inflight)
            if w.overtaken:
                self.counters["admitted_ahead"] += 1
            w.fut.set_result(now_ns())

    async def admit(self, family: Optional[int] = None, info: Optional[dict] = None) -> int:
        fut = asyncio.get_running_loop().create_future()
        w = self.waiters.push(fut, family)
        self.max_queue_seen = max(self.max_queue_seen, len(self.waiters))
        if info is not None:
            info["arrive_seq"] = w.arrive_seq
        self.pump()
        try:
            t = await fut
            if info is not None:
                info.update(admit_seq=w.admit_seq, admit_overtaken=w.overtaken, queue_len_admit=w.queue_len_admit, admission=w.mode)
            return t
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()
            else:
                self.waiters.remove(w)
            raise

    async def wait_engine(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while not self.engine_url and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

    def release(self) -> None:
        self.inflight -= 1
        if self.inflight == 0:
            self.drain_event.set()
        self.pump()

    def set_paused(self, paused: bool) -> None:
        if paused and not self.paused:
            self.paused = True
            self.pause_since = now_ns()
            self.pauses += 1
            self._pause_gen += 1
            self.event("pause", queue_len=len(self.waiters), inflight=self.inflight)
            if self.args.max_pause_s > 0:
                self._tasks.append(asyncio.create_task(self._pause_watchdog(self._pause_gen)))
        elif not paused and self.paused:
            self.paused = False
            dt = now_ns() - self.pause_since
            self.pause_total_ns += dt
            self.event("resume", paused_ns=dt, queue_len=len(self.waiters), inflight=self.inflight)
            self.pause_since = None
            self.pump()

    async def _pause_watchdog(self, gen: int) -> None:
        await asyncio.sleep(self.args.max_pause_s)
        if self.paused and self._pause_gen == gen:
            self.event("pause_timeout", max_pause_s=self.args.max_pause_s)
            self.set_paused(False)

    def apply_config(self, cfg: dict) -> dict:
        new = dict(self.config)
        for k, v in cfg.items():
            if k not in NON_CONFIG_KEYS:
                new[k] = v
        if not isinstance(new["batch_cap"], int) or isinstance(new["batch_cap"], bool) or new["batch_cap"] < 1:
            raise ValueError("batch_cap must be a positive integer")
        if not isinstance(new["escalate_family2"], bool):
            raise ValueError("escalate_family2 must be a boolean")
        if not isinstance(new["router"], str):
            raise ValueError("router must be a string")
        if new.get("admission") not in ADMISSION_MODES:
            raise ValueError(f"admission must be one of {ADMISSION_MODES}")
        if "router_params" in cfg:
            params = cfg["router_params"] or {}
        elif new["router"] == self.config["router"]:
            params = self.config.get("router_params") or {}
        else:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("router_params must be an object")
        new["router_params"] = params
        sig = router_signature(new["router"], params, new.get("block_id"))
        router = self.router
        try:
            if sig != self.router_sig:
                router = get_router(new["router"], params, self.router_ctx)
            router.configure(new)
        except (KeyError, TypeError) as e:
            raise ValueError(f"router {new['router']!r} rejected the configuration: {type(e).__name__}: {e}")
        if cfg.get("run_id") and cfg["run_id"] != self.run_id:
            self.open_logs(cfg["run_id"])
        if cfg.get("engine_url"):
            self.engine_url = cfg["engine_url"]
        elif new.get("profile") and new["profile"] in self.engine_map:
            self.engine_url = self.engine_map[new["profile"]]
        self.config = new
        self.config_version += 1
        self.router, self.router_sig = router, sig
        self.event("config", config=new, engine_url=self.engine_url, version=self.config_version)
        self.pump()
        return new

    async def cloud_poller(self) -> None:
        fails = 0
        while True:
            await asyncio.sleep(self.args.cloud_poll_s)
            if not self.cloud_url or not self.cloud_poll_supported:
                continue
            t0 = now_ns()
            try:
                async with self.session.get(self.cloud_url.rstrip("/") + "/state",
                                            timeout=aiohttp.ClientTimeout(total=3)) as r:
                    if r.status == 404:
                        self.cloud_poll_supported = False
                        continue
                    js = await r.json()
                dt = (now_ns() - t0) / 1e9
                self.cloud_rtt_s = dt if self.cloud_rtt_s is None else 0.7 * self.cloud_rtt_s + 0.3 * dt
                self.cloud_state = {"queue_len": js.get("queue_len", 0), "in_flight": js.get("in_flight", js.get("inflight", 0))}
                fails, self.cloud_poll_ok = 0, True
            except Exception:
                fails += 1
                if fails >= 3 and self.cloud_state:
                    self.cloud_poll_ok = False

    def upstream_headers(self, hdrs, tier: str, extra: Optional[dict] = None) -> dict:
        out = {"Content-Type": "application/json", "X-Ciara-Node": str(self.node)}
        for h in HDR.values():
            if h in hdrs:
                out[h] = hdrs[h]
        if extra:
            out.update(extra)
        tok = self.token if tier == "edge" else self.cloud_token
        if tok:
            out["Authorization"] = f"Bearer {tok}"
        return out

    async def call_upstream(self, tier: str, body: dict, hdrs, rec: dict, relay, buffer_only: bool = False,
                            extra_fields: Optional[dict] = None, collect_top: int = 0) -> dict:
        t0 = time.monotonic()
        summary = await self.call_upstream_once(tier, body, hdrs, rec, relay, buffer_only, extra_fields, collect_top)
        if summary["status"] is None and rec.get("t_first_byte_client") is None and stale_connection_error(summary["error"], time.monotonic() - t0, tier):
            rec["upstream_retry"] = 1
            self.counters["upstream_retry"] += 1
            summary = await self.call_upstream_once(tier, body, hdrs, rec, relay, buffer_only, extra_fields, collect_top)
        return summary

    async def call_upstream_once(self, tier: str, body: dict, hdrs, rec: dict, relay, buffer_only: bool = False,
                                 extra_fields: Optional[dict] = None, collect_top: int = 0) -> dict:
        url = (self.engine_url if tier == "edge" else self.cloud_url)
        body = dict(body)
        body["model"] = (self.config.get("profile") if tier == "edge" else self.cloud_model) or body.get("model")
        stream = bool(body.get("stream"))
        if stream and self.args.inject_usage:
            so = dict(body.get("stream_options") or {})
            so.setdefault("include_usage", True)
            body["stream_options"] = so
        if extra_fields:
            body.update(extra_fields)
        data = json.dumps(body).encode()
        rec["bytes_up_" + tier] = len(data)
        rec["upstream_" + tier] = url
        parser = SSEParser(collect_top)
        summary = {"status": None, "text": "", "tool_calls": None, "finish_reason": None, "usage": None,
                   "t_first_token": None, "t_done": None, "error": None, "raw": b"", "headers": None, "token_top": []}
        chunks: list = []
        try:
            async with self.session.post(url.rstrip("/") + "/v1/chat/completions", data=data,
                                         headers=self.upstream_headers(hdrs, tier)) as resp:
                summary["status"] = resp.status
                ctype = resp.headers.get("Content-Type", "")
                summary["headers"] = {"Content-Type": ctype}
                if resp.status == 200 and stream and "text/event-stream" in ctype:
                    async for chunk in resp.content.iter_any():
                        t = now_ns()
                        parser.feed(chunk, t)
                        if buffer_only:
                            if not collect_top:
                                chunks.append((t, chunk))
                        else:
                            if relay is not None and rec.get("t_first_byte_client") is None:
                                rec["t_first_byte_client"] = t
                            await relay.write(chunk)
                    summary.update(parser.result())
                    summary["t_first_token"] = parser.t_first_token or parser.t_first_chunk
                    summary["token_top"] = parser.token_top
                    summary["chunks"] = [(0, c) for c in parser.clean] if collect_top else chunks
                else:
                    raw = await resp.read()
                    t = now_ns()
                    if collect_top and resp.status == 200:
                        raw, summary["token_top"] = strip_json_logprobs(raw, collect_top)
                    summary["raw"] = raw
                    summary["t_first_token"] = t
                    if resp.status == 200:
                        try:
                            js = json.loads(raw)
                            ch = (js.get("choices") or [{}])[0]
                            msg = ch.get("message") or {}
                            summary["text"] = msg.get("content") or ""
                            summary["tool_calls"] = msg.get("tool_calls")
                            summary["finish_reason"] = ch.get("finish_reason")
                            summary["usage"] = js.get("usage")
                        except ValueError:
                            pass
                summary["t_done"] = now_ns()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            summary["error"] = f"{type(e).__name__}: {e}"
            summary["t_done"] = now_ns()
        if tier == "edge":
            bad = summary["error"] is not None or (summary["status"] or 0) >= 500
            self.edge_fail_streak = self.edge_fail_streak + 1 if bad else 0
        return summary

    async def post_json(self, tier: str, body: dict, hdrs, extra_headers: dict, timeout: float = 600.0):
        url = (self.engine_url if tier == "edge" else self.cloud_url).rstrip("/") + "/v1/chat/completions"
        try:
            async with self.session.post(url, data=json.dumps(body).encode(), headers=self.upstream_headers(hdrs, tier, extra_headers),
                                         timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return r.status, await r.json(content_type=None), None
        except Exception as e:
            return None, None, f"{type(e).__name__}: {e}"

    async def call_hook(self, rec: dict, name: str, fn, *args, default=None):
        try:
            r = fn(*args)
            if inspect.isawaitable(r):
                r = await r
            return r
        except Exception as e:
            rec["router_error"] = f"{name}: {type(e).__name__}: {e}"
            self.counters["router_errors"] += 1
            return default

    def call_hook_sync(self, rec: dict, name: str, fn, *args, default=None):
        try:
            return fn(*args)
        except Exception as e:
            rec["router_error"] = f"{name}: {type(e).__name__}: {e}"
            self.counters["router_errors"] += 1
            return default

    async def run_probe(self, router, scorer, meta: dict, body: dict, hdrs, rec: dict) -> None:
        from ciara_baselines.race import build_probe_request, probe_log_record
        if scorer.mode == "prefill":
            return await self.run_prefill_probe(scorer, meta, body, hdrs, rec, probe_log_record)
        probe_meta = dict(meta, model=self.config.get("profile") or meta.get("model"), max_tokens=meta.get("max_tokens") or 512)
        pr = build_probe_request(probe_meta, router.race_params)
        pbody = pr["body"]
        pbody["model"] = self.config.get("profile") or pbody.get("model")
        t0 = now_ns()
        await self.admit(meta.get("family"))
        try:
            status, js, err = await self.post_json("edge", pbody, hdrs, pr["headers"])
        finally:
            self.release()
        t1 = now_ns()
        self.counters["probes"] += 1
        samples = [(c.get("message") or {}).get("content") or "" for c in (js or {}).get("choices", [])] if status == 200 else []
        if not samples:
            rec["router_error"] = f"probe failed: status {status} {err or ''}"
            return
        u = await self.call_hook(rec, "score", scorer.score, meta, samples, default=None)
        usage = dict((js or {}).get("usage") or {}, n=len(samples))
        self.probef.write(json.dumps(probe_log_record(meta, usage, t0, t1, u if u is not None else float("nan")),
                                     separators=(",", ":")) + "\n")
        rec["race_probe_s"] = (t1 - t0) / 1e9
        if u is not None:
            meta["race_u"] = u
            rec["score_u"] = u

    async def run_prefill_probe(self, scorer, meta: dict, body: dict, hdrs, rec: dict, probe_log_record) -> None:
        pbody = {k: v for k, v in body.items() if k not in ("stream_options", "tools_stream")}
        pbody.update(scorer.request_fields())
        pbody["model"] = self.config.get("profile") or meta.get("model")
        rid = str(meta["req_id"])
        t0 = now_ns()
        await self.admit(meta.get("family"))
        t_adm = now_ns()
        try:
            status, js, err = await self.post_json("edge", pbody, hdrs, {"X-Ciara-Req": f"{rid}.probe", "X-Ciara-Kind": "race_probe",
                                                                         "X-Ciara-Parent": rid})
        finally:
            self.release()
        t1 = now_ns()
        self.counters["probes"] += 1
        usage = (js or {}).get("usage") or {}
        rec["race_probe_s"] = (t1 - t0) / 1e9
        rec["race_probe_service_s"] = (t1 - t_adm) / 1e9
        rec["race_probe_prompt_tokens"] = usage.get("prompt_tokens")
        rec["race_probe_completion_tokens"] = usage.get("completion_tokens")
        try:
            lp = js["choices"][0]["logprobs"]["content"][0]["logprob"]
        except (TypeError, KeyError, IndexError):
            rec["router_error"] = f"prefill probe failed: status {status} {err or ''}"
            return
        u = await self.call_hook(rec, "score", scorer.score, meta, lp, default=None)
        self.probef.write(json.dumps(probe_log_record(meta, dict(usage, n=1), t0, t1, u if u is not None else float("nan")),
                                     separators=(",", ":")) + "\n")
        if u is not None:
            meta["race_u"] = u
            rec["score_u"] = u

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        return await self.handle_request(request, compose=False)

    async def handle_compose(self, request: web.Request) -> web.StreamResponse:
        return await self.handle_request(request, compose=True)

    def expand_compact(self, payload: dict, hdrs) -> tuple:
        self.counters["compose_requests"] += 1
        if self.composer is None:
            self.counters["compose_error"] += 1
            return {}, {"compose_ok": False}, (501, "no_context_store: the gateway was started without --context-store")
        fam = hdrs.get(HDR["family"])
        if fam is not None and str(payload.get("f")) != fam:
            self.counters["compose_error"] += 1
            return {}, {"compose_ok": False}, (422, f"bad_family: header {fam!r} against body {payload.get('f')!r}")
        try:
            body, info = self.composer.expand(payload, hdrs.get(HDR["item"]), self.compose_verify)
        except ComposeError as e:
            self.counters["compose_error"] += 1
            return {}, {"compose_ok": False}, (422, str(e))
        if info["compose_ok"] is False:
            self.counters["compose_mismatch"] += 1
            return {}, info, (422, "compose_mismatch: the composed messages differ from the frozen messages of the item"
                              if info.get("expected") else "compose_unverifiable: no expected hash for the item")
        if info["compose_ok"]:
            self.counters["compose_ok"] += 1
        return body, info, None

    async def handle_request(self, request: web.Request, compose: bool) -> web.StreamResponse:
        t_arrive = now_ns()
        t_wall = time.time_ns()
        h = request.headers
        raw = await request.read()
        t_body = now_ns()
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return web.json_response({"error": "request body must be a JSON object"}, status=400)
        compose_err = None
        compose_info: dict = {}
        if compose:
            body, compose_info, compose_err = self.expand_compact(body, h)
        fam = h.get(HDR["family"])
        dl = h.get(HDR["deadline_ms"])
        meta = {"req": h.get(HDR["req"]), "req_id": h.get(HDR["req"]), "device": h.get(HDR["device"]),
                "class": h.get(HDR["class"]), "family": int(fam) if fam and fam.isdigit() else None,
                "item": h.get(HDR["item"]), "deadline_ms": int(dl) if dl and dl.isdigit() else dl, "node": self.node,
                "t_arrive": t_arrive, "model": body.get("model"), "max_tokens": body.get("max_tokens"),
                "messages": body.get("messages"), "stream": bool(body.get("stream"))}
        rec = {"req": meta["req"], "device": meta["device"], "class": meta["class"], "family": meta["family"],
               "item": meta["item"], "deadline_ms": meta["deadline_ms"], "node": self.node,
               "t_arrive": t_arrive, "t_arrive_wall": t_wall, "t_admit": None, "t_first_token": None, "t_done": None,
               "t_done_wall": None, "t_first_byte_client": None, "served_by": None, "routed": None,
               "escalated_after_edge": False, "arm_id": self.config.get("arm_id"), "block_id": self.config.get("block_id"),
               "arm_id_arrival": self.config.get("arm_id"), "block_id_arrival": self.config.get("block_id"),
               "profile": None, "batch_cap": self.config["batch_cap"], "router": self.config["router"],
               "queue_len_arrival": len(self.waiters), "inflight_arrival": self.inflight, "paused_arrival": self.paused,
               "prompt_tokens": None, "completion_tokens": None, "status": None, "finish_reason": None,
               "reply_text": None, "reply_tool_calls": None, "error": None, "bytes_down": 0,
               "wire": "compact" if compose else "full", "bytes_up_client": len(raw), "t_body": t_body}
        rec.update(compose_ok=compose_info.get("compose_ok"), t_compose_ms=compose_info.get("t_compose_ms"),
                   t_verify_ms=compose_info.get("t_verify_ms"), compose_error=compose_err[1] if compose_err else None)
        self.counters["arrived"] += 1
        router = self.router
        admitted = False
        compact = self.compactor is not None and meta["class"] in self.compact_classes

        def slot_done() -> None:
            nonlocal admitted
            if admitted:
                admitted = False
                self.release()

        pending_step = False
        try:
            if compose_err is not None:
                raise GatewayError(compose_err[0], compose_err[1])
            scorer = router.scorer
            stream = bool(body.get("stream"))
            edge_pass = scorer is not None and scorer.mode == "edge_pass"
            if scorer is not None:
                rec["score_kind"] = scorer.kind
            if scorer is not None and scorer.mode in ("probe", "prefill"):
                await self.run_probe(router, scorer, meta, body, h, rec)
            if edge_pass:
                tier = "edge"
            else:
                tier = await self.call_hook(rec, "route", router.route, meta, self.node_state(), default="edge")
                if tier not in ("edge", "cloud"):
                    rec["router_error"] = f"route returned {tier!r}"
                    tier = "edge"
                rec["routed"] = tier
                rec["router_info"] = router.decision_info()
                self.counters["routed_" + tier] += 1
                pending_step = tier == "edge" and router.wants_feedback
            rec["served_by"] = tier
            if tier == "edge" and not self.engine_url:
                await self.wait_engine(self.args.max_pause_s if self.args.max_pause_s > 0 else 900.0)
            if (tier == "edge" and not self.engine_url) or (tier == "cloud" and not self.cloud_url):
                raise GatewayError(503, f"no {tier} endpoint configured")
            if tier == "edge":
                adm: dict = {}
                rec["t_admit"] = await self.admit(meta["family"], adm)
                admitted = True
                rec.update(adm)
                rec["arm_id"], rec["block_id"] = self.config.get("arm_id"), self.config.get("block_id")
                rec["profile"], rec["batch_cap"] = self.config.get("profile"), self.config["batch_cap"]
                self.counters["admitted"] += 1
                rec["queue_wait_ms"] = (rec["t_admit"] - t_arrive) / 1e6
            else:
                rec["t_admit"] = now_ns()
            cascade = tier == "edge" and not edge_pass and bool(self.call_hook_sync(rec, "wants_after_edge", router.wants_after_edge, meta, default=bool(router.needs_after_edge)))
            buffered = tier == "edge" and (cascade or edge_pass)
            edge_body = body
            if cascade:
                lim = self.call_hook_sync(rec, "edge_max_tokens", router.edge_max_tokens, meta, default=None)
                if lim:
                    edge_body = dict(body, max_tokens=int(lim))
                    rec["cascade_budget"] = int(lim)
            fb_done = False
            if not buffered:
                if compact:
                    summary = await self.compact_through(tier, body, h, rec, slot_done)
                else:
                    summary = await self.proxy_through(tier, body, h, rec, request, stream, on_upstream_done=slot_done)
            else:
                edge = await self.call_upstream("edge", edge_body, h, rec, None, buffer_only=True,
                                                extra_fields=scorer.request_fields() if edge_pass else None,
                                                collect_top=scorer.top_k if edge_pass else 0)
                self.release()
                admitted = False
                edge_ok = edge["status"] == 200 and edge["error"] is None
                rec["edge_attempt"] = {"status": edge["status"], "t_first_token": edge["t_first_token"],
                                       "t_done": edge["t_done"], "text": edge["text"], "error": edge["error"],
                                       "usage": edge["usage"]}
                edge_reply = {k: edge[k] for k in ("text", "tool_calls", "finish_reason", "usage", "status",
                                                   "t_first_token", "t_done", "error")}
                if "cascade_budget" in rec:
                    rec["edge_attempt_tokens"] = (edge["usage"] or {}).get("completion_tokens")
                    rec["edge_attempt_ms"] = None if edge["t_done"] is None else (edge["t_done"] - rec["t_admit"]) / 1e6
                    rec["edge_finish_reason"] = edge["finish_reason"]
                if edge_pass:
                    dec = "accept"
                    if edge_ok:
                        u = await self.call_hook(rec, "score", scorer.score, meta, dict(edge_reply, token_top=edge["token_top"]),
                                                 default=None)
                        if u is not None:
                            meta["race_u"] = u
                            rec["score_u"] = u
                        r = await self.call_hook(rec, "route", router.route, meta, self.node_state(), default="edge")
                        if r not in ("edge", "cloud"):
                            r = "edge"
                        rec["routed"] = r
                        rec["router_info"] = router.decision_info()
                        self.counters["routed_" + r] += 1
                        pending_step = r == "edge" and router.wants_feedback
                        dec = "escalate" if r == "cloud" else "accept"
                    else:
                        rec["routed"] = "edge"
                else:
                    dec = await self.call_hook(rec, "after_edge", router.after_edge, meta, edge_reply, self.node_state(),
                                               default="accept")
                if "cascade_budget" in rec:
                    rec["cascade_reason"] = (router.decision_info() or {}).get("reason") if dec == "escalate" else "complete"
                if dec == "escalate":
                    rec["escalated_after_edge"] = True
                    self.counters["escalated_after_edge"] += 1
                    if compact:
                        summary = await self.compact_through("cloud", body, h, rec)
                    else:
                        summary = await self.proxy_through("cloud", body, h, rec, request, stream)
                    rec["served_by"] = "cloud"
                else:
                    summary = await self.replay(edge, stream, rec, request, compact)
                    rec["served_by"] = "edge"
                    if edge_pass and edge_ok and pending_step:
                        await self.call_hook(rec, "after_edge", router.after_edge, meta, edge_reply, self.node_state(),
                                             default="accept")
                        fb_done = True
            rec["upstream_status"] = summary["status"]
            rec["status"] = summary["status"] if summary["error"] is None else 502
            rec["error"] = summary["error"]
            rec["t_first_token"] = summary["t_first_token"]
            rec["t_done"] = summary["t_done"]
            rec["finish_reason"] = summary["finish_reason"]
            rec["reply_text"] = summary["text"]
            rec["reply_tool_calls"] = summary["tool_calls"]
            if summary["status"] and summary["status"] != 200 and not summary["text"]:
                rec["reply_text"] = summary["raw"][:500].decode("utf-8", "replace") if summary["raw"] else None
            u = summary["usage"] or {}
            rec["prompt_tokens"], rec["completion_tokens"] = u.get("prompt_tokens"), u.get("completion_tokens")
            resp_obj = summary.get("response")
            if pending_step and not fb_done:
                if rec["status"] == 200 and rec["served_by"] == "edge":
                    await self.call_hook(rec, "after_edge", router.after_edge, meta,
                                         {"text": summary["text"], "tool_calls": summary["tool_calls"],
                                          "finish_reason": summary["finish_reason"], "usage": summary["usage"],
                                          "status": 200, "t_first_token": summary["t_first_token"],
                                          "t_done": summary["t_done"]}, self.node_state(), default="accept")
                else:
                    self.abandon(router, meta)
                fb_done = True
            pending_step = not fb_done and pending_step
            ok = rec["status"] == 200
            self.counters["completed" if ok else "errors"] += 1
            if not ok:
                self.counters["failed"] += 1
            if rec["served_by"] == "edge":
                self.counters["served_edge"] += 1
            elif rec["served_by"] == "cloud":
                self.counters["served_cloud"] += 1
                self.counters["escalated"] += 1
            if ok:
                self.recent.add(rec["t_done"], (rec["t_done"] - t_arrive) / 1e6,
                                None if rec["t_first_token"] is None else (rec["t_first_token"] - t_arrive) / 1e6)
            return resp_obj
        except asyncio.CancelledError:
            rec["error"] = "client_disconnected"
            rec["status"] = rec["status"] or 499
            self.counters["client_disconnected"] += 1
            raise
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            status = e.status if isinstance(e, GatewayError) else 500
            rec["status"] = rec["status"] or status
            self.counters["errors"] += 1
            self.counters["failed"] += 1
            return web.json_response({"error": rec["error"]}, status=status)
        finally:
            rec["t_done"] = rec["t_done"] or now_ns()
            if admitted:
                self.release()
            if pending_step:
                self.abandon(router, meta)
            rec["t_done_wall"] = time.time_ns()
            self.logf.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")

    @staticmethod
    def abandon(router, meta: dict) -> None:
        fn = getattr(router, "abandon", None)
        if fn is not None:
            try:
                fn(meta)
            except Exception:
                pass

    async def proxy_through(self, tier: str, body: dict, hdrs, rec: dict, request: web.Request, stream: bool,
                            on_upstream_done=None) -> dict:
        resp_holder: Dict[str, Any] = {"resp": None, "failed": None}
        queue: asyncio.Queue = asyncio.Queue()

        async def pump() -> None:
            try:
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        return
                    if resp_holder["resp"] is None:
                        r = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                                                                    "X-Accel-Buffering": "no"})
                        r.enable_chunked_encoding()
                        await r.prepare(request)
                        resp_holder["resp"] = r
                    rec["bytes_down"] += len(chunk)
                    await resp_holder["resp"].write(chunk)
            except Exception as e:
                resp_holder["failed"] = e

        class Relay:
            async def write(inner, chunk):
                if resp_holder["failed"] is not None:
                    raise resp_holder["failed"]
                queue.put_nowait(chunk)

        writer = asyncio.ensure_future(pump()) if stream else None
        try:
            summary = await self.call_upstream(tier, body, hdrs, rec, Relay() if stream else None)
            if on_upstream_done is not None:
                on_upstream_done()
            if writer is not None:
                queue.put_nowait(None)
                await writer
                if resp_holder["failed"] is not None and summary["error"] is None:
                    summary["error"] = f"{type(resp_holder['failed']).__name__}: {resp_holder['failed']}"
        except BaseException:
            if writer is not None:
                writer.cancel()
            raise
        if resp_holder["resp"] is not None:
            await resp_holder["resp"].write_eof()
            summary["response"] = resp_holder["resp"]
        else:
            summary["response"] = self.plain_response(summary, rec)
        return summary

    async def replay(self, edge: dict, stream: bool, rec: dict, request: web.Request, compact: bool = False) -> dict:
        if compact:
            edge = dict(edge)
            edge["response"] = self.compact_response(edge, rec)
            return edge
        if edge["error"] is not None or not stream or edge["status"] != 200:
            edge = dict(edge)
            edge["response"] = self.plain_response(edge, rec)
            return edge
        r = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        r.enable_chunked_encoding()
        await r.prepare(request)
        for _, chunk in edge["chunks"]:
            if rec.get("t_first_byte_client") is None:
                rec["t_first_byte_client"] = now_ns()
            rec["bytes_down"] += len(chunk)
            await r.write(chunk)
        await r.write_eof()
        edge = dict(edge)
        edge["response"] = r
        return edge

    async def compact_through(self, tier: str, body: dict, hdrs, rec: dict, on_upstream_done=None) -> dict:
        summary = await self.call_upstream(tier, body, hdrs, rec, None, buffer_only=True)
        if on_upstream_done is not None:
            on_upstream_done()
        summary["response"] = self.compact_response(summary, rec)
        return summary

    def compact_response(self, summary: dict, rec: dict) -> web.Response:
        if summary["error"] is not None or summary["status"] != 200:
            return self.plain_response(summary, rec)
        content, ok = self.compactor.compact(rec.get("family"), summary["text"])
        rec["delivery"] = "compact"
        rec["compact_extracted"] = ok
        rec["compact_bytes"] = len(content.encode("utf-8"))
        self.counters["compact_replies"] += 1
        if not ok:
            self.counters["compact_fallback"] += 1
        data = json.dumps({"id": "ciara-compact", "object": "chat.completion", "model": self.config.get("profile"),
                           "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                        "finish_reason": summary["finish_reason"]}],
                           "usage": summary["usage"]}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        rec["bytes_down"] += len(data)
        rec["t_first_byte_client"] = rec.get("t_first_byte_client") or now_ns()
        return web.Response(body=data, status=200, content_type="application/json")

    def plain_response(self, summary: dict, rec: dict) -> web.Response:
        if summary["error"] is not None:
            return web.json_response({"error": summary["error"]}, status=502)
        ct = (summary.get("headers") or {}).get("Content-Type", "application/json")
        body = summary["raw"]
        rec["bytes_down"] += len(body)
        rec["t_first_byte_client"] = rec.get("t_first_byte_client") or now_ns()
        return web.Response(body=body, status=summary["status"], content_type=ct.split(";")[0])

    async def h_config_post(self, request: web.Request) -> web.Response:
        try:
            cfg = await request.json()
            if not isinstance(cfg, dict):
                raise ValueError("configuration must be a JSON object")
            new = self.apply_config(cfg)
        except (ValueError, KeyError) as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True, "config": new, "version": self.config_version, "engine_url": self.engine_url})

    async def h_config_get(self, request: web.Request) -> web.Response:
        return web.json_response({"config": self.config, "version": self.config_version})

    async def h_state(self, request: web.Request) -> web.Response:
        t = now_ns()
        paused_ns = self.pause_total_ns + ((t - self.pause_since) if self.paused else 0)
        rstate = getattr(self.router, "state", None)
        return web.json_response({
            "node": self.node, "run_id": self.run_id, "queue_len": len(self.waiters), "in_flight": self.inflight,
            "inflight": self.inflight, "paused": self.paused, "config": self.config, "config_version": self.config_version,
            "engine_url": self.engine_url, "cloud_url": self.cloud_url, "counters": dict(self.counters),
            "recent": self.recent.stats(t), "router_state": rstate() if callable(rstate) else None,
            "cloud_state": {**self.cloud_state, "ok": self.cloud_poll_ok, "rtt_s": self.cloud_rtt_s} if self.cloud_state else None,
            "max_inflight_seen": self.max_inflight_seen, "max_queue_seen": self.max_queue_seen,
            "pauses": self.pauses, "paused_total_s": paused_ns / 1e9, "uptime_s": (t - self.started) / 1e9,
            "context_store": self.store_info or None,
            "t": t, "t_wall": time.time_ns(), "log": self.log_path})

    async def h_admission(self, request: web.Request) -> web.Response:
        try:
            js = await request.json()
        except ValueError:
            return web.json_response({"error": "JSON body required"}, status=400)
        paused = bool(js.get("paused"))
        if js.get("engine_url"):
            self.engine_url = js["engine_url"]
        if js.get("profile"):
            self.config = dict(self.config, profile=js["profile"])
            try:
                self.router.configure(self.config)
            except Exception as e:
                self.event("router_reconfigure_error", error=repr(e))
        drained = True
        if paused:
            self.set_paused(True)
            if js.get("drain"):
                try:
                    await asyncio.wait_for(self.drain_event.wait(), timeout=float(js.get("timeout_s", 60)))
                except asyncio.TimeoutError:
                    drained = False
        else:
            self.edge_fail_streak = 0
            self.set_paused(False)
        return web.json_response({"ok": True, "paused": self.paused, "drained": drained, "inflight": self.inflight,
                                  "queue_len": len(self.waiters), "engine_url": self.engine_url})

    async def h_feedback(self, request: web.Request) -> web.Response:
        try:
            js = await request.json()
        except ValueError:
            return web.json_response({"error": "JSON body required"}, status=400)
        labels = js.get("labels") if isinstance(js, dict) and "labels" in js else [js]
        label = getattr(self.router, "label", None)
        applied = unknown = 0
        for l in labels:
            if not isinstance(l, dict) or "req" not in l:
                return web.json_response({"error": "each label needs req"}, status=400)
            wrong = l.get("wrong")
            if wrong is None and l.get("correct") is not None:
                wrong = not bool(l["correct"])
            if label is not None and label(str(l["req"]), wrong):
                applied += 1
            else:
                unknown += 1
        self.counters["labels_applied"] += applied
        return web.json_response({"ok": True, "applied": applied, "unknown": unknown})

    async def h_healthz(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def start(self) -> None:
        conn = aiohttp.TCPConnector(limit=0, keepalive_timeout=60)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=self.args.upstream_read_timeout)
        self.session = aiohttp.ClientSession(connector=conn, timeout=timeout, auto_decompress=False)
        data = web.Application(client_max_size=8 * 1024 * 1024)
        data.router.add_post("/v1/chat/completions", self.handle_chat)
        data.router.add_post(COMPOSE_PATH, self.handle_compose)
        data.router.add_get("/healthz", self.h_healthz)
        ctl = web.Application()
        ctl.router.add_post("/config", self.h_config_post)
        ctl.router.add_get("/config", self.h_config_get)
        ctl.router.add_get("/state", self.h_state)
        ctl.router.add_post("/admission", self.h_admission)
        ctl.router.add_post("/feedback", self.h_feedback)
        ctl.router.add_get("/healthz", self.h_healthz)
        self.runners = []
        for app, port in ((data, self.args.data_port), (ctl, self.args.ctl_port)):
            runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
            await runner.setup()
            await web.TCPSite(runner, self.args.bind, port, backlog=1024).start()
            self.runners.append(runner)
        if self.args.cloud_poll_s > 0 and self.cloud_url:
            self._tasks.append(asyncio.create_task(self.cloud_poller()))
        self.event("start", data_port=self.args.data_port, ctl_port=self.args.ctl_port, context_store=self.store_info or None,
                   compose_verify=self.compose_verify)

    async def stop(self) -> None:
        self.event("stop", inflight=self.inflight, queue_len=len(self.waiters))
        for t in self._tasks:
            t.cancel()
        for r in self.runners:
            await r.shutdown()
            await r.cleanup()
        await self.session.close()
        for f in (self.logf, self.evf, self.probef):
            f.close()


async def amain(args):
    gw = Gateway(args)
    await gw.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    print(f"edge-gw-{args.node} listening: data {args.bind}:{args.data_port}, control {args.bind}:{args.ctl_port}, "
          f"log {gw.log_path}", flush=True)
    await stop.wait()
    await gw.stop()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Edge gateway: admission queue, streaming proxy, pluggable router, JSONL log.')
    ap.add_argument("--node", type=int, default=int(os.environ.get("NODE", "1")))
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", "adhoc"))
    ap.add_argument("--log-dir", default=os.environ.get("LOG_DIR") or os.path.join(os.environ.get("CIARA_ROOT", "."), "runs"))
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--data-port", type=int, default=8000)
    ap.add_argument("--ctl-port", type=int, default=8001)
    ap.add_argument("--engine-url", default=os.environ.get("ENGINE_URL", ""))
    ap.add_argument("--engine-map", default=os.environ.get("ENGINE_MAP", ""), help="JSON {profile: url}")
    ap.add_argument("--cloud-url", default=os.environ.get("CLOUD_URL", ""),
                    help="cloud gateway base URL; its GET /state (control and data share port 8001) feeds router I3")
    ap.add_argument("--cloud-model", default=os.environ.get("CLOUD_MODEL", ""))
    ap.add_argument("--cloud-poll-s", type=float, default=float(os.environ.get("CLOUD_POLL_S", "1.0")))
    ap.add_argument("--backhaul-mbps", type=float, default=float(os.environ.get("BACKHAUL_MBPS", "0")),
                    help="shaped backhaul rate for router I3 (0: the router's placeholder)")
    ap.add_argument("--profile-table", default=os.environ.get("PROFILE_TABLE", ""), help="profile table (JSON) of the baselines package, for router I3")
    ap.add_argument("--feedback-fn", default=os.environ.get("FEEDBACK_FN", ""),
                    help="module:callable(req_meta, edge_reply) -> True|False|None, inline RACE labels; default: POST /feedback")
    ap.add_argument("--upstream-read-timeout", type=float, default=600.0)
    ap.add_argument("--compact-classes", default=os.environ.get("COMPACT_CLASSES", "A,B"),
                    help="device classes that receive the compact final reply without streaming; empty turns it off")
    ap.add_argument("--pilot-root", default=os.environ.get("CIARA_PILOT_ROOT", ""),
                    help="directory with eval/ (frozen suites) for the compact extraction")
    ap.add_argument("--context-store", default=os.environ.get("CONTEXT_STORE", ""),
                    help="JSON context store: static prompt parts, home inventories, sensor metadata; enables POST /v1/ciara/compose")
    ap.add_argument("--compose-verify", type=int, default=int(os.environ.get("COMPOSE_VERIFY", "1")),
                    help="check every composed request against the expected messages hash of its item (0 turns the check off)")
    ap.add_argument("--max-pause-s", type=float, default=900.0, help="release an admission pause after this long (0: no limit)")
    ap.add_argument("--recent-window-s", type=float, default=60.0)
    ap.add_argument("--inject-usage", type=int, default=1, help="add stream_options.include_usage to streamed requests")
    return ap.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
