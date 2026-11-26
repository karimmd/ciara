from __future__ import annotations

import asyncio
import collections
import json
import math
import os
import queue
import threading
from typing import Any, Optional

from .util import dumps, repair_jsonl


class FifoAdmission:
    def __init__(self, limit: int):
        self.limit = int(limit)
        self.in_flight = 0
        self._waiters: collections.deque = collections.deque()

    @property
    def queue_len(self) -> int:
        return len(self._waiters)

    async def acquire(self) -> None:
        if self.in_flight < self.limit and not self._waiters:
            self.in_flight += 1
            return
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()
            else:
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass
            raise

    def release(self) -> None:
        self.in_flight -= 1
        self._grant()

    def set_limit(self, limit: int) -> None:
        self.limit = int(limit)
        self._grant()

    def _grant(self) -> None:
        while self._waiters and self.in_flight < self.limit:
            fut = self._waiters.popleft()
            if fut.done():
                continue
            self.in_flight += 1
            fut.set_result(None)


class JsonlWriter:
    _STOP = object()

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        repair_jsonl(path)
        self.path = path
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._fh = open(path, "ab", buffering=0)
        self._thread = threading.Thread(target=self._loop, name="jsonl-writer", daemon=True)
        self._thread.start()

    def write(self, rec: dict) -> None:
        self._q.put(rec)

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is self._STOP:
                break
            batch = [item]
            try:
                while len(batch) < 256:
                    nxt = self._q.get_nowait()
                    if nxt is self._STOP:
                        self._q.put(self._STOP)
                        break
                    batch.append(nxt)
            except queue.Empty:
                pass
            self._fh.write("".join(dumps(r) + "\n" for r in batch).encode("utf-8"))
        self._fh.flush()

    def close(self) -> None:
        self._q.put(self._STOP)
        self._thread.join(timeout=10)
        self._fh.close()


class SseSplitter:
    def __init__(self):
        self._buf = b""

    def feed(self, chunk: bytes) -> list:
        self._buf += chunk.replace(b"\r\n", b"\n")
        events = []
        while True:
            idx = self._buf.find(b"\n\n")
            if idx < 0:
                break
            ev, self._buf = self._buf[:idx], self._buf[idx + 2:]
            if ev.strip():
                events.append(ev)
        return events

    def flush(self) -> list:
        rest, self._buf = self._buf, b""
        return [rest] if rest.strip() else []


def parse_sse_event(ev: bytes) -> tuple:
    data_lines = []
    for line in ev.split(b"\n"):
        if line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip(b" "))
    if not data_lines:
        return "other", None
    data = b"\n".join(data_lines)
    if data.strip() == b"[DONE]":
        return "done", None
    try:
        return "json", json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "other", None


class RecentWindow:
    def __init__(self, window_s: float = 60.0, maxlen: int = 20000):
        self.window_ns = int(window_s * 1e9)
        self._items: collections.deque = collections.deque(maxlen=maxlen)

    def add(self, t_done_ns: int, latency_ms: float, ttft_ms: Optional[float]) -> None:
        self._items.append((t_done_ns, latency_ms, ttft_ms))

    @staticmethod
    def _q(sorted_vals: list, q: float) -> Optional[float]:
        if not sorted_vals:
            return None
        k = max(0, min(len(sorted_vals) - 1, math.ceil(q * len(sorted_vals)) - 1))
        return sorted_vals[k]

    def stats(self, now_ns: int) -> dict:
        lo = now_ns - self.window_ns
        lat = sorted(x[1] for x in self._items if x[0] >= lo)
        ttft = sorted(x[2] for x in self._items if x[0] >= lo and x[2] is not None)
        n = len(lat)
        return {"window_s": self.window_ns / 1e9, "n": n,
                "mean_ms": (sum(lat) / n) if n else None, "p50_ms": self._q(lat, 0.5),
                "p95_ms": self._q(lat, 0.95), "ttft_p95_ms": self._q(ttft, 0.95)}


_HEADER_FIELDS = {
    "x-ciara-req": "req_id", "x-ciara-device": "device", "x-ciara-class": "class",
    "x-ciara-family": "family", "x-ciara-item": "item", "x-ciara-deadline-ms": "deadline_ms",
    "x-ciara-node": "node", "x-ciara-origin": "node", "x-ciara-arm": "arm_id",
    "x-ciara-block": "block_id", "x-ciara-escalated": "escalated_after_edge",
}


def ciara_headers(headers: Any) -> dict:
    out: dict[str, Any] = {}
    for k, v in headers.items():
        field = _HEADER_FIELDS.get(k.lower())
        if field is None:
            continue
        if field in ("family", "deadline_ms", "block_id"):
            try:
                v = int(v)
            except (TypeError, ValueError):
                pass
        elif field == "escalated_after_edge":
            v = str(v).strip().lower() in ("1", "true", "yes")
        out.setdefault(field, v)
    return out


STALE_RETRY_S = 2.0
_STALE_PREFIXES = ("upstream_error: ServerDisconnectedError", "upstream_error: ClientOSError: [Errno 104]", "upstream_error: ClientConnectionResetError")


def stale_connection_error(error: Optional[str], status: Optional[int]) -> bool:
    return status == 502 and bool(error) and str(error).startswith(_STALE_PREFIXES)
