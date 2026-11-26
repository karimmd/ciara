from __future__ import annotations

import asyncio
import codecs
import json
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

Stamp = Tuple[int, int]


def stamp() -> Stamp:
    return time.monotonic_ns(), time.time_ns()


class Conn:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, port: int) -> None:
        self.reader = reader
        self.writer = writer
        self.port = port
        self.last_used = time.monotonic()
        self.uses = 0

    def close(self) -> None:
        try:
            self.writer.close()
        except Exception:
            pass


async def open_conn(host: str, port: int, local_port: int, timeout: float) -> Conn:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setblocking(False)
        sock.bind(("0.0.0.0", local_port))
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(loop.sock_connect(sock, (host, port)), timeout)
        reader, writer = await asyncio.open_connection(sock=sock)
        return Conn(reader, writer, local_port)
    except BaseException:
        sock.close()
        raise


def build_request(host_header: str, path: str, headers: Dict[str, str], body: bytes, keep_alive: bool) -> bytes:
    lines = [f"POST {path} HTTP/1.1", f"Host: {host_header}", "Content-Type: application/json",
             "Accept: text/event-stream", f"Content-Length: {len(body)}",
             "Connection: " + ("keep-alive" if keep_alive else "close")]
    lines += [f"{k}: {v}" for k, v in headers.items()]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + body


class SSE:
    def __init__(self) -> None:
        self.buf = ""
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")
        self.text: List[str] = []
        self.tool: List[str] = []
        self.events = 0
        self.token_events = 0
        self.finish: Optional[str] = None
        self.usage: Optional[Dict[str, Any]] = None
        self.done = False
        self.first_token_stamp: Optional[Stamp] = None
        self.error: Optional[str] = None

    def feed(self, data: bytes) -> None:
        self.buf += self._dec.decode(data)
        while True:
            i = self.buf.find("\n\n")
            if i < 0:
                return
            event, self.buf = self.buf[:i], self.buf[i + 2:]
            for line in event.split("\n"):
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    self.done = True
                    continue
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                self.events += 1
                if "error" in obj:
                    self.error = json.dumps(obj["error"])[:500]
                if obj.get("usage"):
                    self.usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content")
                    tcs = delta.get("tool_calls")
                    if piece:
                        self.text.append(piece)
                    if tcs:
                        self.tool.append(json.dumps(tcs, separators=(",", ":")))
                    if (piece or tcs) and self.first_token_stamp is None:
                        self.first_token_stamp = stamp()
                    if piece or tcs:
                        self.token_events += 1
                    if ch.get("finish_reason"):
                        self.finish = ch["finish_reason"]


def parse_json_reply(body: str) -> Optional[Tuple[str, Optional[str], Optional[Dict[str, Any]]]]:
    try:
        js = json.loads(body)
        ch = js["choices"][0]
        return (ch.get("message") or {}).get("content") or "", ch.get("finish_reason"), js.get("usage")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None


class Response:
    def __init__(self) -> None:
        self.status = 0
        self.headers: Dict[str, str] = {}
        self.first_byte: Optional[Stamp] = None
        self.last_byte: Optional[Stamp] = None
        self.bytes_down = 0
        self.body_text = ""
        self.sse = SSE()
        self.keep_alive = True
        self.eof_before_data = False


async def _read(reader: asyncio.StreamReader, resp: Response, n: int = 65536) -> bytes:
    data = await reader.read(n)
    if data:
        if resp.first_byte is None:
            resp.first_byte = stamp()
        resp.bytes_down += len(data)
    return data


async def read_response(conn: Conn) -> Response:
    resp = Response()
    reader = conn.reader
    buf = b""
    while b"\r\n\r\n" not in buf:
        data = await _read(reader, resp)
        if not data:
            if not buf:
                resp.eof_before_data = True
            raise ConnectionError("connection closed before the response head")
        buf += data
    head, rest = buf.split(b"\r\n\r\n", 1)
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ", 2)
    resp.status = int(parts[1])
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            resp.headers[k.strip().lower()] = v.strip()
    resp.keep_alive = resp.headers.get("connection", "keep-alive").lower() != "close" and parts[0] == "HTTP/1.1"
    is_sse = "text/event-stream" in resp.headers.get("content-type", "")
    sink = bytearray()

    def deliver(chunk: bytes) -> None:
        if is_sse:
            resp.sse.feed(chunk)
        elif len(sink) < 65536:
            sink.extend(chunk[: 65536 - len(sink)])

    if resp.headers.get("transfer-encoding", "").lower() == "chunked":
        pending = rest
        while True:
            while b"\r\n" not in pending:
                data = await _read(reader, resp)
                if not data:
                    raise ConnectionError("connection closed inside a chunked body")
                pending += data
            size_line, pending = pending.split(b"\r\n", 1)
            size = int(size_line.split(b";")[0].strip() or b"0", 16)
            if size == 0:
                while b"\r\n" not in pending:
                    data = await _read(reader, resp)
                    if not data:
                        break
                    pending += data
                break
            while len(pending) < size + 2:
                data = await _read(reader, resp)
                if not data:
                    raise ConnectionError("connection closed inside a chunk")
                pending += data
            deliver(pending[:size])
            pending = pending[size + 2:]
    elif "content-length" in resp.headers:
        need = int(resp.headers["content-length"])
        got = len(rest)
        if rest:
            deliver(rest)
        while got < need:
            data = await _read(reader, resp)
            if not data:
                raise ConnectionError("connection closed inside the body")
            got += len(data)
            deliver(data)
    else:
        resp.keep_alive = False
        if rest:
            deliver(rest)
        while True:
            data = await _read(reader, resp)
            if not data:
                break
            deliver(data)
    resp.last_byte = stamp()
    resp.body_text = bytes(sink).decode("utf-8", "replace")
    return resp
