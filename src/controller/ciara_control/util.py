from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from typing import Any, Iterable


def mono_ns() -> int:
    return time.monotonic_ns()


def wall_ns() -> int:
    return time.time_ns()


def sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(sanitize(obj), separators=(",", ":"), default=str)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: str, text: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def repair_jsonl(path: str) -> int:
    if not os.path.exists(path):
        return 0
    size = os.path.getsize(path)
    if size == 0:
        return 0
    with open(path, "rb+") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) == b"\n":
            return 0
        pos = size
        block = 65536
        last_nl = -1
        while pos > 0 and last_nl < 0:
            start = max(0, pos - block)
            fh.seek(start)
            chunk = fh.read(pos - start)
            idx = chunk.rfind(b"\n")
            if idx >= 0:
                last_nl = start + idx
            pos = start
        keep = last_nl + 1 if last_nl >= 0 else 0
        fh.truncate(keep)
        return size - keep


def read_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


class JsonlLog:
    def __init__(self, path: str, durable: bool = True):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        repair_jsonl(path)
        self.path = path
        self.durable = durable
        self._lock = threading.Lock()
        self._fh = open(path, "ab", buffering=0)

    def append(self, rec: dict) -> None:
        line = (dumps(rec) + "\n").encode("utf-8")
        with self._lock:
            self._fh.write(line)
            if self.durable:
                os.fsync(self._fh.fileno())

    def extend(self, recs: Iterable[dict]) -> None:
        data = "".join(dumps(r) + "\n" for r in recs).encode("utf-8")
        if not data:
            return
        with self._lock:
            self._fh.write(data)
            if self.durable:
                os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


def parse_int_ranges(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)
