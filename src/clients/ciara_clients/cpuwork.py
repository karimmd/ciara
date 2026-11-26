from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

_INV = re.compile(r"^(\w+\.\w+) '([^']*)' = (.*)$", re.M)
_TOK = re.compile(r"\w+|[^\w\s]")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")

Work = Generator[None, None, Any]


def prep_family1(system: str, user: str, body_fn: Callable[[], bytes]) -> Work:
    devices = _INV.findall(system)
    yield
    names = " ".join(d[1] for d in devices)
    tokens = _TOK.findall(user.lower())
    hits = sum(1 for t in tokens if t in names.lower())
    yield
    state_counts: Dict[str, int] = {}
    for _, _, st in devices:
        key = st.split(";")[0]
        state_counts[key] = state_counts.get(key, 0) + 1
    yield
    body = body_fn()
    return {"devices": len(devices), "tokens": len(tokens), "hits": hits, "body": body}


def _readings_features(lines: List[str]) -> Tuple[List[List[float]], List[Tuple[float, float, float, float]]]:
    rows: List[List[float]] = []
    for ln in lines:
        if "|" in ln and ln[:1].isdigit():
            cells = [c.strip() for c in ln.split("|")]
            vals = []
            for c in cells[1:]:
                vals.append(float(c) if _NUM.fullmatch(c) else float("nan"))
            rows.append(vals)
    feats = []
    if rows:
        for j in range(len(rows[0])):
            col = [r[j] for r in rows if j < len(r) and r[j] == r[j]]
            if col:
                m = sum(col) / len(col)
                feats.append((min(col), max(col), m, sum((x - m) ** 2 for x in col) / len(col)))
    return rows, feats


def prep_family2(system: str, user: str, body_fn: Callable[[], bytes]) -> Work:
    lines = system.split("\n")
    yield
    rows, feats = _readings_features(lines)
    yield
    tokens = _TOK.findall(user.lower())
    yield
    body = body_fn()
    return {"rows": len(rows), "channels": len(feats), "tokens": len(tokens), "body": body}


def prep_family1_compact(user: str, home: str, body_fn: Callable[[], bytes]) -> Work:
    tokens = _TOK.findall(user.lower())
    yield
    body = body_fn()
    return {"tokens": len(tokens), "body": body}


def prep_family2_compact(readings: str, question: str, body_fn: Callable[[], bytes]) -> Work:
    lines = readings.split("\n")
    yield
    rows, feats = _readings_features(lines)
    yield
    tokens = _TOK.findall(question.lower())
    yield
    body = body_fn()
    return {"rows": len(rows), "channels": len(feats), "tokens": len(tokens), "body": body}


def post_reply(text: str) -> Work:
    try:
        obj = json.loads(text)
        n = len(obj) if hasattr(obj, "__len__") else 1
    except ValueError:
        n = len(_TOK.findall(text))
    yield
    return {"items": n}


async def run_throttled(share: float, work: Work) -> Tuple[Any, Dict[str, float]]:
    t0 = time.monotonic_ns()
    cpu_ns = 0
    owed = 0.0
    result: Any = None
    factor = 1.0 / share - 1.0
    while True:
        c0 = time.thread_time_ns()
        done = False
        try:
            next(work)
        except StopIteration as stop:
            result = stop.value
            done = True
        d = time.thread_time_ns() - c0
        cpu_ns += d
        owed += d * factor / 1e9
        if owed >= 0.002 or (done and owed > 0.0005):
            t = time.monotonic()
            await asyncio.sleep(owed)
            owed -= time.monotonic() - t
        elif not done:
            await asyncio.sleep(0)
        if done:
            break
    wall_ms = (time.monotonic_ns() - t0) / 1e6
    cpu_ms = cpu_ns / 1e6
    return result, {"cpu_host_ms": round(cpu_ms, 4), "dev_ms": round(wall_ms, 4),
                    "share_eff": round(cpu_ms / wall_ms, 4) if wall_ms > 0 else None}
