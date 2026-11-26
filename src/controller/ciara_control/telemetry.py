from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import promtext

_QUEUE = ("queue_len", "queue_length", "queue")
_INFLIGHT = ("in_flight", "inflight")
_CONFIG = ("config", "config_in_force")
_COUNTERS = ("counters", "counts")
_ARRIVED = ("arrived", "arrivals", "received", "n_arrived")
_ADMITTED = ("admitted", "n_admitted")
_COMPLETED = ("completed", "done", "finished", "n_completed")
_ESCALATED = ("escalated", "n_escalated", "to_cloud")
_ERRORS = ("errors", "failed", "n_errors")
_RECENT = ("recent", "recent_latency", "latency")


def _pick(d: Optional[dict], names: tuple):
    if not isinstance(d, dict):
        return None
    for n in names:
        if n in d:
            return d[n]
    return None


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def parse_gateway_state(raw: Optional[dict]) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    counters_raw = _pick(raw, _COUNTERS)
    src = counters_raw if isinstance(counters_raw, dict) else raw
    recent = _pick(raw, _RECENT)
    return {
        "queue_len": _num(_pick(raw, _QUEUE)),
        "in_flight": _num(_pick(raw, _INFLIGHT)),
        "config": _pick(raw, _CONFIG) if isinstance(_pick(raw, _CONFIG), dict) else None,
        "counters": {
            "arrived": _num(_pick(src, _ARRIVED)),
            "admitted": _num(_pick(src, _ADMITTED)),
            "completed": _num(_pick(src, _COMPLETED)),
            "escalated": _num(_pick(src, _ESCALATED)),
            "errors": _num(_pick(src, _ERRORS)),
        },
        "recent": recent if isinstance(recent, dict) else None,
    }


@dataclass
class Snapshot:
    t_mono_ns: int
    wall_ns: int
    gateway_raw: Optional[dict] = None
    gateway: Optional[dict] = None
    status_raw: Optional[dict] = None
    profile: Optional[str] = None
    endpoint: Optional[str] = None
    metrics: Optional[dict] = None
    errors: list = field(default_factory=list)
    poll_ms: float = 0.0


def derive_telemetry(prev: Optional[Snapshot], cur: Snapshot) -> dict:
    out: dict = {"dt_s": None, "arrival_rate": None, "arrivals": None, "completed": None,
                 "escalated": None, "errors": None, "queue_len": None, "in_flight": None,
                 "recent": None, "gateway_ok": cur.gateway is not None}
    g = cur.gateway
    if g:
        out["queue_len"] = g.get("queue_len")
        out["in_flight"] = g.get("in_flight")
        out["recent"] = g.get("recent")
    if prev is not None:
        dt = (cur.t_mono_ns - prev.t_mono_ns) / 1e9
        out["dt_s"] = dt
        if g and prev.gateway:
            for key, ck in (("arrivals", "arrived"), ("completed", "completed"),
                            ("escalated", "escalated"), ("errors", "errors")):
                a, b = g["counters"].get(ck), prev.gateway["counters"].get(ck)
                out[key] = (a - b) if (a is not None and b is not None and a >= b) else None
            if out["arrivals"] is not None and dt > 0:
                out["arrival_rate"] = out["arrivals"] / dt
    same_engine = (prev is not None and prev.endpoint is not None and prev.endpoint == cur.endpoint
                   and prev.metrics is not None)
    out["engine_ok"] = cur.metrics is not None
    out["engine"] = (promtext.summarize_vllm(cur.metrics, prev.metrics if same_engine else None)
                     if cur.metrics is not None else None)
    out["profile"] = cur.profile
    out["engine_endpoint"] = cur.endpoint
    return out


def config_matches(in_force: Optional[dict], pushed: Optional[dict]) -> bool:
    if not in_force or not pushed:
        return True
    for key in ("arm_id", "profile", "batch_cap", "block_id", "router", "escalate_family2"):
        if key in in_force and key in pushed and in_force[key] != pushed[key]:
            return False
    return True
