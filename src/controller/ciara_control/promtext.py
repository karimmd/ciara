from __future__ import annotations

import math
import re
from typing import Optional

_LINE = re.compile(r"^(?P<name>[^\s{]+)(?P<labels>\{.*\})?\s+(?P<val>\S+)(?:\s+\S+)?$")


def _to_float(s: str) -> float:
    s = s.strip()
    if s in ("+Inf", "Inf"):
        return math.inf
    if s == "-Inf":
        return -math.inf
    if s == "NaN":
        return math.nan
    return float(s)


def parse(text: str) -> dict:
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            v = _to_float(m.group("val"))
        except ValueError:
            continue
        out[m.group("name") + (m.group("labels") or "")] = v
    return out


def sum_by_name(samples: dict, include_buckets: bool = False) -> dict:
    out: dict[str, float] = {}
    for key, v in samples.items():
        name = key.split("{", 1)[0]
        if not include_buckets and name.endswith("_bucket"):
            continue
        if isinstance(v, float) and not math.isfinite(v):
            continue
        out[name] = out.get(name, 0.0) + v
    return out


def keep_vllm(samples: dict) -> dict:
    return {k: v for k, v in samples.items() if k.startswith("vllm:")}


def _first(d: dict, names: tuple) -> Optional[float]:
    for n in names:
        if n in d:
            return d[n]
    return None


_GAUGES = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}
_COUNTERS = {
    "prompt_tokens": ("vllm:prompt_tokens_total", "vllm:prompt_tokens"),
    "generation_tokens": ("vllm:generation_tokens_total", "vllm:generation_tokens"),
    "requests_finished": ("vllm:request_success_total", "vllm:request_success"),
    "preemptions": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
}
_HISTS = {
    "e2e": "vllm:e2e_request_latency_seconds",
    "ttft": "vllm:time_to_first_token_seconds",
    "itl": "vllm:inter_token_latency_seconds",
    "tpot": "vllm:time_per_output_token_seconds",
    "queue": "vllm:request_queue_time_seconds",
}


def summarize_vllm(curr: dict, prev: Optional[dict] = None) -> dict:
    out: dict = {}
    for key, names in _GAUGES.items():
        out[key] = _first(curr, names)
    reset = False
    for key, names in _COUNTERS.items():
        c = _first(curr, names)
        p = _first(prev, names) if prev else None
        if c is None or p is None or c < p:
            out[key + "_delta"] = None
            reset = reset or (c is not None and p is not None and c < p)
        else:
            out[key + "_delta"] = c - p
    for key, base in _HISTS.items():
        cs, cc = curr.get(base + "_sum"), curr.get(base + "_count")
        ps = prev.get(base + "_sum") if prev else None
        pc = prev.get(base + "_count") if prev else None
        if None in (cs, cc, ps, pc) or cc < pc or cs < ps:
            out[key + "_mean_s"] = None
            out[key + "_count_delta"] = None
        else:
            dn = cc - pc
            out[key + "_count_delta"] = dn
            out[key + "_mean_s"] = (cs - ps) / dn if dn > 0 else None
    out["counter_reset"] = reset
    return out
