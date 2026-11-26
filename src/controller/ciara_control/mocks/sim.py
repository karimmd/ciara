from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np

CAPACITY = {"q35-4b-fp16": 0.9, "q35-4b-ih-w8a8": 1.3, "q35-4b-ih-w4a16": 1.6, "q35-2b-fp16": 1.9,
            "q35-0.8b-fp16": 3.2}


class SimNode:
    def __init__(self, node_id: str, node_index: int, now_s: Callable[[], float], seed: int = 0,
                 rate: float = 1.0, switch_mean_s: float = 0.2, endpoint_base: str = "http://127.0.0.1:0"):
        self.node_id = node_id
        self.now_s = now_s
        self.rate = rate
        self.switch_mean_s = switch_mean_s
        self.endpoint_base = endpoint_base.rstrip("/")
        self.rng = np.random.default_rng([seed, node_index, 7])
        self.lock = threading.RLock()
        self.prepared: list = []
        self.active: Optional[str] = None
        self.last_switch_s: Optional[float] = None
        self.activations: list = []
        self.cfg: Optional[dict] = None
        self.configs: list = []
        self.arrived = self.admitted = self.completed = self.escalated = self.errors = 0
        self.queue = 0
        self.in_flight = 0
        self.last_latency_ms = 0.0
        self._t_last = now_s()
        self.engines: dict[str, dict] = {}


    def endpoint_of(self, profile: str) -> str:
        return f"{self.endpoint_base}/engine/{profile}"

    def prepare(self, profiles: list) -> dict:
        with self.lock:
            for p in profiles:
                if p not in self.prepared:
                    self.prepared.append(p)
                    self.engines.setdefault(p, self._new_engine())
            return {"ok": True, "prepared": list(self.prepared)}

    def status(self) -> dict:
        with self.lock:
            self._advance()
            return {"active_profile": self.active, "profile": self.active,
                    "endpoint": self.endpoint_of(self.active) if self.active else None,
                    "engine_health": "ok" if self.active else "idle", "last_switch_time_s": self.last_switch_s,
                    "engines": {p: {"endpoint": self.endpoint_of(p), "state": "awake" if p == self.active else "sleeping"}
                                for p in self.prepared}}

    def switch_time(self) -> float:
        with self.lock:
            return 0.0 if self.switch_mean_s <= 0 else float(self.rng.gamma(4.0, self.switch_mean_s / 4.0))

    def activate(self, profile: str, switch_s: Optional[float] = None) -> dict:
        with self.lock:
            self._advance()
            if profile not in self.prepared:
                self.prepared.append(profile)
                self.engines.setdefault(profile, self._new_engine())
            changed = profile != self.active
            st = (self.switch_time() if switch_s is None else switch_s) if changed else 0.0
            self.active = profile
            self.last_switch_s = st
            self.activations.append((self.now_s(), profile))
            return {"ok": True, "profile": profile, "switch_time_s": st, "switched": changed,
                    "endpoint": self.endpoint_of(profile)}


    def set_config(self, cfg: dict) -> dict:
        with self.lock:
            self._advance()
            self.cfg = dict(cfg)
            self.configs.append((self.now_s(), dict(cfg)))
            return {"ok": True, "block_id": cfg.get("block_id"), "arm_id": cfg.get("arm_id")}

    def state(self) -> dict:
        with self.lock:
            self._advance()
            return {"queue_len": self.queue, "in_flight": self.in_flight, "config": self.cfg,
                    "counters": {"arrived": self.arrived, "admitted": self.admitted, "completed": self.completed,
                                 "escalated": self.escalated, "errors": self.errors},
                    "recent": {"n": self.completed, "mean_ms": self.last_latency_ms,
                               "p95_ms": 1.8 * self.last_latency_ms}}


    @staticmethod
    def _new_engine() -> dict:
        return {"prompt": 0.0, "gen": 0.0, "done": 0.0, "e2e_sum": 0.0, "ttft_sum": 0.0, "preempt": 0.0}

    def metrics_text(self, profile: str) -> str:
        with self.lock:
            self._advance()
            e = self.engines.setdefault(profile, self._new_engine())
            active = profile == self.active
            running = self.in_flight if active else 0
            waiting = self.queue if active else 0
            return "\n".join([
                "# HELP vllm:num_requests_running mock",
                f"vllm:num_requests_running{{engine=\"0\",model_name=\"{profile}\"}} {running}",
                f"vllm:num_requests_waiting{{engine=\"0\",model_name=\"{profile}\"}} {waiting}",
                f"vllm:kv_cache_usage_perc{{engine=\"0\",model_name=\"{profile}\"}} {min(0.95, 0.05 * running):.4f}",
                f"vllm:prompt_tokens_total{{engine=\"0\",model_name=\"{profile}\"}} {e['prompt']:.1f}",
                f"vllm:generation_tokens_total{{engine=\"0\",model_name=\"{profile}\"}} {e['gen']:.1f}",
                f"vllm:request_success_total{{engine=\"0\",finished_reason=\"stop\"}} {e['done']:.1f}",
                f"vllm:num_preemptions_total{{engine=\"0\"}} {e['preempt']:.1f}",
                f"vllm:e2e_request_latency_seconds_sum{{engine=\"0\"}} {e['e2e_sum']:.4f}",
                f"vllm:e2e_request_latency_seconds_count{{engine=\"0\"}} {e['done']:.1f}",
                f"vllm:e2e_request_latency_seconds_bucket{{engine=\"0\",le=\"1.0\"}} {e['done'] * 0.5:.1f}",
                f"vllm:e2e_request_latency_seconds_bucket{{engine=\"0\",le=\"+Inf\"}} {e['done']:.1f}",
                f"vllm:time_to_first_token_seconds_sum{{engine=\"0\"}} {e['ttft_sum']:.4f}",
                f"vllm:time_to_first_token_seconds_count{{engine=\"0\"}} {e['done']:.1f}",
                "",
            ])


    def _capacity(self) -> float:
        if not self.cfg or not self.active:
            return 0.0
        cap = CAPACITY.get(self.active, 1.0) * float(self.cfg.get("batch_cap") or 8) / 8.0
        return cap

    def _advance(self) -> None:
        now = self.now_s()
        dt = now - self._t_last
        if dt <= 0:
            return
        self._t_last = now
        n = int(self.rng.poisson(self.rate * dt))
        self.arrived += n
        self.admitted += n
        cap = self._capacity()
        served = min(self.queue + n, int(self.rng.poisson(cap * dt))) if cap > 0 else 0
        self.queue = self.queue + n - served
        self.completed += served
        self.in_flight = min(int((self.cfg or {}).get("batch_cap") or 8), served + self.queue) if cap > 0 else 0
        if self.cfg and self.cfg.get("escalate_family2"):
            self.escalated += int(self.rng.binomial(n, 0.3)) if n else 0
        lat = (1000.0 / max(cap, 0.1)) * (1.0 + self.queue / 10.0)
        self.last_latency_ms = lat
        if self.active:
            e = self.engines.setdefault(self.active, self._new_engine())
            e["prompt"] += served * 120.0
            e["gen"] += served * 40.0
            e["done"] += served
            e["e2e_sum"] += served * lat / 1000.0
            e["ttft_sum"] += served * lat / 4000.0
