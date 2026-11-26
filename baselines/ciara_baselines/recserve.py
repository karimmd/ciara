from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import Config, Incumbent, Obs, req_family

SCORES = ("perplexity", "first_token")


@dataclass(frozen=True)
class RecServeParams:
    beta: float = 0.3
    window: int = 10000
    min_history: int = 2

    score: str = "perplexity"

    per_family: bool = True
    escalable_families: Tuple[str, ...] = ("f2",)

    def __post_init__(self) -> None:
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must lie in (0, 1)")
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if self.min_history < 1:
            raise ValueError("min_history must be at least 1")
        if self.score not in SCORES:
            raise ValueError(f"RecServeParams.score must be one of {SCORES}, got {self.score!r}")

    @staticmethod
    def from_router_params(d: Mapping[str, Any]) -> "RecServeParams":
        d = dict(d)
        d["escalable_families"] = tuple(d.get("escalable_families", ("f2",)))
        known = set(RecServeParams.__dataclass_fields__)
        return RecServeParams(**{k: v for k, v in d.items() if k in known})

    def to_router_params(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(self).items()}


def perplexity(token_logprobs: Sequence[float]) -> float:
    n = len(token_logprobs)
    if n == 0:
        raise ValueError("perplexity needs at least one token")
    nll = -sum(float(v) for v in token_logprobs) / n
    try:
        return math.exp(nll)
    except OverflowError:
        return float("inf")


def confidence_from_logprobs(token_logprobs: Sequence[float]) -> Optional[float]:
    if len(token_logprobs) == 0:
        return None
    ppl = perplexity(token_logprobs)
    return 0.0 if math.isinf(ppl) else 1.0 / (1.0 + ppl)


def confidence_from_token_top(token_top: Sequence[Sequence[float]]) -> Optional[float]:
    lps = [float(t[0]) for t in token_top if t]
    return confidence_from_logprobs(lps)


def max_prob_confidence(first_token_logprob: float) -> float:
    return math.exp(min(0.0, float(first_token_logprob)))


def confidence_from_prefill_response(response: Mapping[str, Any]) -> float:
    content = response["choices"][0]["logprobs"]["content"][0]
    lps = [content["logprob"]] + [t["logprob"] for t in content.get("top_logprobs", [])]
    return max_prob_confidence(max(lps))


class ConfidenceQueue:
    def __init__(self, capacity: int):
        self.k = int(capacity)
        self._fifo: Deque[float] = deque()
        self._sorted: List[float] = []

    def __len__(self) -> int:
        return len(self._fifo)

    def push(self, c: float) -> None:
        c = float(c)
        if len(self._fifo) >= self.k:
            old = self._fifo.popleft()
            del self._sorted[bisect.bisect_left(self._sorted, old)]
        self._fifo.append(c)
        bisect.insort(self._sorted, c)

    def quantile(self, beta: float) -> float:
        n = len(self._sorted)
        if n == 0:
            raise ValueError("empty queue")
        r = beta * (n - 1)
        lo = int(math.floor(r))
        hi = int(math.ceil(r))
        frac = r - lo
        return self._sorted[lo] * (1.0 - frac) + self._sorted[hi] * frac

    def scores(self) -> List[float]:
        return list(self._fifo)


@dataclass(frozen=True)
class RecServeDecision:
    step: int
    c: float
    threshold: Optional[float]
    n_hist: int
    escalate: bool


class RecServeNode:
    def __init__(self, params: RecServeParams):
        self.p = params
        self.queue = ConfidenceQueue(params.window)
        self.t = 0
        self.n_escalated = 0

    def step(self, c: float) -> RecServeDecision:
        self.t += 1
        self.queue.push(c)
        n = len(self.queue)
        if n < self.p.min_history:
            return RecServeDecision(self.t, float(c), None, n, False)
        thr = self.queue.quantile(self.p.beta)
        esc = float(c) < thr
        if esc:
            self.n_escalated += 1
        return RecServeDecision(self.t, float(c), thr, n, esc)

    def state(self) -> Dict[str, Any]:
        n = len(self.queue)
        thr = self.queue.quantile(self.p.beta) if n >= self.p.min_history else None
        return {"t": self.t, "n_hist": n, "threshold": thr, "n_escalated": self.n_escalated,
                "escalated_share": (self.n_escalated / self.t) if self.t else None}


class RecServeRouter:
    def __init__(self, params: RecServeParams):
        self.p = params
        self.nodes: Dict[str, RecServeNode] = {}
        self.events: Deque[Dict[str, Any]] = deque(maxlen=100000)
        self.n_no_score = 0
        self.last: Optional[Dict[str, Any]] = None

    def _node(self, family: str) -> RecServeNode:
        key = family if self.p.per_family else "all"
        if key not in self.nodes:
            self.nodes[key] = RecServeNode(self.p)
        return self.nodes[key]

    def decide(self, req_meta: Mapping[str, Any], confidence: float) -> RecServeDecision:
        fam = req_family(req_meta)
        dec = self._node(fam).step(float(confidence))
        eligible = fam in self.p.escalable_families
        if dec.escalate and not eligible:
            dec = RecServeDecision(dec.step, dec.c, dec.threshold, dec.n_hist, False)
        self.last = {"req": str(req_meta.get("req_id")), "family": fam, "step": dec.step, "c": dec.c,
                     "threshold": dec.threshold, "n_hist": dec.n_hist, "eligible": eligible,
                     "route": "cloud" if dec.escalate else "edge", "score": self.p.score}
        self.events.append(self.last)
        return dec

    def route(self, req_meta: Mapping[str, Any], node_state: Mapping[str, Any]) -> str:
        if self.p.score != "first_token":
            return "edge"
        c = req_meta.get("recserve_c")
        if c is None:
            self.n_no_score += 1
            self.last = {"req": str(req_meta.get("req_id")), "route": "edge", "reason": "no_score"}
            return "edge"
        return "cloud" if self.decide(req_meta, float(c)).escalate else "edge"

    def after_edge(self, req_meta: Mapping[str, Any], edge_reply: Any, node_state: Mapping[str, Any]) -> str:
        if self.p.score != "perplexity":
            return "accept"
        c = None
        if isinstance(edge_reply, Mapping):
            c = edge_reply.get("confidence")
            if c is None and edge_reply.get("token_top"):
                c = confidence_from_token_top(edge_reply["token_top"])
        if c is None:
            self.n_no_score += 1
            self.last = {"req": str(req_meta.get("req_id")), "route": "edge", "reason": "no_score"}
            return "accept"
        return "escalate" if self.decide(req_meta, float(c)).escalate else "accept"

    def state(self) -> Dict[str, Any]:
        return {"score": self.p.score, "beta": self.p.beta, "window": self.p.window,
                "escalable_families": list(self.p.escalable_families), "n_no_score": self.n_no_score,
                "queues": {k: n.state() for k, n in self.nodes.items()}}


class RecServeIncumbent(Incumbent):
    name = "RecServe"

    def __init__(self, base_config: Config, params: RecServeParams, name: str = "RecServe"):
        self.name = name
        self.base = base_config
        self.p = params
        self.router = RecServeRouter(params)

    def reset(self, block_id: Optional[int] = None, **kwargs: Any) -> None:
        self.router = RecServeRouter(self.p)

    def decide(self, obs: Obs) -> Config:
        b = self.base
        return Config(
            arm_id=self.name, profile=b.profile, batch_cap=b.batch_cap, escalate_family2=False, router="RecServe",
            block_id=obs.block_id, router_params=self.p.to_router_params(), base_arm=b.base_arm,
        )

    def route(self, req_meta: Mapping[str, Any], node_state: Mapping[str, Any]) -> str:
        return self.router.route(req_meta, node_state)

    def after_edge(self, req_meta: Mapping[str, Any], edge_reply: Any, node_state: Mapping[str, Any]) -> str:
        return self.router.after_edge(req_meta, edge_reply, node_state)

    def diagnostics(self) -> Dict[str, Any]:
        return {"regime": self.name, **self.router.state()}
