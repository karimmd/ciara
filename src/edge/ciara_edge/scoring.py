from __future__ import annotations

import importlib
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


def load_callable(spec: str) -> Callable:
    mod, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:callable', got {spec!r}")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def token_entropy(top_logprobs: Sequence[float], k: int) -> float:
    ps = [math.exp(lp) for lp in top_logprobs[:k]]
    rest = max(0.0, 1.0 - sum(ps))
    h = -sum(p * math.log(p) for p in ps if p > 0.0)
    if rest > 1e-12:
        h -= rest * math.log(rest)
    return min(1.0, max(0.0, h / math.log(k + 1)))


def mean_token_entropy(token_top: Sequence[Sequence[float]], k: int) -> float:
    if not token_top:
        return 0.0
    return sum(token_entropy(t, k) for t in token_top) / len(token_top)


class PrefillScorer:
    mode = "prefill"
    kind = "prefill"

    def __init__(self, fn: Optional[Callable] = None):
        self.fn = fn

    def request_fields(self) -> Dict[str, Any]:
        return {"max_tokens": 1, "stream": False, "n": 1, "logprobs": True, "top_logprobs": 1, "temperature": 0.0}

    def score(self, req_meta: Mapping[str, Any], first_token_logprob: float):
        if self.fn is not None:
            return self.fn(req_meta, first_token_logprob)
        return min(1.0, max(0.0, 1.0 - math.exp(first_token_logprob)))


class LogprobEntropyScorer:
    mode = "edge_pass"
    kind = "logprob"

    def __init__(self, top_k: int = 5, fn: Optional[Callable] = None):
        self.top_k = int(top_k)
        self.fn = fn

    def request_fields(self) -> Dict[str, Any]:
        return {"logprobs": True, "top_logprobs": self.top_k}

    def score(self, req_meta: Mapping[str, Any], reply: Mapping[str, Any]):
        if self.fn is not None:
            return self.fn(req_meta, reply)
        return mean_token_entropy(reply.get("token_top") or [], self.top_k)


class SemanticEntropyScorer:
    mode = "probe"
    kind = "semantic_entropy"

    def __init__(self, race_params: Any, fn: Optional[Callable] = None):
        self.params = race_params
        self.fn = fn

    def score(self, req_meta: Mapping[str, Any], samples: Sequence[str]):
        if self.fn is not None:
            return self.fn(req_meta, samples)
        from ciara_baselines.race import semantic_entropy
        return semantic_entropy(list(samples))


def make_scorer(gw_params: Mapping[str, Any], race_params: Any):
    kind = gw_params.get("score", "prefill")
    top_k = int(gw_params.get("top_k", 5))
    if kind == "prefill":
        return PrefillScorer()
    if kind == "logprob":
        return LogprobEntropyScorer(top_k)
    if kind == "semantic_entropy":
        return SemanticEntropyScorer(race_params)
    fn = load_callable(kind)
    mode = gw_params.get("score_mode", "edge_pass")
    if mode == "probe":
        return SemanticEntropyScorer(race_params, fn)
    if mode == "prefill":
        return PrefillScorer(fn)
    if mode == "edge_pass":
        return LogprobEntropyScorer(top_k, fn)
    raise ValueError(f"score_mode must be prefill, edge_pass or probe, got {mode!r}")


class PerplexityConfidenceScorer:
    mode = "edge_pass"
    kind = "perplexity"
    top_k = 1

    def __init__(self, fn: Optional[Callable] = None):
        self.fn = fn

    def request_fields(self) -> Dict[str, Any]:
        return {"logprobs": True, "top_logprobs": self.top_k}

    def score(self, req_meta: Mapping[str, Any], reply: Mapping[str, Any]):
        if self.fn is not None:
            return self.fn(req_meta, reply)
        from ciara_baselines.recserve import confidence_from_token_top
        return confidence_from_token_top(reply.get("token_top") or [])


class FirstTokenConfidenceScorer(PrefillScorer):
    kind = "first_token"

    def score(self, req_meta: Mapping[str, Any], first_token_logprob: float):
        if self.fn is not None:
            return self.fn(req_meta, first_token_logprob)
        from ciara_baselines.recserve import max_prob_confidence
        return max_prob_confidence(first_token_logprob)


def make_recserve_scorer(score: str):
    if score == "perplexity":
        return PerplexityConfidenceScorer()
    if score == "first_token":
        return FirstTokenConfidenceScorer()
    raise ValueError(f"RecServe score must be 'perplexity' or 'first_token', got {score!r}")
