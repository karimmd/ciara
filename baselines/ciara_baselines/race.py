from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .base import Config, Incumbent, Obs, req_family

try:
    import numpy as _np
except Exception:
    _np = None


SCORES = ("first_token", "semantic_entropy", "sep")


@dataclass(frozen=True)
class RaceParams:
    alpha: Optional[Mapping[str, float]] = None

    window: int = 100
    eta0: float = 0.1
    rho: float = 0.99
    eps: float = 0.01
    c_lo: float = 0.8
    c_hi: float = 1.2
    lo_frac: float = 0.5
    hi_frac: float = 1.5
    use_budget: bool = True

    tau_init: float = 0.5

    horizon: int = 2000

    per_family: bool = True
    escalable_families: Tuple[str, ...] = ("f2",)

    score: str = "first_token"
    prefill_top_logprobs: int = 5

    label_fraction: float = 1.0

    n_samples: int = 5
    sample_temperature: float = 0.7
    probe_max_tokens: Optional[int] = None

    reset_each_block: bool = True

    @staticmethod
    def from_table(table: Any, delta: float = 0.05, families: Sequence[str] = ("f1", "f2"), **kw: Any) -> "RaceParams":
        alpha = {}
        for f in families:
            a = table.reference_accuracy(f)
            if a is None:
                raise ValueError(f"reference accuracy missing for {f}")
            alpha[f] = 1.0 - (a - delta)
        return RaceParams(alpha=alpha, **kw)

    def __post_init__(self) -> None:
        if self.score not in SCORES:
            raise ValueError(f"RaceParams.score must be one of {SCORES}, got {self.score!r}")

    @staticmethod
    def from_router_params(d: Mapping[str, Any]) -> "RaceParams":
        d = dict(d)
        d.pop("probe", None)
        d["escalable_families"] = tuple(d.get("escalable_families", ("f1", "f2")))
        known = set(RaceParams.__dataclass_fields__)
        return RaceParams(**{k: v for k, v in d.items() if k in known})

    def alpha_for(self, family: str) -> float:
        if self.alpha is None:
            raise ValueError("RaceParams.alpha is required; use RaceParams.from_table() or pass a mapping")
        return float(self.alpha[family])

    def to_router_params(self) -> Dict[str, Any]:
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(self).items()}
        d["alpha"] = None if self.alpha is None else dict(self.alpha)
        d["probe"] = {"n": self.n_samples, "temperature": self.sample_temperature, "max_tokens": self.probe_max_tokens}
        return d


@dataclass(frozen=True)
class RaceDecision:
    step: int
    u: float
    tau: float
    tau_tilde: float
    mode: str
    b_t: float
    offload: bool


class RaceCalibrator:
    def __init__(self, alpha: float, params: RaceParams, horizon: Optional[int] = None):
        self.alpha = float(alpha)
        self.p = params
        self.T = int(horizon if horizon is not None else params.horizon)
        self.tau = params.tau_init
        self.eta = params.eta0
        self.b_rem = self.T * self.alpha
        self.t = 0
        self.win: Deque[int] = deque(maxlen=params.window)
        self._pending: Set[int] = set()
        self.n_cloud = 0
        self.n_edge = 0
        self.n_edge_err = 0

    def threshold(self) -> Tuple[float, float, str]:
        t = self.t + 1
        p = self.p
        if t > self.T:
            return self.tau, self.alpha, "normal"
        b_t = self.b_rem / (self.T - t + 1)
        if p.use_budget and b_t < p.lo_frac * self.alpha:
            return p.c_lo * self.tau, b_t, "conservative"
        if p.use_budget and b_t > p.hi_frac * self.alpha:
            return min(1.0 - p.eps, p.c_hi * self.tau), b_t, "aggressive"
        return self.tau, b_t, "normal"

    def decide(self, u: float) -> RaceDecision:
        tau_tilde, b_t, mode = self.threshold()
        t = self.t + 1
        offload = u > tau_tilde
        dec = RaceDecision(t, u, self.tau, tau_tilde, mode, b_t, offload)
        self.t = t
        if offload:
            self.n_cloud += 1
            self._complete()
        else:
            self.n_edge += 1
            self._pending.add(t)
        return dec

    def feedback(self, step: int, err: Optional[bool]) -> bool:
        if step not in self._pending:
            return False
        self._pending.discard(step)
        if err is not None:
            if err:
                self.b_rem -= 1.0
                self.n_edge_err += 1
            self.win.append(1 if err else 0)
        self._complete()
        return True

    def _complete(self) -> None:
        if self.win:
            e_hat = sum(self.win) / len(self.win)
            tau = self.tau - self.eta * (e_hat - self.alpha)
            self.tau = min(1.0 - self.p.eps, max(self.p.eps, tau))
        self.eta *= self.p.rho

    def state(self) -> Dict[str, Any]:
        return {"t": self.t, "tau": self.tau, "eta": self.eta, "b_rem": self.b_rem, "window": len(self.win),
                "window_error": (sum(self.win) / len(self.win)) if self.win else None,
                "n_edge": self.n_edge, "n_cloud": self.n_cloud, "n_edge_err": self.n_edge_err,
                "pending": len(self._pending)}


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def _last_json_object(text: str) -> Optional[Any]:
    dec = json.JSONDecoder()
    idx = 0
    last = None
    while True:
        i = text.find("{", idx)
        if i < 0:
            return last
        try:
            obj, end = dec.raw_decode(text, i)
            last = obj
            idx = end
        except ValueError:
            idx = i + 1


def _norm(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k).strip().lower(): _norm(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_norm(v) for v in x]
    if isinstance(x, str):
        return " ".join(x.strip().lower().split())
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


def canonical_form(text: str) -> str:
    t = _THINK.sub(" ", text)
    obj = _last_json_object(t)
    if obj is not None:
        return "J:" + json.dumps(_norm(obj), sort_keys=True, separators=(",", ":"))
    return "T:" + " ".join(t.lower().split())


def semantic_entropy(samples: Sequence[str], canon: Callable[[str], Any] = canonical_form) -> float:
    k = len(samples)
    if k < 2:
        return 0.0
    counts = Counter(canon(s) for s in samples)
    se = -sum((n / k) * math.log(n / k) for n in counts.values())
    return min(1.0, max(0.0, se / math.log(k)))


class SepProbe:
    def __init__(self, layers: Sequence[Tuple[Sequence[Any], Sequence[float]]]):
        if not layers:
            raise ValueError("a probe needs at least one layer")
        norm = []
        for i, (w, b) in enumerate(layers):
            last = i == len(layers) - 1
            rows = [list(w)] if last and w and not isinstance(w[0], (list, tuple)) else [list(r) for r in w]
            if len(rows) != len(b) and not (last and len(b) == 1):
                raise ValueError("inconsistent probe shapes")
            norm.append((rows, list(b)))
        if len(norm[-1][1]) != 1:
            raise ValueError("the last layer must have one output unit")
        for (w0, _), (w1, _) in zip(norm, norm[1:]):
            if len(w0) != len(w1[0]):
                raise ValueError("inconsistent probe shapes")
        self.layers = norm

    @staticmethod
    def one_hidden(w1: Sequence[Sequence[float]], b1: Sequence[float], w2: Sequence[float], b2: float) -> "SepProbe":
        return SepProbe([(w1, b1), (w2, [b2])])

    def __call__(self, h: Sequence[float]) -> float:
        x: Any = _np.asarray(h, dtype=float) if _np is not None else list(h)
        for i, (w, b) in enumerate(self.layers):
            if _np is not None:
                x = _np.asarray(w, dtype=float) @ x + _np.asarray(b, dtype=float)
                if i < len(self.layers) - 1:
                    x = _np.maximum(x, 0.0)
            else:
                x = [sum(a * c for a, c in zip(row, x)) + bias for row, bias in zip(w, b)]
                if i < len(self.layers) - 1:
                    x = [max(v, 0.0) for v in x]
        s = float(x[0])
        return 1.0 / (1.0 + math.exp(-s)) if s > -700 else 0.0


def first_token_score(first_token_logprob: float) -> float:
    return min(1.0, max(0.0, 1.0 - math.exp(min(0.0, float(first_token_logprob)))))


def build_prefill_request(req_meta: Mapping[str, Any], params: RaceParams) -> Dict[str, Any]:
    rid = str(req_meta["req_id"])
    body = {
        "model": req_meta.get("model"),
        "messages": req_meta["messages"],
        "max_tokens": 1,
        "temperature": 0.0,
        "seed": 0,
        "logprobs": True,
        "top_logprobs": params.prefill_top_logprobs,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"X-Ciara-Req": f"{rid}.prefill", "X-Ciara-Kind": "race_prefill", "X-Ciara-Parent": rid}
    return {"body": body, "headers": headers}


def score_from_prefill_response(response: Mapping[str, Any]) -> float:
    content = response["choices"][0]["logprobs"]["content"][0]
    lps = [content["logprob"]] + [t["logprob"] for t in content.get("top_logprobs", [])]
    return first_token_score(max(lps))


def score_request(req_meta: Mapping[str, Any], params: RaceParams) -> Dict[str, Any]:
    if params.score == "first_token":
        return build_prefill_request(req_meta, params)
    if params.score == "semantic_entropy":
        return build_probe_request(req_meta, params)
    raise NotImplementedError("score 'sep' needs an engine hook for hidden states (not available in the testbed)")


def _stable_seed(req_id: str) -> int:
    return int(hashlib.sha256(req_id.encode()).hexdigest()[:8], 16) % (2 ** 31)


def build_probe_request(req_meta: Mapping[str, Any], params: RaceParams) -> Dict[str, Any]:
    rid = str(req_meta["req_id"])
    cap = params.probe_max_tokens if params.probe_max_tokens is not None else int(req_meta.get("max_tokens", 512))
    body = {
        "model": req_meta.get("model"),
        "messages": req_meta["messages"],
        "n": params.n_samples,
        "temperature": params.sample_temperature,
        "seed": _stable_seed(rid),
        "max_tokens": cap,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"X-Ciara-Req": f"{rid}.probe", "X-Ciara-Kind": "race_probe", "X-Ciara-Parent": rid}
    return {"body": body, "headers": headers}


def probe_log_record(req_meta: Mapping[str, Any], usage: Mapping[str, Any], t_start_ns: int, t_done_ns: int,
                     u: float, params_kind: str = "semantic_entropy") -> Dict[str, Any]:
    return {
        "kind": "race_probe" if params_kind == "semantic_entropy" else "race_prefill",
        "req": str(req_meta["req_id"]), "family": str(req_meta.get("family")),
        "n": int(usage.get("n", 0)), "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)), "t_start_ns": int(t_start_ns),
        "t_done_ns": int(t_done_ns), "race_probe_s": (t_done_ns - t_start_ns) / 1e9, "u": u,
    }


FeedbackFn = Callable[[Mapping[str, Any], Any], Optional[bool]]
ScoreFn = Callable[[Mapping[str, Any]], float]


class RaceRouter:
    def __init__(self, params: RaceParams, feedback_fn: Optional[FeedbackFn] = None,
                 scorer: Optional[ScoreFn] = None, horizon: Optional[int] = None):
        if params.alpha is None:
            raise ValueError("RaceParams.alpha is required")
        self.p = params
        self.feedback_fn = feedback_fn
        self.scorer = scorer
        self.horizon = horizon if horizon is not None else params.horizon
        self.cal: Dict[str, RaceCalibrator] = {}
        self.pending: Dict[str, Tuple[str, int]] = {}
        self.events: Deque[Dict[str, Any]] = deque(maxlen=100000)

    def _calibrator(self, family: str) -> RaceCalibrator:
        key = family if self.p.per_family else "all"
        if not self.p.per_family and len(set(self.p.alpha.values())) > 1:
            raise ValueError("a pooled calibrator (per_family=False) needs the same alpha for every family")
        if key not in self.cal:
            self.cal[key] = RaceCalibrator(self.p.alpha_for(family), self.p, self.horizon)
        return self.cal[key]

    def _labelled(self, req_id: str) -> bool:
        if self.p.label_fraction >= 1.0:
            return True
        h = int(hashlib.sha256(req_id.encode()).hexdigest()[:8], 16) / 2 ** 32
        return h < self.p.label_fraction

    def route(self, req_meta: Mapping[str, Any], node_state: Mapping[str, Any]) -> str:
        fam = req_family(req_meta)
        if fam not in self.p.escalable_families:
            return "edge"
        u = req_meta.get("race_u")
        if u is None and self.scorer is not None:
            u = self.scorer(req_meta)
        if u is None:
            raise KeyError("req_meta['race_u'] is missing: the gateway must supply the uncertainty score before route()")
        u = min(1.0, max(0.0, float(u)))
        dec = self._calibrator(fam).decide(u)
        if not dec.offload:
            self.pending[str(req_meta["req_id"])] = (fam, dec.step)
        self.events.append({"req": str(req_meta["req_id"]), "family": fam, "step": dec.step, "u": u, "tau": dec.tau,
                            "tau_tilde": dec.tau_tilde, "mode": dec.mode, "b_t": dec.b_t,
                            "route": "cloud" if dec.offload else "edge", "score": self.p.score})
        return "cloud" if dec.offload else "edge"

    def after_edge(self, req_meta: Mapping[str, Any], edge_reply: Any, node_state: Mapping[str, Any]) -> str:
        rid = str(req_meta["req_id"])
        item = self.pending.get(rid)
        if item is None or self.feedback_fn is None:
            return "accept"
        del self.pending[rid]
        fam, step = item
        err: Optional[bool] = self.feedback_fn(req_meta, edge_reply) if self._labelled(rid) else None
        self._calibrator(fam).feedback(step, err)
        return "accept"

    def label(self, req_id: str, err: Optional[bool]) -> bool:
        item = self.pending.pop(str(req_id), None)
        if item is None:
            return False
        if not self._labelled(str(req_id)):
            err = None
        return self._calibrator(item[0]).feedback(item[1], err)

    def abandon(self, req_meta: Mapping[str, Any]) -> None:
        item = self.pending.pop(str(req_meta["req_id"]), None)
        if item is not None:
            self._calibrator(item[0]).feedback(item[1], None)

    def state(self) -> Dict[str, Any]:
        return {"score": self.p.score, "escalable_families": list(self.p.escalable_families),
                "calibrators": {k: c.state() for k, c in self.cal.items()}}


class RaceIncumbent(Incumbent):
    name = "I2"

    def __init__(self, base_config: Config, params: RaceParams, feedback_fn: Optional[FeedbackFn] = None,
                 scorer: Optional[ScoreFn] = None, name: str = "I2"):
        self.name = name
        self.base = base_config
        self.p = params
        self.feedback_fn = feedback_fn
        self.scorer = scorer
        self.router = RaceRouter(params, feedback_fn, scorer)

    def reset(self, block_id: Optional[int] = None, horizon: Optional[int] = None, **kwargs: Any) -> None:
        if self.p.reset_each_block or horizon is not None:
            self.router = RaceRouter(self.p, self.feedback_fn, self.scorer, horizon)

    def decide(self, obs: Obs) -> Config:
        b = self.base
        return Config(
            arm_id=self.name, profile=b.profile, batch_cap=b.batch_cap, escalate_family2=False, router="I2",
            block_id=obs.block_id, router_params=self.p.to_router_params(), base_arm=b.base_arm,
        )

    def route(self, req_meta: Mapping[str, Any], node_state: Mapping[str, Any]) -> str:
        return self.router.route(req_meta, node_state)

    def after_edge(self, req_meta: Mapping[str, Any], edge_reply: Any, node_state: Mapping[str, Any]) -> str:
        return self.router.after_edge(req_meta, edge_reply, node_state)

    def diagnostics(self) -> Dict[str, Any]:
        return {"regime": self.name, **self.router.state()}
