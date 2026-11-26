from __future__ import annotations

import importlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import Config, Incumbent, Obs, arm_id, family_key
from .lp import solve_distribution_lp
from .profile import ADMISSIBLE_PROFILES, MAX_SEQ_TOKENS, ProfileTable

REWARD_KEY = "Y_Q"
TIE_BREAK = 1e-7
DEFAULT_CONSTRAINTS: Dict[str, float] = {"Y_L_f1": 0.05, "Y_L_f2": 0.05, "Y_D": 0.05, "Y_C": 0.30}


@dataclass(frozen=True)
class CBanditParams:
    reward: str = REWARD_KEY
    constraints: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_CONSTRAINTS))

    gamma: Optional[float] = 1.0
    confidence_delta: float = 0.05
    horizon_epochs: int = 1200

    epsilon: float = 0.0

    profiles: Tuple[str, ...] = ADMISSIBLE_PROFILES
    batch_caps: Tuple[int, ...] = (8, 16)
    kappas: Tuple[int, ...] = (0, 1)
    max_seq_tokens: int = MAX_SEQ_TOKENS

    seed: int = 0
    reset_each_block: bool = False

    def __post_init__(self) -> None:
        if not self.constraints:
            raise ValueError("at least one constraint is required")
        for k, v in self.constraints.items():
            if not v > 0.0:
                raise ValueError(f"budget of {k} must be positive")
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError("epsilon must lie in [0, 1)")
        if self.gamma is not None and not self.gamma > 0.0:
            raise ValueError("gamma must be positive")
        if self.reward in self.constraints:
            raise ValueError("the reward outcome cannot also carry a budget")


@dataclass(frozen=True)
class Arm:
    profile: str
    batch_cap: int
    kappa: int
    index: int

    @property
    def arm(self) -> str:
        return arm_id(self.profile, self.batch_cap, self.kappa)


def admissible_grid(table: ProfileTable, params: CBanditParams) -> List[Arm]:
    arms: List[Arm] = []
    for ckpt in params.profiles:
        row = table.edge.get(ckpt)
        if row is None:
            continue
        for b in params.batch_caps:
            if not row.admissible(b, params.max_seq_tokens):
                continue
            for k in params.kappas:
                arms.append(Arm(ckpt, b, k, len(arms)))
    if not arms:
        raise ValueError("empty arm set")
    return arms


def paper_gamma(m: int, horizon: int, d: int, delta: float) -> float:
    return math.log(m * horizon * d / delta)


def rad(nu: float, n: float, gamma: float) -> float:
    return math.sqrt(gamma * nu / n) + gamma / n


class CBanditCore:
    def __init__(self, m: int, reward: str, budgets: Mapping[str, float], gamma: float, epsilon: float = 0.0,
                 rng: Optional[random.Random] = None, tie_break: float = TIE_BREAK):
        self.m = int(m)
        self.tie_break = float(tie_break)
        self.reward = reward
        self.names: List[str] = list(budgets)
        self.budgets: List[float] = [float(budgets[k]) for k in self.names]
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.rng = rng if rng is not None else random.Random(0)
        keys = [reward] + self.names
        self.sum: Dict[str, List[float]] = {k: [0.0] * self.m for k in keys}
        self.cnt: Dict[str, List[int]] = {k: [0] * self.m for k in keys}
        self.plays = [0] * self.m
        self.last_status = "none"

    def mean(self, key: str, i: int) -> float:
        return self.sum[key][i] / (self.cnt[key][i] + 1.0)

    def ucb(self, key: str, i: int) -> float:
        v = self.mean(key, i)
        return min(1.0, v + 2.0 * rad(v, self.cnt[key][i] + 1.0, self.gamma))

    def lcb(self, key: str, i: int) -> float:
        v = self.mean(key, i)
        return max(0.0, v - 2.0 * rad(v, self.cnt[key][i] + 1.0, self.gamma))

    def observe(self, i: int, outcomes: Mapping[str, Optional[float]]) -> int:
        n = 0
        for key in self.sum:
            v = outcomes.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            self.sum[key][i] += min(1.0, max(0.0, float(v)))
            self.cnt[key][i] += 1
            n += 1
        if n:
            self.plays[i] += 1
        return n

    def bounds(self) -> Tuple[List[float], List[List[float]]]:
        r = [self.ucb(self.reward, i) for i in range(self.m)]
        a = [[self.lcb(k, i) for i in range(self.m)] for k in self.names]
        return r, a

    def solve(self) -> Tuple[List[float], str]:
        r, a = self.bounds()
        b = [(1.0 - self.epsilon) * c for c in self.budgets]
        res = solve_distribution_lp([r[i] + self.tie_break * self.rng.random() for i in range(self.m)], a, b)
        if res.status == "optimal" and res.p is not None:
            self.last_status = "optimal"
            return res.p, "optimal"
        viol = [max(a[j][i] / self.budgets[j] for j in range(len(self.names))) for i in range(self.m)]
        best = min(range(self.m), key=lambda i: (viol[i], -r[i], i))
        p = [0.0] * self.m
        p[best] = 1.0
        self.last_status = "fallback"
        return p, "fallback"

    def sample(self, p: Sequence[float]) -> int:
        u = self.rng.random()
        acc = 0.0
        for i, pi in enumerate(p):
            acc += pi
            if u < acc:
                return i
        return max(range(len(p)), key=lambda i: p[i])

    def step(self) -> Tuple[int, List[float], str]:
        p, status = self.solve()
        return self.sample(p), p, status


def epoch_outcomes(records: Sequence[Mapping[str, Any]], latency_targets_s: Mapping[str, float],
                   quality_fn: Optional[Callable[[Mapping[str, Any]], Optional[float]]] = None,
                   target_offset_s: Optional[Mapping[str, float]] = None,
                   energy_fn: Optional[Callable[[Mapping[str, Any]], Optional[float]]] = None) -> Dict[str, Any]:
    offs = dict(target_offset_s or {})
    n = 0
    fam_n: Dict[str, int] = {}
    fam_x: Dict[str, int] = {}
    n_dl = 0
    n_cloud = 0
    qs: List[float] = []
    es: Dict[str, List[float]] = {}
    for r in records:
        fam = family_key(r["family"])
        ok = r.get("status") == 200 and not r.get("error")
        t0, t1 = r.get("t_arrive"), r.get("t_done")
        lat = (t1 - t0) / 1e9 if (ok and t0 is not None and t1 is not None) else None
        tgt = float(latency_targets_s[fam]) - float(offs.get(fam, 0.0))
        exceed = (not ok) or (lat is not None and lat > tgt)
        dl_ms = r.get("deadline_ms")
        late = (not ok) or (lat is not None and dl_ms is not None and lat * 1000.0 > float(dl_ms) - 1000.0 * float(offs.get(fam, 0.0)))
        n += 1
        fam_n[fam] = fam_n.get(fam, 0) + 1
        fam_x[fam] = fam_x.get(fam, 0) + (1 if exceed else 0)
        n_dl += 1 if late else 0
        n_cloud += 1 if r.get("served_by") == "cloud" else 0
        if quality_fn is not None:
            q = 0.0 if not ok else quality_fn(r)
            if q is not None:
                qs.append(float(q))
        if energy_fn is not None:
            e = energy_fn(r)
            cls = r.get("class")
            if e is not None and cls is not None:
                es.setdefault(str(cls), []).append(min(1.0, max(0.0, float(e))))
    out: Dict[str, Any] = {"n": n}
    out[REWARD_KEY] = (sum(qs) / len(qs)) if qs else None
    for fam in sorted(latency_targets_s):
        out["n_" + fam] = fam_n.get(fam, 0)
        out["Y_L_" + fam] = (fam_x.get(fam, 0) / fam_n[fam]) if fam_n.get(fam) else None
    out["Y_D"] = (n_dl / n) if n else None
    out["Y_C"] = (n_cloud / n) if n else None
    for cls, v in sorted(es.items()):
        out["Y_E_" + cls] = sum(v) / len(v)
    return out


_SUITE: Dict[str, Any] = {}


def suite_quality(record: Mapping[str, Any]) -> Optional[float]:
    from ciara_analysis import quality as q

    if "scorers" not in _SUITE:
        _SUITE["scorers"] = q.default_scorers()
    ok = record.get("status") == 200 and not record.get("error")
    val, _ = q.score_request(_SUITE["scorers"], int(record["family"]), str(record.get("item")), record.get("reply_text"),
                             record.get("finish_reason"), ok)
    return val


def load_callable(spec: str) -> Callable:
    mod, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:callable', got {spec!r}")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


class GatewayOutcomeSource:
    def __init__(self, path_template: str, latency_targets_s: Mapping[str, float],
                 quality_fn: Optional[Callable[[Mapping[str, Any]], Optional[float]]] = None,
                 target_offset_s: Optional[Mapping[str, float]] = None):
        self.template = path_template
        self.targets = dict(latency_targets_s)
        self.quality_fn = quality_fn
        self.offsets = dict(target_offset_s or {})
        self._pos: Dict[str, int] = {}

    def path_for(self, node: str, node_index: Optional[int] = None) -> str:
        if node_index is not None:
            n: Any = int(node_index) + 1
        else:
            digits = "".join(ch for ch in str(node) if ch.isdigit())
            n = int(digits) if digits else node
        return self.template.format(node=node, n=n)

    def read_records(self, path: str) -> List[Dict[str, Any]]:
        try:
            size = os.path.getsize(path)
        except OSError:
            return []
        pos = self._pos.get(path, 0)
        if size < pos:
            pos = 0
        with open(path, "rb") as fh:
            fh.seek(pos)
            data = fh.read()
        end = data.rfind(b"\n")
        if end < 0:
            return []
        self._pos[path] = pos + end + 1
        recs: List[Dict[str, Any]] = []
        for line in data[: end + 1].splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and "family" in r and r.get("family") is not None and r.get("t_done") is not None:
                recs.append(r)
        return recs

    def __call__(self, obs: Obs) -> Optional[Dict[str, Any]]:
        tel = obs.telemetry or {}
        node = tel.get("node")
        if node is None:
            return None
        recs = self.read_records(self.path_for(str(node), tel.get("node_index")))
        return epoch_outcomes(recs, self.targets, self.quality_fn, self.offsets)


OutcomeSource = Callable[[Obs], Optional[Mapping[str, Any]]]


class CBanditIncumbent(Incumbent):
    name = "CB"

    def __init__(self, table: ProfileTable, params: CBanditParams, outcome_source: Optional[OutcomeSource] = None,
                 name: str = "CB"):
        self.name = name
        self.p = params
        self.arms = admissible_grid(table, params)
        self.source = outcome_source
        d = len(params.constraints)
        self.gamma = params.gamma if params.gamma is not None else paper_gamma(len(self.arms), params.horizon_epochs, d, params.confidence_delta)
        self._rng_node: Optional[str] = None
        self.core = CBanditCore(len(self.arms), params.reward, params.constraints, self.gamma, params.epsilon,
                                random.Random(f"{params.seed}:"))
        self._last: Optional[int] = None
        self.stats = {"epochs": 0, "updates": 0, "no_feedback": 0, "skipped_updates": 0, "switches": 0, "fallbacks": 0}
        self.last: Dict[str, Any] = {}
        self._prev_arm: Optional[int] = None

    def reset(self, block_id: Optional[int] = None, **kwargs: Any) -> None:
        self._last = None
        self._prev_arm = None
        if self.p.reset_each_block:
            self.core = CBanditCore(len(self.arms), self.p.reward, self.p.constraints, self.gamma, self.p.epsilon,
                                    random.Random(f"{self.p.seed}:{self._rng_node or ''}"))

    def _outcomes(self, obs: Obs) -> Optional[Mapping[str, Any]]:
        out = (obs.telemetry or {}).get("outcomes")
        if out is None and self.source is not None:
            out = self.source(obs)
        return out

    def decide(self, obs: Obs) -> Config:
        node = (obs.telemetry or {}).get("node")
        if self._rng_node is None and node is not None:
            self._rng_node = str(node)
            self.core.rng = random.Random(f"{self.p.seed}:{self._rng_node}")
        out = self._outcomes(obs)
        credited = 0
        if self._last is not None:
            if out is None:
                self.stats["no_feedback"] += 1
            else:
                credited = self.core.observe(self._last, out)
                if credited:
                    self.stats["updates"] += 1
                else:
                    self.stats["skipped_updates"] += 1
        i, p, status = self.core.step()
        if self._prev_arm is not None and i != self._prev_arm:
            self.stats["switches"] += 1
        if status == "fallback":
            self.stats["fallbacks"] += 1
        self.stats["epochs"] += 1
        self._last = i
        self._prev_arm = i
        arm = self.arms[i]
        self.last = {"regime": self.name, "epoch": obs.epoch, "arm": arm.arm, "lp_status": status,
                     "credited_components": credited,
                     "p": {self.arms[k].arm: round(pk, 6) for k, pk in enumerate(p) if pk > 1e-9},
                     "gamma": self.gamma, **self.stats}
        return Config(arm_id=self.name, profile=arm.profile, batch_cap=arm.batch_cap, escalate_family2=bool(arm.kappa),
                      router="kappa", block_id=obs.block_id, base_arm=arm.arm)

    def diagnostics(self) -> Dict[str, Any]:
        return dict(self.last)

    def pulls(self) -> Dict[str, int]:
        return {a.arm: self.core.plays[a.index] for a in self.arms}
