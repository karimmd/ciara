from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import Config, Incumbent, Obs, arm_id
from .profile import (
    ADMISSIBLE_PROFILES,
    MAX_SEQ_TOKENS,
    NOMINAL_BUDGET_GIB,
    ProfileRow,
    ProfileTable,
    build_load_curve,
)

INF = float("inf")


@dataclass(frozen=True)
class I1Params:
    families: Tuple[str, ...] = ("f1", "f2")
    escalated_family: str = "f2"

    profiles: Tuple[str, ...] = ADMISSIBLE_PROFILES
    batch_caps: Tuple[int, ...] = (8, 16)
    kappas: Tuple[int, ...] = (0, 1)
    max_seq_tokens: int = MAX_SEQ_TOKENS

    delta: float = 0.05

    latency_targets_s: Optional[Mapping[str, float]] = None
    target_offset_s: Mapping[str, float] = field(default_factory=lambda: {"f1": 0.0, "f2": 0.0})

    memory_reserve_gib: float = 0.0
    stability_margin: float = 1.0

    tie_margin: float = 0.01

    family_mix: Mapping[str, float] = field(default_factory=lambda: {"f1": 0.5, "f2": 0.5})

    backhaul_rtt_s: float = 0.05
    backhaul_mbps: float = 100.0
    bytes_per_token: float = 4.0
    cloud_load_multiplier: float = 1.0

    epoch_s: float = 30.0
    reselect_every_epochs: int = 1

    design_load: Optional[float] = None
    design_memory_budget_gib: float = NOMINAL_BUDGET_GIB

    objective: str = "latency"


@dataclass(frozen=True)
class Candidate:
    profile: str
    batch_cap: int
    kappa: int
    index: int

    @property
    def arm(self) -> str:
        return arm_id(self.profile, self.batch_cap, self.kappa)

    def key(self) -> Tuple[str, int, int]:
        return (self.profile, self.batch_cap, self.kappa)


@dataclass
class Evaluation:
    cand: Candidate
    mem_gib: float
    mem_ok: bool
    acc_ok: bool
    lat_ok: bool
    stable: bool
    profiled: bool
    throughput: float
    mean_latency: float
    p95_norm: float
    mean_by_family: Dict[str, float]
    p95_by_family: Dict[str, float]
    accuracy: float
    reasons: Tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return self.profiled and self.mem_ok and self.acc_ok and self.lat_ok and self.stable


@dataclass
class Selection:
    chosen: Candidate
    evaluation: Evaluation
    feasible: List[Evaluation]
    evaluations: List[Evaluation]
    reason: str


class I1Selector:
    def __init__(self, table: ProfileTable, params: I1Params):
        if params.latency_targets_s is None:
            raise ValueError("I1Params.latency_targets_s is required (the targets L1 and L2 are set before the campaign)")
        for f in params.families:
            if f not in params.latency_targets_s:
                raise ValueError(f"no latency target for family {f}")
            if table.reference_accuracy(f) is None:
                raise ValueError(f"the reference checkpoint {table.reference} has no accuracy for family {f}")
        if params.objective not in ("latency", "throughput"):
            raise ValueError("objective must be 'latency' or 'throughput'")
        if any(k == 1 for k in params.kappas) and table.cloud is None:
            raise ValueError("kappa = 1 needs a cloud row in the profile table")
        self.table = table
        self.p = params
        self._cands = self._build_grid()

    def _build_grid(self) -> List[Candidate]:
        cands: List[Candidate] = []
        for ckpt in self.p.profiles:
            row = self.table.edge.get(ckpt)
            if row is None:
                continue
            for b in self.p.batch_caps:
                if not row.admissible(b, self.p.max_seq_tokens):
                    continue
                for k in self.p.kappas:
                    cands.append(Candidate(ckpt, b, k, len(cands)))
        if not cands:
            raise ValueError("empty candidate grid")
        return cands

    def candidates(self) -> List[Candidate]:
        return list(self._cands)

    def rates(self, arrival_rate: float, by_family: Optional[Mapping[str, float]]) -> Dict[str, float]:
        if by_family:
            return {f: max(float(by_family.get(f, 0.0)), 0.0) for f in self.p.families}
        mix = {f: self.p.family_mix.get(f, 0.0) for f in self.p.families}
        z = sum(mix.values())
        return {f: max(arrival_rate, 0.0) * mix[f] / z for f in self.p.families}

    def _shares(self, rates: Mapping[str, float]) -> Tuple[float, Dict[str, float]]:
        total = sum(rates.values())
        if total > 0:
            return total, {f: rates[f] / total for f in self.p.families}
        mix = {f: self.p.family_mix.get(f, 0.0) for f in self.p.families}
        z = sum(mix.values())
        return 0.0, {f: mix[f] / z for f in self.p.families}

    def evaluate(self, cand: Candidate, rates: Mapping[str, float], memory_budget_gib: float) -> Evaluation:
        p = self.p
        row = self.table.edge[cand.profile]
        total, shares = self._shares(rates)
        esc = p.escalated_family
        cloud_fams = [esc] if cand.kappa == 1 and esc in p.families else []
        edge_fams = [f for f in p.families if f not in cloud_fams]
        edge_share = sum(shares[f] for f in edge_fams)
        cloud_share = sum(shares[f] for f in cloud_fams)
        reasons: List[str] = []

        mem = row.mem_gib(cand.batch_cap, p.max_seq_tokens)
        mem_ok = mem <= memory_budget_gib - p.memory_reserve_gib + 1e-9
        if not mem_ok:
            reasons.append("memory")

        profiled = True
        acc_ok = True
        acc_mix = 0.0
        site_row: Dict[str, ProfileRow] = {f: row for f in edge_fams}
        for f in cloud_fams:
            site_row[f] = self.table.cloud
        for f in p.families:
            fp = site_row[f].families.get(f)
            if fp is None or fp.accuracy is None or not fp.has_workload():
                profiled = False
                reasons.append(f"unprofiled:{f}@{site_row[f].ckpt}")
                continue
            ref = self.table.reference_accuracy(f)
            if fp.accuracy < ref - p.delta - 1e-12:
                acc_ok = False
                reasons.append(f"accuracy:{f}")
            acc_mix += shares[f] * fp.accuracy
        if not profiled:
            return Evaluation(cand, mem, mem_ok, False, False, False, False, 0.0, INF, INF, {}, {}, acc_mix,
                              tuple(reasons))

        sites: List[Tuple[str, Any, Dict[str, float], float, float]] = []
        t_total = INF
        if edge_share > 0:
            curves = {f: row.curve(f, cand.batch_cap) for f in edge_fams if shares[f] > 0}
            mix_e = {f: shares[f] / edge_share for f in curves}
            lc = build_load_curve(curves, mix_e)
            sites.append(("edge", lc, mix_e, total * edge_share, lc.max_load / edge_share))
            t_total = min(t_total, lc.max_load / edge_share)
        if cloud_share > 0:
            crow = self.table.cloud
            curves = {f: crow.curve(f, crow.max_batch) for f in cloud_fams if shares[f] > 0}
            mix_c = {f: shares[f] / cloud_share for f in curves}
            lc = build_load_curve(curves, mix_c)
            load_c = total * cloud_share * p.cloud_load_multiplier
            sites.append(("cloud", lc, mix_c, load_c, lc.max_load / (cloud_share * p.cloud_load_multiplier)))
            t_total = min(t_total, lc.max_load / (cloud_share * p.cloud_load_multiplier))
        stable = total <= p.stability_margin * t_total * (1.0 + 1e-12)
        if not stable:
            reasons.append("throughput")

        mean_f: Dict[str, float] = {}
        p95_f: Dict[str, float] = {}
        if stable:
            for site, lc, _mix, load, _t in sites:
                at = lc.at(load)
                assert at is not None
                for f, v in at.items():
                    if site == "cloud":
                        net = self._cloud_net_s(site_row[f], f)
                        mean_f[f] = v["mean"] + net
                        p95_f[f] = v["p95"] + net
                    else:
                        mean_f[f] = v["mean"]
                        p95_f[f] = v["p95"]

        lat_ok = stable
        p95_norm = INF
        mean_lat = INF
        if stable:
            norms = []
            for f in p.families:
                if shares[f] <= 0:
                    continue
                tgt = p.latency_targets_s[f] - p.target_offset_s.get(f, 0.0)
                norms.append(p95_f[f] / tgt if tgt > 0 else INF)
                if p95_f[f] > tgt + 1e-12:
                    lat_ok = False
                    reasons.append(f"latency:{f}")
            p95_norm = max(norms) if norms else 0.0
            mean_lat = sum(shares[f] * mean_f[f] for f in p.families if shares[f] > 0)
        return Evaluation(cand, mem, mem_ok, acc_ok, lat_ok, stable, True, t_total, mean_lat, p95_norm,
                          mean_f, p95_f, acc_mix, tuple(reasons))

    def _cloud_net_s(self, crow: ProfileRow, f: str) -> float:
        fp = crow.families[f]
        bits = 8.0 * self.p.bytes_per_token * (fp.prompt_tokens + (fp.out_tokens or 0.0))
        return self.p.backhaul_rtt_s + bits / (self.p.backhaul_mbps * 1e6)

    def select(self, rates: Mapping[str, float], memory_budget_gib: float) -> Selection:
        evals = [self.evaluate(c, rates, memory_budget_gib) for c in self._cands]
        feas = [e for e in evals if e.feasible]
        p = self.p
        if feas:
            if p.objective == "latency":
                best = min(e.mean_latency for e in feas)
                tie = [e for e in feas if e.mean_latency <= best * (1.0 + p.tie_margin) + 1e-15]
                reason = "argmin"
            else:
                best = max(e.throughput for e in feas)
                tie = [e for e in feas if e.throughput >= best * (1.0 - p.tie_margin) - 1e-15]
                reason = "argmax_throughput"
            pick = min(tie, key=lambda e: (e.mem_gib, -e.accuracy, e.cand.index))
            return Selection(pick.cand, pick, feas, evals, reason)
        cloud = [e for e in evals if e.cand.kappa == 1 and e.mem_ok and e.profiled]
        if cloud:
            pick = min(cloud, key=lambda e: (e.p95_norm, -e.throughput, e.cand.index))
            return Selection(pick.cand, pick, [], evals, "fallback_cloud")
        fits = [e for e in evals if e.mem_ok and e.profiled]
        if fits:
            pick = max(fits, key=lambda e: (e.accuracy, -e.cand.index))
            return Selection(pick.cand, pick, [], evals, "fallback_accuracy")
        pick = min(evals, key=lambda e: (e.mem_gib, e.cand.index))
        return Selection(pick.cand, pick, [], evals, "fallback_min_memory")


def _config(name: str, cand: Candidate, block_id: Optional[int]) -> Config:
    return Config(
        arm_id=name, profile=cand.profile, batch_cap=cand.batch_cap, escalate_family2=bool(cand.kappa),
        router="kappa", block_id=block_id, base_arm=cand.arm,
    )


class I1E(Incumbent):
    name = "I1-E"

    def __init__(self, table: ProfileTable, params: I1Params):
        self.selector = I1Selector(table, params)
        self.p = params
        self.reset()

    def reset(self, block_id: Optional[int] = None, **kwargs: Any) -> None:
        self._current: Optional[Candidate] = None
        self._since = 10 ** 9
        self._last_budget: Optional[float] = None
        self.last: Dict[str, Any] = {}

    def decide(self, obs: Obs) -> Config:
        p = self.p
        due = (
            self._current is None
            or self._since >= p.reselect_every_epochs
            or (self._last_budget is not None and abs(obs.memory_budget_gib - self._last_budget) > 1e-9)
        )
        if not due:
            self._since += 1
            return _config(self.name, self._current, obs.block_id)
        rates = self.selector.rates(obs.arrival_rate, obs.arrival_rate_by_family)
        sel = self.selector.select(rates, obs.memory_budget_gib)
        chosen, why = self._guard(sel, rates, obs.memory_budget_gib)
        self._current = chosen
        self._since = 1
        self._last_budget = obs.memory_budget_gib
        self.last = {
            "regime": self.name, "epoch": obs.epoch, "arrival_rate": obs.arrival_rate, "arm": chosen.arm,
            "proposed": sel.chosen.arm, "selection_reason": sel.reason, "guard": why,
            "feasible_count": len(sel.feasible), "grid_size": len(sel.evaluations),
            "mean_latency_s": sel.evaluation.mean_latency, "p95_norm": sel.evaluation.p95_norm,
        }
        return _config(self.name, chosen, obs.block_id)

    def _guard(self, sel: Selection, rates: Mapping[str, float], budget: float) -> Tuple[Candidate, str]:
        cur = self._current
        if cur is None:
            return sel.chosen, "init"
        if cur.key() == sel.chosen.key():
            return cur, "same"
        ev_cur = self.selector.evaluate(cur, rates, budget)
        if not ev_cur.feasible:
            return sel.chosen, "forced_current_infeasible"
        ev_new = sel.evaluation
        if self.p.objective == "latency":
            gain = (ev_cur.mean_latency - ev_new.mean_latency) / ev_cur.mean_latency if ev_cur.mean_latency > 0 else 0.0
        else:
            gain = (ev_new.throughput - ev_cur.throughput) / ev_cur.throughput if ev_cur.throughput > 0 else 0.0
        cost = self.selector.table.overhead.switch_time_s(cur.profile, sel.chosen.profile)
        if gain > 0 and gain * self.p.epoch_s > cost:
            return sel.chosen, "switched"
        return cur, "kept_by_guard"

    def diagnostics(self) -> Dict[str, Any]:
        return dict(self.last)


class I1T(I1E):
    name = "I1-T"

    def __init__(self, table: ProfileTable, params: I1Params):
        super().__init__(table, _with(params, objective="throughput"))


class I1Oracle(I1E):
    name = "I1-oracle"


class I1S(Incumbent):
    name = "I1-S"

    def __init__(self, table: ProfileTable, params: I1Params):
        if params.design_load is None:
            raise ValueError("I1S needs I1Params.design_load (the design load lambda_0 of the run)")
        self.selector = I1Selector(table, params)
        self.p = params
        rates = self.selector.rates(params.design_load, None)
        self.selection = self.selector.select(rates, params.design_memory_budget_gib)

    def decide(self, obs: Obs) -> Config:
        return _config(self.name, self.selection.chosen, obs.block_id)

    def base_config(self) -> Config:
        return _config(self.name, self.selection.chosen, None)

    def diagnostics(self) -> Dict[str, Any]:
        s = self.selection
        return {"regime": self.name, "arm": s.chosen.arm, "selection_reason": s.reason,
                "feasible_count": len(s.feasible), "design_load": self.p.design_load}


def _with(params: I1Params, **kw: Any) -> I1Params:
    d = dict(vars(params))
    d.update(kw)
    return I1Params(**d)


def select_base_config(table: ProfileTable, params: I1Params) -> Config:
    return I1S(table, params).base_config()
