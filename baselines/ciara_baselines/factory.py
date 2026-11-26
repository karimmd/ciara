from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .base import Config, Incumbent, Obs, arm_id
from .cbandit import CBanditIncumbent, CBanditParams, GatewayOutcomeSource, load_callable
from .i1 import I1E, I1S, I1T, I1Oracle, I1Params, select_base_config
from .mixorsplit import MixOrSplitIncumbent, MosParams
from .profile import ProfileTable, build_table
from .race import RaceIncumbent, RaceParams
from .recserve import RecServeIncumbent, RecServeParams

NAMES = ("I1-E", "I1-S", "I1-T", "I1-oracle", "I2", "I2-all", "I3", "I3-all", "RecServe", "RecServe-all", "CB")
DEFAULT_TABLE = Path(__file__).resolve().parents[1] / "profiles" / "edge_profile_table.json"
BOTH = ("f1", "f2")


def load_table(path: Optional[str] = None) -> ProfileTable:
    p = Path(path) if path else DEFAULT_TABLE
    if p.exists():
        return ProfileTable.load(p)
    if path:
        raise FileNotFoundError(path)
    return build_table()


def _overrides(cls: Any, params: Optional[Mapping[str, Any]], extra: Mapping[str, Any]) -> Dict[str, Any]:
    out = {**(params or {}), **extra}
    known = {f.name for f in fields(cls)}
    bad = sorted(set(out) - known)
    if bad:
        raise ValueError(f"unknown keyword(s) for {cls.__name__}: {bad}")
    return out


class ControlAdapter:
    def __init__(self, inner: Incumbent, name: str, memory_utilization: float = 0.95,
                 default_rate: float = 0.0, horizon: Optional[int] = None):
        self.inner = inner
        self.name = name
        self.util = memory_utilization
        self.horizon = horizon
        self._last_rate = default_rate
        self.blocks: list = []

    def begin_block(self, info: Mapping[str, Any]) -> None:
        self.blocks.append(dict(info))
        kw: Dict[str, Any] = {}
        if self.horizon is not None:
            kw["horizon"] = self.horizon
        self.inner.reset(block_id=info.get("block_id"), **kw)

    def to_obs(self, obs: Mapping[str, Any]) -> Obs:
        rate = obs.get("arrival_rate")
        if rate is None:
            rate = self._last_rate
        else:
            self._last_rate = float(rate)
        mib = obs.get("memory_budget_mib")
        budget = float(mib) / 1024.0 * self.util if mib is not None else Obs().memory_budget_gib
        tel = {k: obs.get(k) for k in ("gateway", "engine", "profile", "node", "node_index", "context_id", "outcomes") if k in obs}
        return Obs(epoch=int(obs.get("epoch", 0)), arrival_rate=float(rate), memory_budget_gib=budget,
                   block_id=obs.get("block_id"), telemetry=tel)

    def decide(self, obs: Mapping[str, Any]) -> Dict[str, Any]:
        cfg: Config = self.inner.decide(self.to_obs(obs))
        d = cfg.to_dict()
        for k in ("arm_id", "block_id", "base_arm"):
            d.pop(k, None)
        return d

    def diagnostics(self) -> Dict[str, Any]:
        return self.inner.diagnostics()


def make_incumbent(name: str, base_profile: Optional[str] = None, base_batch_cap: Optional[int] = None,
                   table_path: Optional[str] = None, oracle_table_path: Optional[str] = None,
                   latency_targets_s: Optional[Mapping[str, float]] = None, design_load: Optional[float] = None,
                   delta: float = 0.05, memory_utilization: float = 0.95, horizon: Optional[int] = None,
                   params: Optional[Mapping[str, Any]] = None, outcome_log: Optional[str] = None,
                   quality_fn: Optional[str] = None, **extra: Any) -> ControlAdapter:
    if name not in NAMES:
        raise ValueError(f"unknown incumbent {name!r}; expected one of {NAMES}")
    table = load_table(table_path)
    targets = dict(latency_targets_s) if latency_targets_s else None
    rate0 = float(design_load) if design_load is not None else 0.0

    if name.startswith("I1"):
        kw = _overrides(I1Params, params, extra)
        if targets is None:
            raise ValueError(f"{name} needs latency_targets_s (the reference targets L1 and L2)")
        p = I1Params(latency_targets_s=targets, design_load=design_load, delta=delta, **kw)
        if name == "I1-S":
            inner: Incumbent = I1S(table, p)
        elif name == "I1-E":
            inner = I1E(table, p)
        elif name == "I1-T":
            inner = I1T(table, p)
        else:
            if not oracle_table_path:
                raise ValueError("I1-oracle needs oracle_table_path (a table re-measured under co-location)")
            inner = I1Oracle(load_table(oracle_table_path), p)
        return ControlAdapter(inner, name, memory_utilization, rate0, horizon)

    if name == "CB":
        kw = _overrides(CBanditParams, params, extra)
        source = None
        if outcome_log:
            if targets is None:
                raise ValueError("CB with outcome_log needs latency_targets_s (the reference targets L1 and L2)")
            qf = load_callable(quality_fn) if quality_fn else None
            source = GatewayOutcomeSource(outcome_log, targets, qf)
        inner = CBanditIncumbent(table, CBanditParams(**kw), source, name=name)
        return ControlAdapter(inner, name, memory_utilization, rate0, horizon)

    if base_profile is not None:
        if base_batch_cap is None:
            raise ValueError("base_batch_cap is required with base_profile")
        row = table.edge.get(base_profile)
        if row is None or not row.admissible(int(base_batch_cap)):
            raise ValueError(f"base configuration {base_profile} at batch cap {base_batch_cap} is not admissible")
        base = Config("I1-S", base_profile, int(base_batch_cap), False, base_arm=arm_id(base_profile, base_batch_cap, 0))
    else:
        if targets is None or design_load is None:
            raise ValueError(f"{name} needs base_profile and base_batch_cap, or latency_targets_s and design_load "
                             "so that I1-S selects the base")
        base = select_base_config(table, I1Params(latency_targets_s=targets, design_load=design_load, delta=delta))
    both = name.endswith("-all")
    if name.startswith("RecServe"):
        kw = _overrides(RecServeParams, params, extra)
        kw.setdefault("escalable_families", BOTH if both else ("f2",))
        return ControlAdapter(RecServeIncumbent(base, RecServeParams(**kw), name=name), name, memory_utilization, rate0,
                              horizon)
    if name.startswith("I2"):
        kw = _overrides(RaceParams, params, extra)
        kw.setdefault("escalable_families", BOTH if both else ("f2",))
        if "alpha" in kw:
            rparams = RaceParams(**kw)
        else:
            rparams = RaceParams.from_table(table, delta=delta, **kw)
        inner = RaceIncumbent(base, rparams, name=name)
    else:
        kw = _overrides(MosParams, params, extra)
        kw.setdefault("eligible_families", BOTH if both else ("f2",))
        inner = MixOrSplitIncumbent(table, base, MosParams(**kw), name=name)
    return ControlAdapter(inner, name, memory_utilization, rate0, horizon)
