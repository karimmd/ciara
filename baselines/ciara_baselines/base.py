from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

FAMILIES = ("f1", "f2")


def family_key(value: Any) -> str:
    s = str(value).strip().lower()
    if s in ("1", "f1", "family1", "family_1"):
        return "f1"
    if s in ("2", "f2", "family2", "family_2"):
        return "f2"
    raise ValueError(f"unknown task family: {value!r}")


def req_family(req_meta: Mapping[str, Any]) -> str:
    return family_key(req_meta["family"])


def arm_id(profile: str, batch_cap: int, kappa: int) -> str:
    return f"{profile}|b{int(batch_cap)}|k{int(kappa)}"


@dataclass(frozen=True)
class Obs:
    epoch: int = 0
    arrival_rate: float = 0.0
    memory_budget_gib: float = 10.45
    arrival_rate_by_family: Optional[Mapping[str, float]] = None
    block_id: Optional[int] = None
    telemetry: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    arm_id: str
    profile: str
    batch_cap: int
    escalate_family2: bool
    router: str = "kappa"
    block_id: Optional[int] = None
    router_params: Mapping[str, Any] = field(default_factory=dict)
    base_arm: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "arm_id": self.arm_id,
            "profile": self.profile,
            "batch_cap": int(self.batch_cap),
            "escalate_family2": bool(self.escalate_family2),
            "router": self.router,
            "block_id": self.block_id,
        }
        if self.base_arm is not None:
            d["base_arm"] = self.base_arm
        if self.router_params:
            d["router_params"] = dict(self.router_params)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def same_engine_setting(self, other: "Config") -> bool:
        return self.profile == other.profile


class Incumbent:
    name = "incumbent"

    def reset(self, block_id: Optional[int] = None, **kwargs: Any) -> None:
        pass

    def decide(self, obs: Obs) -> Config:
        raise NotImplementedError

    def route(self, req_meta: Mapping[str, Any], node_state: Mapping[str, Any]) -> str:
        return "edge"

    def after_edge(self, req_meta: Mapping[str, Any], edge_reply: Any, node_state: Mapping[str, Any]) -> str:
        return "accept"

    def diagnostics(self) -> Dict[str, Any]:
        return {}
