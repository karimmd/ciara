from __future__ import annotations

import dataclasses
import importlib
from typing import Any, Optional


class Obs(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class StubIncumbent:
    def __init__(self, profile: str = "q35-4b-fp16", batch_cap_low: int = 8, batch_cap_high: int = 16,
                 queue_threshold: float = 4.0, escalate_family2: bool = True, router: str = "kappa"):
        self.profile = profile
        self.low = batch_cap_low
        self.high = batch_cap_high
        self.threshold = queue_threshold
        self.escalate = escalate_family2
        self.router = router
        self.calls = 0
        self.blocks: list = []

    def begin_block(self, info: dict) -> None:
        self.blocks.append(dict(info))

    def decide(self, obs: Any) -> dict:
        self.calls += 1
        gw = obs.get("gateway") or {}
        q = gw.get("queue_len")
        cap = self.high if (q is not None and q > self.threshold) else self.low
        return {"profile": self.profile, "batch_cap": cap, "escalate_family2": self.escalate,
                "router": self.router}


class ScriptedIncumbent:
    def __init__(self, script: list, **_unused):
        self.script = list(script)
        self.calls = 0

    def decide(self, obs: Any) -> dict:
        cfg = self.script[self.calls % len(self.script)]
        self.calls += 1
        return dict(cfg)


class BrokenIncumbent:
    def decide(self, obs: Any) -> dict:
        raise RuntimeError("broken incumbent")


_BUILTIN = {"stub": StubIncumbent, "scripted": ScriptedIncumbent, "broken": BrokenIncumbent}


def load_incumbent(spec: dict) -> Any:
    factory = spec["factory"]
    kwargs = spec.get("kwargs") or {}
    if factory in _BUILTIN:
        return _BUILTIN[factory](**kwargs)
    if ":" not in factory:
        raise ValueError(f"incumbent factory '{factory}': expected 'module:attr' or one of {sorted(_BUILTIN)}")
    mod_name, attr = factory.split(":", 1)
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    inst = obj(**kwargs) if callable(obj) else obj
    if not hasattr(inst, "decide"):
        raise TypeError(f"incumbent '{factory}' has no decide(obs)")
    return inst


def config_to_dict(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "to_dict") and callable(cfg.to_dict):
        return dict(cfg.to_dict())
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    raise TypeError(f"decide() returned {type(cfg).__name__}; expected a mapping or an object with to_dict()")


_KEYS = ("profile", "batch_cap", "escalate_family2", "router", "router_params")


def merge_incumbent_config(returned: Any, arm, previous: Optional[dict], block_id: int) -> dict:
    d = config_to_dict(returned)
    base = arm.config_object(block_id)
    out = dict(base)
    if previous:
        for k in _KEYS:
            if k in previous and previous[k] is not None:
                out[k] = previous[k]
    for k in _KEYS:
        if k in d and d[k] is not None:
            out[k] = d[k]
    out["arm_id"] = arm.id
    out["block_id"] = int(block_id)
    out["escalate_family2"] = bool(out.get("escalate_family2", True))
    if out.get("profile") is None or out.get("batch_cap") is None:
        raise ValueError(f"incumbent arm {arm.id}: configuration lacks profile or batch_cap: {out}")
    extras = {k: v for k, v in d.items() if k not in _KEYS and k not in ("arm_id", "block_id")}
    if extras:
        out["router_params"] = {**(out.get("router_params") or {}), **extras}
    if getattr(arm, "kind", None) == "incumbent" and arm.router == "kappaC":
        out["router"], out["router_params"] = arm.router, dict(arm.router_params)
    return out
