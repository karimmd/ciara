from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .util import canonical_json, sha256_hex

DEFAULT_RUNS_ROOT = os.path.join(os.environ.get("CIARA_ROOT", "."), "runs")
DESIGN_LABELS = ("screening", "certification", "deployment", "balanced")
ADMISSION_MODES = ("fifo", "f1prio")
ARM_KINDS = ("static", "incumbent")


class ConfigError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("invalid run configuration:\n  " + "\n  ".join(self.problems))


def make_arm_id(profile: str, batch_cap: int, escalate_family2: bool) -> str:
    return f"{profile}|b{int(batch_cap)}|k{1 if escalate_family2 else 0}"


@dataclass(frozen=True)
class ArmSpec:
    id: str
    kind: str = "static"
    profile: Optional[str] = None
    batch_cap: Optional[int] = None
    escalate_family2: bool = True
    router: str = "kappa"
    router_params: dict = field(default_factory=dict)
    incumbent: Optional[dict] = None
    profiles: tuple = ()
    admission: str = "fifo"

    @property
    def is_incumbent(self) -> bool:
        return self.kind == "incumbent"

    def config_object(self, block_id: int) -> dict:
        cfg: dict[str, Any] = {
            "arm_id": self.id,
            "profile": self.profile,
            "batch_cap": self.batch_cap,
            "escalate_family2": bool(self.escalate_family2),
            "router": self.router,
            "block_id": int(block_id),
            "admission": self.admission,
        }
        if self.router_params:
            cfg["router_params"] = dict(self.router_params)
        return cfg

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "profile": self.profile, "batch_cap": self.batch_cap,
            "escalate_family2": self.escalate_family2, "router": self.router,
            "router_params": dict(self.router_params), "incumbent": self.incumbent,
            "profiles": list(self.profiles),
            **({"admission": self.admission} if self.admission != "fifo" else {}),
        }


@dataclass(frozen=True)
class NodeSpec:
    id: str
    index: int
    gateway: Optional[str] = None
    engine_manager: Optional[str] = None
    memory_budget_mib: int = 11264


@dataclass(frozen=True)
class CloudSpec:
    gateway: Optional[str] = None
    engine: Optional[str] = None
    max_in_flight: int = 16


@dataclass(frozen=True)
class ContextSpec:
    ref: Optional[str] = None
    default: str = "pooled"


@dataclass(frozen=True)
class DesignSpec:
    label: str
    arms: tuple
    incumbent: Optional[str] = None
    epsilon: Optional[float] = None
    policy: Optional[dict] = None
    composition: Optional[tuple] = None


@dataclass(frozen=True)
class ControlSettings:
    start_delay_s: float = 10.0
    resume_margin_s: float = 5.0
    on_resume: str = "continue_block"
    prepare_engines: bool = True
    on_incumbent_error: str = "continue"
    retries: int = 2
    state_s: float = 5.0
    config_s: float = 10.0
    activate_s: float = 300.0
    prepare_s: float = 1800.0
    metrics_s: float = 5.0
    workers: Optional[int] = None


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    run_dir: str
    seed: int
    epoch_s: float
    block_epochs: int
    burn_in_epochs: int
    n_blocks: int
    design: DesignSpec
    arms: dict
    nodes: tuple
    cloud: CloudSpec
    context: ContextSpec
    control: ControlSettings
    source_path: Optional[str] = None
    warnings: tuple = ()

    @property
    def total_epochs(self) -> int:
        return self.n_blocks * self.block_epochs

    @property
    def epoch_ns(self) -> int:
        return int(round(self.epoch_s * 1e9))

    def node(self, node_id: str) -> NodeSpec:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def design_arm_ids(self) -> tuple:
        return tuple(self.design.arms)

    def profiles_to_prepare(self) -> list[str]:
        seen: list[str] = []
        for aid in self.design.arms:
            arm = self.arms[aid]
            for p in ([arm.profile] if arm.profile else []) + list(arm.profiles):
                if p not in seen:
                    seen.append(p)
        return seen

    def design_dict(self) -> dict:
        return {
            "run_id": self.run_id, "seed": self.seed, "epoch_s": self.epoch_s,
            "block_epochs": self.block_epochs, "burn_in_epochs": self.burn_in_epochs,
            "n_blocks": self.n_blocks,
            "design": {"label": self.design.label, "arms": list(self.design.arms),
                       "incumbent": self.design.incumbent, "epsilon": self.design.epsilon,
                       "policy": self.design.policy,
                       **({"composition": list(self.design.composition)} if self.design.composition else {})},
            "arms": [a.to_dict() for a in self.arms.values()],
            "nodes": [{"id": n.id, "index": n.index} for n in self.nodes],
            "context": {"ref": self.context.ref, "default": self.context.default},
        }

    @property
    def design_hash(self) -> str:
        return sha256_hex(canonical_json(self.design_dict()))


def _as_bool(v: Any, default: bool) -> bool:
    return default if v is None else bool(v)


def _node_default_manager(node_id: str) -> Optional[str]:
    m = re.fullmatch(r"E(\d+)", node_id)
    return f"http://127.0.0.1:92{int(m.group(1))}0" if m else None


def parse_config(raw: dict, source_path: Optional[str] = None) -> RunConfig:
    problems: list[str] = []
    warnings: list[str] = []

    def need(d: dict, key: str, kind, where: str):
        if key not in d or d[key] is None:
            problems.append(f"{where}: missing '{key}'")
            return None
        v = d[key]
        if kind is not None and (isinstance(v, bool) or not isinstance(v, kind)):
            problems.append(f"{where}.{key}: expected {getattr(kind, '__name__', kind)}, got {type(v).__name__}")
            return None
        return v

    if not isinstance(raw, dict):
        raise ConfigError(["top level must be a mapping"])

    run_id = need(raw, "run_id", str, "config")
    seed = need(raw, "seed", int, "config")
    l = need(raw, "block_epochs", int, "config")
    b = need(raw, "burn_in_epochs", int, "config")
    k = need(raw, "n_blocks", int, "config")
    epoch_s = raw.get("epoch_s", 30)
    if isinstance(epoch_s, bool) or not isinstance(epoch_s, (int, float)) or epoch_s <= 0:
        problems.append("config.epoch_s: expected a positive number")
        epoch_s = 30
    if l is not None and l < 1:
        problems.append("config.block_epochs: must be >= 1")
    if b is not None and l is not None and not (0 <= b < l):
        problems.append("config.burn_in_epochs: need 0 <= b < block_epochs")
    if k is not None and k < 1:
        problems.append("config.n_blocks: must be >= 1")
    if run_id is not None and not re.fullmatch(r"[A-Za-z0-9._\-]+", run_id):
        problems.append("config.run_id: use letters, digits, '.', '_' and '-' only")

    arms: dict[str, ArmSpec] = {}
    raw_arms = raw.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        problems.append("config.arms: expected a non-empty list")
        raw_arms = []
    for j, a in enumerate(raw_arms):
        where = f"arms[{j}]"
        if not isinstance(a, dict):
            problems.append(f"{where}: expected a mapping")
            continue
        kind = a.get("kind", "static")
        if kind not in ARM_KINDS:
            problems.append(f"{where}.kind: expected one of {ARM_KINDS}")
            continue
        esc = _as_bool(a.get("escalate_family2"), True)
        profile = a.get("profile")
        cap = a.get("batch_cap")
        aid = a.get("id")
        if kind == "static":
            if not isinstance(profile, str) or not profile:
                problems.append(f"{where}: static arm needs 'profile'")
                continue
            if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
                problems.append(f"{where}: static arm needs integer 'batch_cap' >= 1")
                continue
            generated = make_arm_id(profile, cap, esc)
            if aid is None:
                aid = generated
            elif aid != generated:
                warnings.append(f"{where}: id '{aid}' differs from the arm id convention '{generated}'")
        else:
            if not isinstance(aid, str) or not aid:
                problems.append(f"{where}: incumbent arm needs 'id'")
                continue
            inc = a.get("incumbent")
            if not isinstance(inc, dict) or not isinstance(inc.get("factory"), str):
                problems.append(f"{where}.incumbent: expected {{factory: 'module:attr' or 'stub', kwargs: {{...}}}}")
                continue
            if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
                problems.append(f"{where}.batch_cap: expected integer >= 1")
                continue
        if aid in arms:
            problems.append(f"{where}: duplicate arm id '{aid}'")
            continue
        admission = a.get("admission", "fifo")
        if admission not in ADMISSION_MODES:
            problems.append(f"{where}.admission: expected one of {ADMISSION_MODES}")
            admission = "fifo"
        params = a.get("router_params") or {}
        if not isinstance(params, dict):
            problems.append(f"{where}.router_params: expected a mapping")
            params = {}
        arms[aid] = ArmSpec(
            id=aid, kind=kind, profile=profile, batch_cap=cap, escalate_family2=esc,
            router=str(a.get("router", "kappa")), router_params=params,
            incumbent=a.get("incumbent") if kind == "incumbent" else None,
            profiles=tuple(a.get("profiles") or ()),
            admission=admission,
        )

    design = None
    rd = raw.get("design")
    if not isinstance(rd, dict):
        problems.append("config.design: expected a mapping")
    else:
        label = rd.get("label")
        if label not in DESIGN_LABELS:
            problems.append(f"design.label: expected one of {DESIGN_LABELS}")
        else:
            d_arms = rd.get("arms")
            policy = rd.get("policy")
            incumbent = rd.get("incumbent")
            eps = rd.get("epsilon")
            if label == "screening":
                ids = list(d_arms) if d_arms else list(arms)
                if len(ids) < 1:
                    problems.append("design.arms: screening needs at least one arm")
                design = DesignSpec(label, tuple(ids))
            elif label == "certification":
                ids = list(d_arms or [])
                if len(ids) < 1:
                    problems.append("design.arms: certification needs the certification set S")
                if incumbent not in ids:
                    problems.append("design.incumbent: must be a member of design.arms")
                if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not (0 < eps <= 1):
                    problems.append("design.epsilon: expected a number in (0, 1]")
                    eps = None
                design = DesignSpec(label, tuple(ids), incumbent=incumbent,
                                    epsilon=float(eps) if eps is not None else None)
            elif label == "balanced":
                comp = rd.get("composition")
                if not isinstance(comp, list) or not comp or not all(isinstance(x, str) for x in comp):
                    problems.append("design.composition: balanced needs a list of arm ids, one per node")
                    comp = []
                ids = list(dict.fromkeys(comp))
                design = DesignSpec(label, tuple(ids), composition=tuple(comp))
            else:
                if not isinstance(policy, dict) or not policy:
                    problems.append("design.policy: deployment needs {arm_id: prob} or {context: {arm_id: prob}}")
                    policy = {}
                flat_ids: list[str] = []
                tables = ([policy] if all(not isinstance(v, dict) for v in policy.values())
                          else [v for v in policy.values() if isinstance(v, dict)])
                for t in tables:
                    for aid, p in t.items():
                        if aid not in flat_ids:
                            flat_ids.append(aid)
                        if isinstance(p, bool) or not isinstance(p, (int, float)) or p < 0:
                            problems.append(f"design.policy[{aid}]: expected a probability >= 0")
                    s = sum(float(p) for p in t.values() if isinstance(p, (int, float)) and not isinstance(p, bool))
                    if abs(s - 1.0) > 1e-9:
                        problems.append(f"design.policy: probabilities sum to {s}, expected 1")
                design = DesignSpec(label, tuple(flat_ids), policy=policy)
            if design is not None:
                for aid in design.arms:
                    if aid not in arms:
                        problems.append(f"design.arms: unknown arm id '{aid}'")
                if len(set(design.arms)) != len(design.arms):
                    problems.append("design.arms: duplicate entries")

    nodes: list[NodeSpec] = []
    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        problems.append("config.nodes: expected a non-empty list")
        raw_nodes = []
    for j, n in enumerate(raw_nodes):
        where = f"nodes[{j}]"
        if not isinstance(n, dict) or not isinstance(n.get("id"), str):
            problems.append(f"{where}: expected a mapping with 'id'")
            continue
        idx = n.get("index", j + 1)
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
            problems.append(f"{where}.index: expected a non-negative integer")
            continue
        nodes.append(NodeSpec(
            id=n["id"], index=idx, gateway=n.get("gateway"),
            engine_manager=n.get("engine_manager") or _node_default_manager(n["id"]),
            memory_budget_mib=int(n.get("memory_budget_mib", 11264)),
        ))
    if len({n.id for n in nodes}) != len(nodes):
        problems.append("nodes: duplicate ids")
    if len({n.index for n in nodes}) != len(nodes):
        problems.append("nodes: duplicate indexes")
    if design is not None and design.label == "balanced" and design.composition and len(design.composition) != len(nodes):
        problems.append(f"design.composition: {len(design.composition)} entries for {len(nodes)} nodes (one arm per node and round)")

    rc = raw.get("cloud") or {}

    def optional_url(key: str) -> Optional[str]:
        v = rc.get(key)
        if isinstance(v, str) and "${" in v:
            warnings.append(f"cloud.{key}: unresolved variable {v}; the cloud {key} is not used")
            return None
        return v

    cloud = CloudSpec(gateway=optional_url("gateway"), engine=optional_url("engine"),
                      max_in_flight=int(rc.get("max_in_flight", 16)))
    if not (1 <= cloud.max_in_flight <= 16):
        problems.append("cloud.max_in_flight: 1 to 16 are allowed")

    rx = raw.get("context") or {}
    context = ContextSpec(ref=rx.get("ref"), default=str(rx.get("default", "pooled")))

    rk = raw.get("control") or {}
    tmo = rk.get("timeouts") or {}
    defaults = ControlSettings()
    control = ControlSettings(
        start_delay_s=float(rk.get("start_delay_s", defaults.start_delay_s)),
        resume_margin_s=float(rk.get("resume_margin_s", defaults.resume_margin_s)),
        on_resume=str(rk.get("on_resume", defaults.on_resume)),
        prepare_engines=_as_bool(rk.get("prepare_engines"), defaults.prepare_engines),
        on_incumbent_error=str(rk.get("on_incumbent_error", defaults.on_incumbent_error)),
        retries=int(rk.get("retries", defaults.retries)),
        state_s=float(tmo.get("state_s", defaults.state_s)),
        config_s=float(tmo.get("config_s", defaults.config_s)),
        activate_s=float(tmo.get("activate_s", defaults.activate_s)),
        prepare_s=float(tmo.get("prepare_s", defaults.prepare_s)),
        metrics_s=float(tmo.get("metrics_s", defaults.metrics_s)),
        workers=rk.get("workers"),
    )
    if control.on_resume not in ("continue_block", "next_block"):
        problems.append("control.on_resume: expected continue_block or next_block")
    if control.on_incumbent_error not in ("continue", "abort"):
        problems.append("control.on_incumbent_error: expected continue or abort")

    if design is not None and design.label in ("screening", "certification", "deployment"):
        has_inc = any(arms[a].is_incumbent for a in design.arms if a in arms)
        if has_inc and design.label == "deployment":
            warnings.append("design: deployment policy includes an incumbent arm")

    if problems:
        raise ConfigError(problems)

    runs_root = raw.get("runs_root", DEFAULT_RUNS_ROOT)
    run_dir = raw.get("run_dir") or os.path.join(runs_root, run_id)
    return RunConfig(
        run_id=run_id, run_dir=run_dir, seed=seed, epoch_s=float(epoch_s), block_epochs=l,
        burn_in_epochs=b, n_blocks=k, design=design, arms=arms, nodes=tuple(nodes), cloud=cloud,
        context=context, control=control, source_path=source_path, warnings=tuple(warnings),
    )


def load_config(path: str, overrides: Optional[dict] = None) -> RunConfig:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    text = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), m.group(0)), text)
    if path.endswith((".yaml", ".yml")):
        import yaml
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if overrides:
        raw = {**raw, **overrides}
    return parse_config(raw, source_path=os.path.abspath(path))
