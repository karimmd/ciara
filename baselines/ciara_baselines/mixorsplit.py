from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .base import Config, Incumbent, Obs, req_family
from .profile import ProfileRow, ProfileTable

INF = float("inf")
MODE_ORDER = ("MIX", "SPLIT", "CLOUD")


@dataclass(frozen=True)
class MosParams:
    modes: Tuple[str, ...] = ("MIX", "CLOUD")

    ttft_slo_s: Optional[Mapping[str, float]] = None
    tpot_slo_s: Optional[Mapping[str, float]] = None
    default_deadline_s: float = INF

    eligible_families: Tuple[str, ...] = ("f2",)

    infeasible_policy: str = "min_latency"

    output_tokens: str = "profile"

    guard_slots: int = 0
    cloud_cap: int = 16

    backhaul_rtt_s: float = 0.05
    backhaul_mbps: float = 100.0
    bytes_per_token: float = 4.0

    tie_tol_s: float = 1e-9

    def to_router_params(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else (dict(v) if isinstance(v, Mapping) else v))
                for k, v in vars(self).items()}

    @staticmethod
    def from_router_params(d: Mapping[str, Any]) -> "MosParams":
        d = dict(d)
        for k in ("modes", "eligible_families"):
            if k in d:
                d[k] = tuple(d[k])
        known = set(MosParams.__dataclass_fields__)
        return MosParams(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class Prediction:
    total_s: float
    ttft_s: float
    tpot_s: float


@dataclass
class ModeEval:
    mode: str
    available: bool
    pred: Optional[Prediction]
    ttft_ok: bool
    tpot_ok: bool
    deadline_ok: bool
    residual: float
    notes: Tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.available and self.pred is not None and self.ttft_ok and self.tpot_ok and self.deadline_ok


@dataclass
class MosDecision:
    mode: str
    route: str
    infeasible: bool
    evals: List[ModeEval]
    tie_break: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode, "route": self.route, "infeasible": self.infeasible, "tie_break": self.tie_break,
            "evals": [{"mode": e.mode, "available": e.available, "feasible": e.feasible, "residual": e.residual,
                       "total_s": None if e.pred is None else e.pred.total_s,
                       "ttft_s": None if e.pred is None else e.pred.ttft_s,
                       "tpot_s": None if e.pred is None else e.pred.tpot_s, "notes": list(e.notes)}
                      for e in self.evals],
        }


class MixOrSplitRouter:
    def __init__(self, table: ProfileTable, base: Config, params: MosParams,
                 on_decision: Optional[Callable[[Mapping[str, Any], MosDecision], None]] = None):
        if table.cloud is None and "CLOUD" in params.modes:
            raise ValueError("CLOUD mode needs a cloud row in the profile table")
        self.table = table
        self.base = base
        self.p = params
        self.on_decision = on_decision
        self.last: Optional[MosDecision] = None

    def _out_tokens(self, row: ProfileRow, fam: str, req_meta: Mapping[str, Any]) -> float:
        cap = float(req_meta.get("max_tokens", 512))
        if self.p.output_tokens == "request_cap":
            return cap
        fp = row.families.get(fam)
        return float(fp.out_tokens) if fp is not None and fp.out_tokens is not None else cap

    def _family_prompt_tokens(self, fam: str, erow: ProfileRow) -> float:
        rows = [erow, *self.table.edge.values(), *([self.table.cloud] if self.table.cloud else [])]
        for r in rows:
            if fam in r.families:
                return r.families[fam].prompt_tokens
        raise KeyError(f"no prompt length known for family {fam}: pass req_meta['prompt_tokens']")

    def _queue_wait(self, row: ProfileRow, length: float, cap: int, ahead: int, n_out: float) -> float:
        if ahead < cap:
            return 0.0
        cell = row.interp_cell(length, cap)
        service = cell.ttft + max(n_out - 1.0, 0.0) / cell.dec_rate
        return (ahead - cap + 1) * service / cap

    def predict_mix(self, row: ProfileRow, cap: int, length: float, n_out: float, queue_len: int, inflight: int,
                    access_up_s: float = 0.0, access_down_s: float = 0.0) -> Prediction:
        ahead = inflight + queue_len
        c = min(cap, ahead + 1)
        cell = row.interp_cell(length, c)
        tpot = 1.0 / cell.dec_rate
        wait = self._queue_wait(row, length, cap, ahead, n_out)
        ttft = access_up_s + wait + cell.ttft
        total = ttft + max(n_out - 1.0, 0.0) * tpot + access_down_s
        return Prediction(total, ttft, tpot)

    def predict_cloud(self, row: ProfileRow, length: float, n_out: float, queue_len: int, inflight: int,
                      rtt_s: float, mbps: float, access_up_s: float = 0.0, access_down_s: float = 0.0) -> Prediction:
        cap = self.p.cloud_cap
        ahead = inflight + queue_len
        c = min(cap, ahead + 1)
        cell = row.interp_cell(length, c)
        tpot = 1.0 / cell.dec_rate
        wait = self._queue_wait(row, length, cap, ahead, n_out)
        bpt = 8.0 * self.p.bytes_per_token
        up = access_up_s + length * bpt / (mbps * 1e6) + rtt_s / 2.0
        down = access_down_s + n_out * bpt / (mbps * 1e6) + rtt_s / 2.0
        ttft = up + wait + cell.ttft + rtt_s / 2.0
        total = up + wait + cell.ttft + max(n_out - 1.0, 0.0) * tpot + down
        return Prediction(total, ttft, tpot)

    def decide_mode(self, req_meta: Mapping[str, Any], node_state: Mapping[str, Any]) -> MosDecision:
        p = self.p
        fam = req_family(req_meta)
        ns = node_state
        ckpt = ns.get("profile", self.base.profile)
        cap = int(ns.get("batch_cap", self.base.batch_cap))
        erow = self.table.edge[ckpt]
        length = float(req_meta.get("prompt_tokens") or self._family_prompt_tokens(fam, erow))
        deadline_ms = req_meta.get("deadline_ms")
        tau = float(deadline_ms) / 1000.0 if deadline_ms is not None else p.default_deadline_s
        elapsed = float(req_meta.get("elapsed_s", 0.0))
        d_ft = INF if p.ttft_slo_s is None else p.ttft_slo_s.get(fam, INF)
        d_pt = INF if p.tpot_slo_s is None else p.tpot_slo_s.get(fam, INF)
        access_down = float(ns.get("access_down_s", 0.0))
        q, infl = int(ns.get("queue_len", 0)), int(ns.get("inflight", 0))
        cq, cinfl = int(ns.get("cloud_queue_len", 0)), int(ns.get("cloud_inflight", 0))
        rtt = float(ns.get("backhaul_rtt_s", p.backhaul_rtt_s))
        mbps = float(ns.get("backhaul_mbps", p.backhaul_mbps))
        eligible = fam in p.eligible_families

        evals: List[ModeEval] = []
        for mode in MODE_ORDER:
            if mode not in p.modes:
                continue
            notes: List[str] = []
            if mode == "MIX":
                n_out = self._out_tokens(erow, fam, req_meta)
                cap_tokens = float(req_meta.get("max_tokens", 512))
                kv_ok = erow.kv_tokens / cap >= length + cap_tokens
                if not kv_ok:
                    notes.append("kv_slot_too_small")
                healthy = bool(ns.get("health_ok", True)) and not bool(ns.get("switching", False))
                if not healthy:
                    notes.append("engine_unavailable")
                avail = kv_ok and healthy
                pred = self.predict_mix(erow, cap, length, n_out, q, infl, 0.0, access_down)
                residual = float(cap - infl - q - p.guard_slots)
            elif mode == "CLOUD":
                crow = self.table.cloud
                n_out = self._out_tokens(crow, fam, req_meta)
                avail = bool(ns.get("cloud_ok", True)) and eligible and mbps > 0
                if not eligible:
                    notes.append("family_not_eligible")
                pred = self.predict_cloud(crow, length, n_out, cq, cinfl, rtt, mbps, 0.0, access_down)
                residual = float(p.cloud_cap - cinfl - cq - p.guard_slots)
            else:
                avail, pred, residual = False, None, -INF
                notes.append("split_unavailable")
            ttft_ok = pred is not None and elapsed + pred.ttft_s <= d_ft + 1e-12
            tpot_ok = pred is not None and pred.tpot_s <= d_pt + 1e-12
            dl_ok = pred is not None and elapsed + pred.total_s <= tau + 1e-12
            evals.append(ModeEval(mode, avail, pred, ttft_ok, tpot_ok, dl_ok, residual, tuple(notes)))

        feas = [e for e in evals if e.feasible]
        infeasible = not feas
        pool = feas
        if infeasible:
            if p.infeasible_policy == "reject":
                return MosDecision("REJECT", "reject", True, evals)
            pool = [e for e in evals if e.available and e.pred is not None]
            if not pool:
                return MosDecision("MIX", "edge", True, evals)
        best = min(e.pred.total_s for e in pool)
        tied = [e for e in pool if e.pred.total_s <= best + p.tie_tol_s]
        tie_break = len(tied) > 1
        pick = min(tied, key=lambda e: (-e.residual, MODE_ORDER.index(e.mode)))
        return MosDecision(pick.mode, "cloud" if pick.mode == "CLOUD" else "edge", infeasible, evals, tie_break)


class MixOrSplitIncumbent(Incumbent):
    name = "I3"

    def __init__(self, table: ProfileTable, base_config: Config, params: MosParams,
                 on_decision: Optional[Callable[[Mapping[str, Any], MosDecision], None]] = None, name: str = "I3"):
        self.name = name
        self.base = base_config
        self.p = params
        self.router = MixOrSplitRouter(table, base_config, params, on_decision)

    def decide(self, obs: Obs) -> Config:
        b = self.base
        return Config(
            arm_id=self.name, profile=b.profile, batch_cap=b.batch_cap, escalate_family2=False, router="I3",
            block_id=obs.block_id, router_params=self.p.to_router_params(), base_arm=b.base_arm,
        )

    def route(self, req_meta: Mapping[str, Any], node_state: Mapping[str, Any]) -> str:
        d = self.router.decide_mode(req_meta, node_state)
        self.router.last = d
        if self.router.on_decision is not None:
            self.router.on_decision(req_meta, d)
        return d.route

    def after_edge(self, req_meta: Mapping[str, Any], edge_reply: Any, node_state: Mapping[str, Any]) -> str:
        return "accept"

    def diagnostics(self) -> Dict[str, Any]:
        return {"regime": self.name, "last": None if self.router.last is None else self.router.last.to_dict()}
