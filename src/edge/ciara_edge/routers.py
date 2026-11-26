from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .scoring import load_callable, make_recserve_scorer, make_scorer


class RouterContext:
    def __init__(self, profile_table: Optional[str] = None, feedback_fn: Optional[str] = None):
        self.profile_table = profile_table
        self.feedback_spec = feedback_fn
        self._table = None

    def table(self):
        if self._table is None:
            if not self.profile_table:
                raise ValueError("router I3 needs a profile table (--profile-table or PROFILE_TABLE)")
            from ciara_baselines.profile import ProfileTable
            self._table = ProfileTable.load(Path(self.profile_table))
        return self._table

    def feedback_fn(self) -> Optional[Callable]:
        return load_callable(self.feedback_spec) if self.feedback_spec else None


class Router:
    name = "base"
    needs_after_edge = False
    scorer = None
    wants_feedback = False
    rebuild_per_block = False

    def __init__(self, params: Dict[str, Any] | None = None, ctx: RouterContext | None = None):
        self.params = dict(params or {})
        self.ctx = ctx or RouterContext()
        self.config: Dict[str, Any] = {}

    def configure(self, config: Dict[str, Any]) -> None:
        self.config = config

    def route(self, req_meta: Dict[str, Any], node_state: Dict[str, Any]) -> str:
        return "edge"

    def after_edge(self, req_meta: Dict[str, Any], edge_reply: Dict[str, Any], node_state: Dict[str, Any]) -> str:
        return "accept"

    def wants_after_edge(self, req_meta: Dict[str, Any]) -> bool:
        return bool(self.needs_after_edge)

    def edge_max_tokens(self, req_meta: Dict[str, Any]) -> Optional[int]:
        return None

    def decision_info(self) -> Optional[dict]:
        return None


class KappaRouter(Router):
    name = "kappa"

    def __init__(self, params=None, ctx=None):
        super().__init__(params, ctx)
        self.escalate_family2 = False

    def configure(self, config):
        super().configure(config)
        self.escalate_family2 = bool(config.get("escalate_family2", False))

    def route(self, req_meta, node_state):
        if self.escalate_family2 and int(req_meta.get("family", 0) or 0) == 2:
            return "cloud"
        return "edge"


class EdgeOnlyRouter(Router):
    name = "edge"


class CloudOnlyRouter(Router):
    name = "cloud"

    def route(self, req_meta, node_state):
        return "cloud"


class ConfidenceCascadeRouter(Router):
    name = "cascade-test"
    needs_after_edge = True

    def after_edge(self, req_meta, edge_reply, node_state):
        txt = edge_reply.get("text") or ""
        marker = self.params.get("escalate_if_text_contains")
        if edge_reply.get("status") != 200 or not txt.strip() or edge_reply.get("finish_reason") == "length":
            return "escalate"
        if marker and marker in txt:
            return "escalate"
        return "accept"


class RaceGatewayRouter(Router):
    name = "I2"
    wants_feedback = True
    rebuild_per_block = True

    def __init__(self, params=None, ctx=None):
        super().__init__(params, ctx)
        from ciara_baselines.race import RaceParams, RaceRouter
        p = dict(self.params)
        gw = dict(p.pop("gateway", {}) or {})
        self.race_params = RaceParams.from_router_params(p)
        self.inner = RaceRouter(self.race_params, feedback_fn=self.ctx.feedback_fn())
        self.scorer = make_scorer(gw, self.race_params)
        self._info: Optional[dict] = None

    def route(self, req_meta, node_state):
        r = self.inner.route(req_meta, node_state)
        ev = self.inner.events[-1] if self.inner.events else None
        self._info = None if ev is None else {k: ev.get(k) for k in ("u", "tau", "tau_tilde", "mode", "b_t", "step")}
        return r

    def after_edge(self, req_meta, edge_reply, node_state):
        return self.inner.after_edge(req_meta, edge_reply, node_state)

    def decision_info(self):
        return self._info

    def label(self, req_id: str, wrong: Optional[bool]) -> bool:
        return self.inner.label(req_id, wrong)

    def abandon(self, req_meta) -> None:
        self.inner.abandon(req_meta)

    def state(self) -> dict:
        return {"calibrators": self.inner.state(), "pending_labels": len(self.inner.pending),
                "score": getattr(self.scorer, "mode", None)}


class MixOrSplitGatewayRouter(Router):
    name = "I3"

    def __init__(self, params=None, ctx=None):
        super().__init__(params, ctx)
        from ciara_baselines.mixorsplit import MosParams
        p = dict(self.params)
        p.pop("gateway", None)
        self.mos_params = MosParams.from_router_params(p)
        self.inner = None
        self._info: Optional[dict] = None

    def configure(self, config):
        super().configure(config)
        from ciara_baselines.base import Config
        from ciara_baselines.mixorsplit import MixOrSplitRouter
        base = Config(arm_id=str(config.get("arm_id") or "I3"), profile=config["profile"], batch_cap=int(config["batch_cap"]),
                      escalate_family2=bool(config.get("escalate_family2", False)), router="I3",
                      block_id=config.get("block_id"))
        self.inner = MixOrSplitRouter(self.ctx.table(), base, self.mos_params)

    def route(self, req_meta, node_state):
        d = self.inner.decide_mode(req_meta, node_state)
        self.inner.last = d
        self._info = {"mode": d.mode, "infeasible": d.infeasible, "tie_break": d.tie_break,
                      "pred_s": {e.mode: (None if e.pred is None else round(e.pred.total_s, 4)) for e in d.evals}}
        return "cloud" if d.route == "cloud" else "edge"

    def decision_info(self):
        return self._info


class RecServeGatewayRouter(Router):
    name = "RecServe"
    rebuild_per_block = True

    def __init__(self, params=None, ctx=None):
        super().__init__(params, ctx)
        from ciara_baselines.recserve import RecServeParams, RecServeRouter
        p = dict(self.params)
        p.pop("gateway", None)
        self.rec_params = RecServeParams.from_router_params(p)
        self.inner = RecServeRouter(self.rec_params)
        self.scorer = make_recserve_scorer(self.rec_params.score)
        self._info: Optional[dict] = None

    def route(self, req_meta, node_state):
        c = req_meta.get("race_u")
        if self.rec_params.score == "perplexity":
            reply = {} if c is None else {"confidence": float(c)}
            r = "cloud" if self.inner.after_edge(req_meta, reply, node_state) == "escalate" else "edge"
        else:
            meta = dict(req_meta)
            meta["recserve_c"] = None if c is None else float(c)
            r = self.inner.route(meta, node_state)
        last = self.inner.last or {}
        self._info = {k: last.get(k) for k in ("c", "threshold", "n_hist", "step", "reason")}
        return r

    def decision_info(self):
        return self._info

    def state(self) -> dict:
        return self.inner.state()


def request_half(req_id: Any) -> int:
    return hashlib.sha256(str(req_id).encode("utf-8")).digest()[0] & 1


def _family(req_meta: Dict[str, Any]) -> int:
    try:
        return int(req_meta.get("family", 0) or 0)
    except (TypeError, ValueError):
        return 0


class ClassAKappaRouter(Router):
    name = "kappaA"

    def route(self, req_meta, node_state):
        if _family(req_meta) == 2 and str(req_meta.get("class") or "") == "A":
            return "cloud"
        return "edge"


class HalfKappaRouter(Router):
    name = "kappaH"

    def route(self, req_meta, node_state):
        if _family(req_meta) == 2 and request_half(req_meta.get("req_id") or req_meta.get("req")) == 1:
            return "cloud"
        return "edge"


CASCADE_BUDGET = 32


class TokenCascadeRouter(Router):
    name = "kappaC"

    def __init__(self, params=None, ctx=None):
        super().__init__(params, ctx)
        b = self.params.get("budget", CASCADE_BUDGET)
        if isinstance(b, bool) or not isinstance(b, int) or b < 1:
            raise ValueError(f"kappaC needs an integer budget >= 1, got {b!r}")
        self.budget = b
        self._info: Optional[dict] = None

    def route(self, req_meta, node_state):
        return "edge"

    def wants_after_edge(self, req_meta):
        return _family(req_meta) == 2

    def edge_max_tokens(self, req_meta):
        if _family(req_meta) != 2:
            return None
        cap = req_meta.get("max_tokens")
        return min(self.budget, cap) if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0 else self.budget

    def after_edge(self, req_meta, edge_reply, node_state):
        if edge_reply.get("status") != 200 or edge_reply.get("error"):
            reason = "edge_error"
        elif edge_reply.get("finish_reason") == "length":
            reason = "length"
        else:
            reason = "complete"
        self._info = {"budget": self.budget, "reason": reason}
        return "accept" if reason == "complete" else "escalate"

    def decision_info(self):
        return self._info


BUILTIN = {c.name: c for c in (KappaRouter, EdgeOnlyRouter, CloudOnlyRouter, ConfidenceCascadeRouter,
                               RaceGatewayRouter, MixOrSplitGatewayRouter, RecServeGatewayRouter, ClassAKappaRouter, HalfKappaRouter, TokenCascadeRouter)}


def get_router(name: str, params: Dict[str, Any] | None = None, ctx: RouterContext | None = None) -> Router:
    if name not in BUILTIN:
        raise ValueError(f"unknown router {name!r}; available: {sorted(BUILTIN)}")
    try:
        return BUILTIN[name](params, ctx)
    except ImportError as e:
        raise ValueError(f"router {name!r} needs the package ciara_baselines on PYTHONPATH: {e}")


def router_signature(name: str, params: Dict[str, Any], block_id: Any) -> str:
    cls = BUILTIN.get(name)
    per_block = bool(cls and cls.rebuild_per_block)
    return json.dumps([name, params or {}, block_id if per_block else None], sort_keys=True, default=str)
