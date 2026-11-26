from __future__ import annotations

from typing import Callable, Optional

from ..clock import Clock
from ..config import NodeSpec, RunConfig
from ..httpclient import HttpError
from ..nodeio import NodeIO
from .sim import SimNode


class InProcIO(NodeIO):
    def __init__(self, sim: SimNode):
        self.sim = sim
        self.fail: dict[str, int] = {}
        self.calls: list = []

    def _maybe_fail(self, name: str) -> None:
        n = self.fail.get(name, 0)
        if n > 0:
            self.fail[name] = n - 1
            raise HttpError(f"injected failure in {name}")

    def prepare(self, profiles: list) -> dict:
        self.calls.append(("prepare", list(profiles)))
        self._maybe_fail("prepare")
        return self.sim.prepare(profiles)

    def engine_status(self) -> dict:
        self._maybe_fail("engine_status")
        return self.sim.status()

    def activate(self, profile: str) -> dict:
        self.calls.append(("activate", profile))
        self._maybe_fail("activate")
        return self.sim.activate(profile, switch_s=0.5)

    def gateway_state(self) -> dict:
        self._maybe_fail("gateway_state")
        return self.sim.state()

    def gateway_config(self, cfg: dict) -> dict:
        self.calls.append(("config", dict(cfg)))
        self._maybe_fail("gateway_config")
        return self.sim.set_config(cfg)

    def engine_metrics(self, endpoint: str) -> str:
        self._maybe_fail("engine_metrics")
        return self.sim.metrics_text(endpoint.rsplit("/", 1)[-1])


class InProcCloud:
    enabled = True

    def __init__(self):
        self.notices: list = []

    def notify(self, nodes: dict) -> dict:
        self.notices.append(dict(nodes))
        return {"ok": True}


def make_io_factory(clock: Clock, seed: int = 1, rate: float = 1.0,
                    registry: Optional[dict] = None) -> Callable[[NodeSpec, RunConfig], NodeIO]:
    def factory(node: NodeSpec, cfg: RunConfig) -> NodeIO:
        sim = SimNode(node.id, node.index, now_s=lambda: clock.mono_ns() / 1e9, seed=seed, rate=rate,
                      switch_mean_s=0.5, endpoint_base=f"http://sim-{node.id}")
        io = InProcIO(sim)
        if registry is not None:
            registry[node.id] = io
        return io

    return factory
