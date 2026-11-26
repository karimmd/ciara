from __future__ import annotations

import asyncio
import copy
import shutil
import tempfile
import threading
from typing import Callable, Optional

from ciara_control.clock import FakeClock
from ciara_control.config import RunConfig, parse_config
from ciara_control.controller import Controller
from ciara_control.mocks.inproc import InProcCloud, InProcIO
from ciara_control.mocks.sim import SimNode

PROFILES = ["q35-4b-fp16", "q35-4b-ih-w8a8", "q35-2b-fp16"]


def base_config(run_dir: str, **over) -> dict:
    cfg = {
        "run_id": "t1", "run_dir": run_dir, "seed": 20261016, "epoch_s": 30, "block_epochs": 4,
        "burn_in_epochs": 2, "n_blocks": 8,
        "design": {"label": "screening"},
        "arms": [
            {"profile": "q35-4b-fp16", "batch_cap": 8, "escalate_family2": False},
            {"profile": "q35-4b-ih-w8a8", "batch_cap": 16, "escalate_family2": True},
            {"profile": "q35-2b-fp16", "batch_cap": 8, "escalate_family2": True},
            {"id": "I1-E", "kind": "incumbent", "router": "kappa", "profiles": ["q35-4b-fp16"],
             "incumbent": {"factory": "stub", "kwargs": {"profile": "q35-4b-fp16"}}},
        ],
        "nodes": [{"id": f"E{k}", "gateway": f"http://gw{k}:8001", "engine_manager": f"http://mgr{k}:9"}
                  for k in (1, 2, 3)],
        "cloud": {},
        "control": {"start_delay_s": 10, "resume_margin_s": 5},
    }
    cfg.update(over)
    return cfg


class Fleet:
    def __init__(self, clock: FakeClock, seed: int = 1, rate: float = 1.0):
        self.clock = clock
        self.seed = seed
        self.rate = rate
        self.ios: dict[str, InProcIO] = {}
        self.cloud = InProcCloud()

    def factory(self, node, cfg: RunConfig):
        if node.id not in self.ios:
            sim = SimNode(node.id, node.index, now_s=lambda: self.clock.mono_ns() / 1e9, seed=self.seed,
                          rate=self.rate, switch_mean_s=0.5, endpoint_base=f"http://sim-{node.id}")
            self.ios[node.id] = InProcIO(sim)
        return self.ios[node.id]


def make_controller(cfg_dict: dict, clock: FakeClock, fleet: Fleet, **kw) -> Controller:
    cfg = parse_config(cfg_dict)
    return Controller(cfg, clock=clock, io_factory=fleet.factory, cloud_io=fleet.cloud,
                      log=lambda s: None, **kw)


class TmpDir:
    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix="ciara-test-")
        return self.path

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


class ServerThread:
    def __init__(self, app_factory: Callable):
        self.app_factory = app_factory
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.port: Optional[int] = None
        self._ready = threading.Event()
        self._stop: Optional[asyncio.Event] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _run(self) -> None:
        from aiohttp import web
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        async def main():
            runner = web.AppRunner(self.app_factory(), access_log=None)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            self.port = site._server.sockets[0].getsockname()[1]
            self._stop = asyncio.Event()
            self._ready.set()
            await self._stop.wait()
            await runner.cleanup()

        try:
            self.loop.run_until_complete(main())
        finally:
            self.loop.close()

    def __enter__(self) -> "ServerThread":
        self._thread.start()
        assert self._ready.wait(10)
        return self

    def __exit__(self, *exc) -> None:
        self.loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=10)


def deep(d: dict) -> dict:
    return copy.deepcopy(d)
