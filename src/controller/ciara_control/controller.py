from __future__ import annotations

import argparse
import concurrent.futures
import copy
import dataclasses
import json
import os
import platform
import signal
import socket
import sys
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import promtext
from .clock import Clock, FakeClock, RealClock
from .config import ArmSpec, ConfigError, NodeSpec, RunConfig, load_config
from .design import Draw, make_design_law
from .incumbents import Obs, load_incumbent, merge_incumbent_config
from .nodeio import CloudIO, HttpNodeIO, NodeIO, active_profile_of, engine_endpoints, switch_time_s
from .runstate import BLOCKS, EPOCHS, EVENTS, MANIFEST, PROGRESS, STATUS, RunState, load_run_state
from .schedule import BlockGrid, ContextSchedule
from .telemetry import Snapshot, config_matches, derive_telemetry, parse_gateway_state
from .util import JsonlLog, atomic_write_text

EXIT_OK, EXIT_CONFIG, EXIT_PREFLIGHT, EXIT_RESUME, EXIT_EXISTS, EXIT_STOPPED, EXIT_FAILED = 0, 2, 3, 4, 5, 130, 1


class SimulatedCrash(BaseException):
    pass


class ControllerAbort(RuntimeError):
    pass


@dataclass
class BlockRuntime:
    block_id: int
    arm: ArmSpec
    draw: Draw
    context_id: str
    epoch_start: int
    epoch_end: int
    first_epoch: int
    resumed: bool = False
    missed_epochs: list = field(default_factory=list)
    epochs_logged: int = 0
    n_switches: int = 0
    switch_time_total_s: float = 0.0
    n_failures: int = 0
    recorded: bool = False


@dataclass
class OpenEpoch:
    epoch: int
    block: BlockRuntime
    t_boundary_ns: int
    late_ms: float
    baseline: Snapshot
    config: Optional[dict] = None
    intended: Optional[dict] = None
    config_ok: bool = True
    switched: bool = False
    switch_time_s: Optional[float] = None
    activate: Optional[dict] = None
    activate_rtt_s: Optional[float] = None
    t_actuated_ns: Optional[int] = None
    errors: list = field(default_factory=list)
    obs: Optional[dict] = None
    decide_ms: Optional[float] = None
    repushed: bool = False


@dataclass
class NodeRuntime:
    spec: NodeSpec
    io: NodeIO
    active_profile: Optional[str] = None
    pushed_cfg: Optional[dict] = None
    block: Optional[BlockRuntime] = None
    open_epoch: Optional[OpenEpoch] = None
    incumbents: dict = field(default_factory=dict)
    consecutive_errors: int = 0

    @property
    def id(self) -> str:
        return self.spec.id


@dataclass
class StepResult:
    node: str
    epoch_rec: Optional[dict] = None
    block_end: Optional[dict] = None
    block_rec: Optional[dict] = None
    cloud: Optional[dict] = None


class Controller:
    def __init__(self, cfg: RunConfig, *, clock: Optional[Clock] = None,
                 io_factory: Optional[Callable[[NodeSpec, RunConfig], NodeIO]] = None,
                 cloud_io: Optional[CloudIO] = None, run_dir: Optional[str] = None,
                 incumbent_loader: Callable[[dict], Any] = load_incumbent,
                 fault_hook: Optional[Callable[[int], None]] = None,
                 log: Callable[[str], None] = lambda s: print(s, flush=True),
                 context: Optional[ContextSchedule] = None):
        self.cfg = cfg
        self.clock = clock or RealClock()
        self.run_dir = run_dir or cfg.run_dir
        self.grid = BlockGrid(cfg.epoch_ns, cfg.block_epochs, cfg.burn_in_epochs, cfg.n_blocks)
        self.law = make_design_law(cfg)
        base = os.path.dirname(cfg.source_path) if cfg.source_path else None
        self.context = context or ContextSchedule.load(cfg.context.ref, cfg.context.default, base)
        self.io_factory = io_factory or (lambda node, c: HttpNodeIO(node, c.control))
        self.cloud = cloud_io if cloud_io is not None else CloudIO(cfg.cloud.gateway, cfg.control)
        self.incumbent_loader = incumbent_loader
        self.fault_hook = fault_hook
        self.say = log
        self.stop = threading.Event()
        self.nodes = [NodeRuntime(spec=n, io=self.io_factory(n, cfg)) for n in cfg.nodes]
        self.t0_mono_ns = 0
        self.t0_wall_ns = 0
        self.session = 1
        self.abort_reason: Optional[str] = None
        self._logs: dict[str, JsonlLog] = {}


    def _path(self, name: str) -> str:
        return os.path.join(self.run_dir, name)

    def _open_logs(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        for name in (BLOCKS, EPOCHS, EVENTS):
            self._logs[name] = JsonlLog(self._path(name), durable=True)

    def _close_logs(self) -> None:
        for lg in self._logs.values():
            lg.close()

    def _event(self, event: str, **fields: Any) -> None:
        rec = {"event": event, "run_id": self.cfg.run_id, "session": self.session,
               "t_mono_ns": self.clock.mono_ns(), "wall_ns": self.clock.wall_ns(), **fields}
        self._logs[EVENTS].append(rec)

    def _set_status(self, status: str) -> None:
        atomic_write_text(self._path(STATUS), status + "\n")

    def _progress(self, line: str) -> None:
        with open(self._path(PROGRESS), "a", encoding="utf-8") as fh:
            fh.write(f"{self.clock.wall_ns() / 1e9:.3f} {line}\n")
        self.say(line)

    def _write_manifest(self, sessions: list) -> None:
        doc = {
            "run_id": self.cfg.run_id, "design_hash": self.cfg.design_hash,
            "design": self.cfg.design_dict(), "t0_mono_ns": self.t0_mono_ns, "t0_wall_ns": self.t0_wall_ns,
            "t0_boot_id": self.clock.boot_id(), "epoch_ns": self.cfg.epoch_ns, "total_epochs": self.grid.total_epochs,
            "block_epochs": self.cfg.block_epochs, "burn_in_epochs": self.cfg.burn_in_epochs,
            "n_blocks": self.cfg.n_blocks, "sessions": sessions,
        }
        atomic_write_text(self._path(MANIFEST), json.dumps(doc, indent=2, sort_keys=True) + "\n")


    def _preflight(self) -> None:
        profiles = self.cfg.profiles_to_prepare()
        problems: list[str] = []

        def one(node: NodeRuntime) -> Optional[str]:
            try:
                node.io.gateway_state()
            except Exception as exc:
                return f"{node.id}: gateway /state failed: {exc}"
            try:
                if self.cfg.control.prepare_engines:
                    node.io.prepare(profiles)
                st = node.io.engine_status()
                node.active_profile = active_profile_of(st)
            except Exception as exc:
                return f"{node.id}: engine manager failed: {exc}"
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(self.nodes))) as ex:
            for res in ex.map(one, self.nodes):
                if res:
                    problems.append(res)
        if problems:
            raise PreflightError(problems)


    def _plan_resume(self, state: RunState) -> int:
        m = state.manifest
        now_m, now_w = self.clock.mono_ns(), self.clock.wall_ns()
        self.t0_wall_ns = int(m["t0_wall_ns"])
        if m.get("t0_boot_id") == self.clock.boot_id():
            self.t0_mono_ns = int(m["t0_mono_ns"])
        else:
            self.t0_mono_ns = now_m - (now_w - self.t0_wall_ns)
        margin = int(self.cfg.control.resume_margin_s * 1e9)
        e_start = self.grid.next_boundary_epoch(self.t0_mono_ns, now_m, margin)
        total = self.grid.total_epochs
        self._event("resume", from_epoch=e_start, last_logged_epoch=state.last_epoch,
                    missed_epochs=[state.last_epoch + 1, e_start - 1] if e_start > state.last_epoch + 1 else [],
                    t0_mono_ns=self.t0_mono_ns)
        i = self.grid.block_of(e_start) if e_start < total else None
        for node in self.nodes:
            for (nid, bid), rec in sorted(state.blocks.items(), key=lambda kv: kv[0][1]):
                if nid != node.id or (nid, bid) in state.ended:
                    continue
                reenter = (self.cfg.control.on_resume == "continue_block" and i is not None and bid == i
                           and not self.grid.is_block_start(e_start))
                if reenter:
                    arm = self.cfg.arms.get(rec["arm_id"])
                    if arm is None:
                        reenter = False
                if reenter:
                    draw = self.law.draw(node.spec.index, bid, rec["context_id"])
                    if draw.arm_id != rec["arm_id"]:
                        self._event("draw_mismatch", node=node.id, block_id=bid, logged=rec["arm_id"],
                                    recomputed=draw.arm_id)
                        draw = Draw(rec["arm_id"], self.law.arm_ids.index(rec["arm_id"]),
                                    float(rec["propensity"]), float(rec["rng"]["draw"]), draw.entropy,
                                    self.law.label, rec["context_id"])
                    done = state.epochs_of(node.id)
                    start = self.grid.block_start(bid)
                    missed = [x for x in range(start, e_start) if x not in done]
                    node.block = BlockRuntime(
                        block_id=bid, arm=arm, draw=draw, context_id=rec["context_id"], epoch_start=start,
                        epoch_end=self.grid.block_end(bid), first_epoch=e_start, resumed=True,
                        missed_epochs=missed, epochs_logged=sum(1 for x in range(start, e_start) if x in done),
                        recorded=True)
                    self._begin_block_hook(node, node.block, e_start, resumed=True)
                    self._event("reenter_block", node=node.id, block_id=bid, arm_id=arm.id,
                                missed_epochs=missed)
                else:
                    self._logs[BLOCKS].append(self._interrupted_end(node.id, bid, rec, state))
        return e_start

    def _interrupted_end(self, node_id: str, block_id: int, rec: dict, state: RunState) -> dict:
        done = sorted(x for x in state.epochs_of(node_id)
                      if self.grid.block_of(x) == block_id)
        full = len(done) == self.cfg.block_epochs
        return {"rec": "block_end", "run_id": self.cfg.run_id, "node": node_id, "block_id": block_id,
                "status": "complete" if full else "interrupted", "recovered": True,
                "epochs_expected": self.cfg.block_epochs,
                "epochs_logged": len(done), "missed_epochs": [], "switches": None,
                "switch_time_total_s": None, "actuation_failures": None, "session": self.session,
                "t_mono_ns": self.clock.mono_ns(), "wall_ns": self.clock.wall_ns()}


    def run(self, resume: bool = False) -> int:
        cfg = self.cfg
        os.makedirs(self.run_dir, exist_ok=True)
        existing = load_run_state(self.run_dir)
        has_logs = any(os.path.exists(self._path(n)) and os.path.getsize(self._path(n)) > 0
                       for n in (BLOCKS, EPOCHS))
        if (existing is not None or has_logs) and not resume:
            self.say(f"run directory {self.run_dir} already holds a run; use --resume to continue it")
            return EXIT_EXISTS
        if existing is not None and existing.manifest.get("design_hash") != cfg.design_hash:
            self.say("the run configuration differs from the one the run started with (design hash)")
            return EXIT_RESUME
        self._open_logs()
        try:
            return self._run(existing)
        finally:
            self._close_logs()

    def _run(self, existing: Optional[RunState]) -> int:
        cfg = self.cfg
        total = self.grid.total_epochs
        sessions = list(existing.manifest.get("sessions", [])) if existing else []
        self.session = len(sessions) + 1
        src = cfg.source_path
        if src and os.path.exists(src):
            with open(src, "r", encoding="utf-8") as fh:
                atomic_write_text(self._path(f"run-config.session{self.session}{os.path.splitext(src)[1]}"), fh.read())
        sess = {"n": self.session, "pid": os.getpid(), "host": socket.gethostname(),
                "python": platform.python_version(), "start_wall_ns": self.clock.wall_ns(),
                "resume": existing is not None}
        try:
            self._preflight()
        except PreflightError as exc:
            self._open_event_safe("preflight_failed", problems=exc.problems)
            self.say("preflight failed:\n  " + "\n  ".join(exc.problems))
            return EXIT_PREFLIGHT

        if existing is None:
            self.t0_mono_ns = self.clock.mono_ns() + int(cfg.control.start_delay_s * 1e9)
            self.t0_wall_ns = self.clock.wall_ns() + int(cfg.control.start_delay_s * 1e9)
            e_start = 0
            sessions.append(sess)
            self._write_manifest(sessions)
            self._event("start", t0_mono_ns=self.t0_mono_ns, t0_wall_ns=self.t0_wall_ns,
                        design_hash=cfg.design_hash, arms=list(self.law.arm_ids),
                        propensities=self.law.propensities(cfg.context.default))
        else:
            e_start = self._plan_resume(existing)
            sess["resume_from_epoch"] = e_start
            sessions.append(sess)
            self._write_manifest(sessions)
        self._set_status("RUNNING")
        with open(self._path("controller.pid"), "w") as fh:
            fh.write(f"{os.getpid()}\n")

        status = "DONE"
        code = EXIT_OK
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=cfg.control.workers or max(1, len(self.nodes)), thread_name_prefix="node")
        lockstep = isinstance(self.clock, FakeClock)
        running: dict = {}
        try:
            for e in range(e_start, total + 1):
                t_b = self.grid.boundary_ns(self.t0_mono_ns, e)
                if not self.clock.wait_until(t_b, self.stop):
                    status, code = "STOPPED", EXIT_STOPPED
                    break
                if self.fault_hook is not None:
                    self.fault_hook(e)
                if self.stop.is_set():
                    status, code = "STOPPED", EXIT_STOPPED
                    break
                futs: list = []
                late = self.clock.mono_ns() - t_b
                for node in self.nodes:
                    busy = running.get(node.id)
                    if e > e_start and e < total and ((busy is not None and not busy.done()) or late > self.cfg.epoch_ns):
                        self._event("missed_boundary", node=node.id, epoch=e, late_ms=late / 1e6)
                        continue
                    fut = pool.submit(self._run_node, node, e, t_b, not lockstep)
                    running[node.id] = fut
                    futs.append(fut)
                if lockstep:
                    self._write_results(e, [f.result() for f in futs])
                if self.abort_reason:
                    status, code = "FAILED", EXIT_FAILED
                    break
        except SimulatedCrash:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        except ControllerAbort as exc:
            status, code = "FAILED", EXIT_FAILED
            self.abort_reason = str(exc)
        except BaseException as exc:
            status, code = "FAILED", EXIT_FAILED
            self.abort_reason = f"{type(exc).__name__}: {exc}"
            self._event("error", message=self.abort_reason)
            pool.shutdown(wait=True)
            self._finish(status)
            raise
        pool.shutdown(wait=True)
        if status != "DONE":
            self._event("stop", status=status, reason=self.abort_reason)
        self._finish(status)
        return code

    def _open_event_safe(self, event: str, **fields: Any) -> None:
        if EVENTS in self._logs:
            self._event(event, **fields)

    def _finish(self, status: str) -> None:
        if status != "DONE":
            for node in self.nodes:
                if node.block is not None:
                    blk = node.block
                    self._logs[BLOCKS].append(self._block_end_record(node, blk, "interrupted"))
                    node.block = None
        self._set_status(status)
        self._event("finish", status=status)
        self._progress(f"{status} run {self.cfg.run_id}")

    def request_stop(self) -> None:
        self.stop.set()


    def _poll(self, node: NodeRuntime) -> Snapshot:
        clk = self.clock
        t_start = clk.mono_ns()
        snap = Snapshot(t_mono_ns=t_start, wall_ns=clk.wall_ns())
        try:
            raw = node.io.gateway_state()
            snap.t_mono_ns, snap.wall_ns = clk.mono_ns(), clk.wall_ns()
            snap.gateway_raw = raw
            snap.gateway = parse_gateway_state(raw)
        except Exception as exc:
            snap.errors.append(f"gateway_state: {exc}")
        try:
            st = node.io.engine_status()
            snap.status_raw = st
            snap.profile = active_profile_of(st)
            snap.endpoint = engine_endpoints(st).get(snap.profile) if snap.profile else None
        except Exception as exc:
            snap.errors.append(f"engine_status: {exc}")
        if snap.endpoint:
            try:
                text = node.io.engine_metrics(snap.endpoint)
                snap.metrics = promtext.sum_by_name(promtext.parse(text))
            except Exception as exc:
                snap.errors.append(f"engine_metrics: {exc}")
        snap.poll_ms = (clk.mono_ns() - t_start) / 1e6
        return snap

    def _rebaseline_engine(self, node: NodeRuntime, snap: Snapshot) -> Snapshot:
        base = copy.copy(snap)
        try:
            st = node.io.engine_status()
            base.status_raw = st
            base.profile = active_profile_of(st)
            base.endpoint = engine_endpoints(st).get(base.profile) if base.profile else None
            base.metrics = None
            if base.endpoint:
                base.metrics = promtext.sum_by_name(promtext.parse(node.io.engine_metrics(base.endpoint)))
        except Exception as exc:
            base.metrics = None
            base.errors = list(base.errors) + [f"baseline: {exc}"]
        return base

    def _incumbent(self, node: NodeRuntime, arm: ArmSpec) -> Any:
        inst = node.incumbents.get(arm.id)
        if inst is None:
            inst = self.incumbent_loader(arm.incumbent)
            node.incumbents[arm.id] = inst
        return inst

    MAX_CONSECUTIVE_NODE_ERRORS = 5

    def _safe_step(self, node: NodeRuntime, e: int, t_b: int) -> StepResult:
        try:
            res = self._node_step(node, e, t_b)
            node.consecutive_errors = 0
            return res
        except Exception as exc:
            node.consecutive_errors += 1
            node.open_epoch = None
            self._event("node_error", node=node.id, epoch=e, message=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc()[-1500:], consecutive=node.consecutive_errors)
            if node.consecutive_errors >= self.MAX_CONSECUTIVE_NODE_ERRORS:
                self.abort_reason = f"{node.id} failed {node.consecutive_errors} boundaries in a row: {exc}"
            return StepResult(node.id)

    def _node_step(self, node: NodeRuntime, e: int, t_b: int) -> StepResult:
        res = StepResult(node.id)
        clk = self.clock
        late_ms = (clk.mono_ns() - t_b) / 1e6
        snap = self._poll(node)
        total = self.grid.total_epochs

        oe = node.open_epoch
        tel = derive_telemetry(oe.baseline if oe else None, snap)
        if oe is not None:
            res.epoch_rec = self._epoch_record(node, oe, snap, tel)
            oe.block.epochs_logged += 1
            node.open_epoch = None

        blk = node.block
        if blk is not None and e >= blk.epoch_end:
            complete = (blk.epochs_logged == self.cfg.block_epochs and not blk.missed_epochs)
            res.block_end = self._block_end_record(node, blk, "complete" if complete else "partial")
            node.block = None
        if e >= total:
            return res

        if node.block is None:
            if not self.grid.is_block_start(e):
                return res
            node.block = self._start_block(node, e)
        blk = node.block

        oe = OpenEpoch(epoch=e, block=blk, t_boundary_ns=t_b, late_ms=late_ms, baseline=snap)
        arm = blk.arm
        force = False
        in_force = (snap.gateway or {}).get("config")
        if node.pushed_cfg is not None and not config_matches(in_force, node.pushed_cfg):
            force = True
            oe.repushed = True
            oe.errors.append("config_mismatch: gateway reports a configuration other than the last push")
        if snap.profile is not None:
            node.active_profile = snap.profile

        if arm.is_incumbent:
            desired = self._decide(node, blk, e, snap, tel, oe)
        else:
            desired = arm.config_object(blk.block_id)
        oe.intended = desired
        if desired is not None and (force or desired != node.pushed_cfg):
            self._actuate(node, desired, oe)
        oe.config = copy.deepcopy(node.pushed_cfg)
        oe.config_ok = (desired is not None and node.pushed_cfg == desired)
        node.open_epoch = oe
        if oe.switched:
            oe.baseline = self._rebaseline_engine(node, snap)

        if not blk.recorded:
            blk.recorded = True
            res.block_rec = self._block_record(node, blk, oe, e)
            res.cloud = {node.id: {"arm_id": arm.id, "block_id": blk.block_id, "epoch": e}}
        return res

    def _start_block(self, node: NodeRuntime, e: int) -> BlockRuntime:
        bid = self.grid.block_of(e)
        ctx = self.context.context_at(self.grid.block_start(bid))
        draw = self.law.draw(node.spec.index, bid, ctx)
        arm = self.cfg.arms[draw.arm_id]
        blk = BlockRuntime(block_id=bid, arm=arm, draw=draw, context_id=ctx, epoch_start=e,
                           epoch_end=self.grid.block_end(bid), first_epoch=e)
        self._begin_block_hook(node, blk, e, resumed=False)
        return blk

    def _begin_block_hook(self, node: NodeRuntime, blk: BlockRuntime, e: int, resumed: bool) -> None:
        if not blk.arm.is_incumbent:
            return
        try:
            hook = getattr(self._incumbent(node, blk.arm), "begin_block", None)
            if callable(hook):
                hook({"node": node.id, "block_id": blk.block_id, "epoch": e, "context_id": blk.context_id,
                      "arm_id": blk.arm.id, "resumed": resumed})
        except Exception as exc:
            self._event("incumbent_hook_error", node=node.id, block_id=blk.block_id, message=str(exc))

    def _build_obs(self, node: NodeRuntime, blk: BlockRuntime, e: int, snap: Snapshot, tel: dict) -> Obs:
        g = snap.gateway or {}
        counters = g.get("counters") or {}
        return Obs(
            node=node.id, node_index=node.spec.index, epoch=e, block_id=blk.block_id,
            epoch_in_block=self.grid.epoch_in_block(e), context_id=blk.context_id,
            epoch_s=self.cfg.epoch_s, t_mono_ns=snap.t_mono_ns, wall_ns=snap.wall_ns,
            arrival_rate=tel.get("arrival_rate"), memory_budget_mib=node.spec.memory_budget_mib,
            gateway={"queue_len": g.get("queue_len"), "in_flight": g.get("in_flight"),
                     "arrivals": tel.get("arrivals"), "completed": tel.get("completed"),
                     "escalated": tel.get("escalated"), "errors": tel.get("errors"),
                     "counters": counters, "recent": g.get("recent"), "config": g.get("config"),
                     "raw": snap.gateway_raw},
            engine=tel.get("engine"), profile=snap.profile, config=copy.deepcopy(node.pushed_cfg),
            first_epoch_of_block=(self.grid.epoch_in_block(e) == 0),
        )

    def _decide(self, node: NodeRuntime, blk: BlockRuntime, e: int, snap: Snapshot, tel: dict,
                oe: OpenEpoch) -> Optional[dict]:
        arm = blk.arm
        obs = self._build_obs(node, blk, e, snap, tel)
        oe.obs = {k: v for k, v in obs.items() if k != "gateway"}
        oe.obs["gateway"] = {k: v for k, v in obs["gateway"].items() if k != "raw"}
        prev = node.pushed_cfg if (node.pushed_cfg and node.pushed_cfg.get("arm_id") == arm.id) else None
        t0 = self.clock.mono_ns()
        try:
            ret = self._incumbent(node, arm).decide(obs)
            desired = merge_incumbent_config(ret, arm, prev, blk.block_id)
        except Exception as exc:
            oe.errors.append(f"decide: {type(exc).__name__}: {exc}")
            blk.n_failures += 1
            self._event("incumbent_error", node=node.id, epoch=e, arm_id=arm.id, message=str(exc))
            if self.cfg.control.on_incumbent_error == "abort":
                self.abort_reason = f"incumbent {arm.id} failed on {node.id} epoch {e}: {exc}"
                return None
            if prev is not None:
                desired = dict(prev)
                desired["block_id"] = blk.block_id
            else:
                desired = None
        oe.decide_ms = (self.clock.mono_ns() - t0) / 1e6
        return desired

    def _actuate(self, node: NodeRuntime, desired: dict, oe: OpenEpoch) -> bool:
        blk = oe.block
        clk = self.clock
        if desired.get("profile") != node.active_profile:
            t0 = clk.mono_ns()
            try:
                resp = node.io.activate(desired["profile"])
                oe.activate = resp
                oe.switch_time_s = switch_time_s(resp)
                oe.activate_rtt_s = (clk.mono_ns() - t0) / 1e9
                oe.switched = True
                node.active_profile = desired["profile"]
                blk.n_switches += 1
                blk.switch_time_total_s += oe.switch_time_s or 0.0
            except Exception as exc:
                oe.errors.append(f"activate: {exc}")
                blk.n_failures += 1
                return False
        try:
            node.io.gateway_config(desired)
            node.pushed_cfg = copy.deepcopy(desired)
        except Exception as exc:
            oe.errors.append(f"config: {exc}")
            blk.n_failures += 1
            return False
        oe.t_actuated_ns = clk.mono_ns()
        return True


    def _epoch_record(self, node: NodeRuntime, oe: OpenEpoch, snap: Snapshot, tel: dict) -> dict:
        blk = oe.block
        e = oe.epoch
        return {
            "rec": "epoch", "run_id": self.cfg.run_id, "session": self.session, "node": node.id,
            "node_index": node.spec.index, "epoch": e, "block_id": blk.block_id,
            "epoch_in_block": self.grid.epoch_in_block(e), "burn_in": self.grid.is_burn_in(e),
            "arm_id": blk.arm.id, "kind": blk.arm.kind, "context_id": blk.context_id,
            "propensity": blk.draw.propensity, "design": self.law.label,
            "config": oe.config, "intended": oe.intended if oe.intended != oe.config else None,
            "config_ok": oe.config_ok, "repushed": oe.repushed,
            "switched": oe.switched, "switch_time_s": oe.switch_time_s, "activate": oe.activate,
            "activate_rtt_s": oe.activate_rtt_s,
            "t_start_mono_ns": oe.t_boundary_ns, "t_actuated_mono_ns": oe.t_actuated_ns,
            "t_end_mono_ns": snap.t_mono_ns, "wall_end_ns": snap.wall_ns, "late_ms": oe.late_ms,
            "end_late_ms": (snap.t_mono_ns - oe.t_boundary_ns - self.cfg.epoch_ns) / 1e6,
            "overrun": (snap.t_mono_ns - oe.t_boundary_ns) > 1.5 * self.cfg.epoch_ns,
            "decide_ms": oe.decide_ms, "obs": oe.obs, "telemetry": tel,
            "gateway_state": snap.gateway_raw, "poll_ms": snap.poll_ms,
            "errors": oe.errors + snap.errors,
        }

    def _block_record(self, node: NodeRuntime, blk: BlockRuntime, oe: OpenEpoch, e: int) -> dict:
        lo, hi = self.grid.measurement_range(blk.block_id)
        d = blk.draw
        return {
            "rec": "block", "run_id": self.cfg.run_id, "session": self.session, "node": node.id,
            "node_index": node.spec.index, "block_id": blk.block_id, "epoch_start": blk.epoch_start,
            "epoch_end": blk.epoch_end, "burn_in_epochs": self.cfg.burn_in_epochs,
            "measure_start_epoch": lo, "measure_end_epoch": hi, "arm_id": blk.arm.id,
            "kind": blk.arm.kind, "config": oe.config, "intended_config": oe.intended, "propensity": d.propensity,
            "context_id": blk.context_id, "design": self.law.label, "epsilon": self.law.epsilon,
            "incumbent_arm": self.law.incumbent, "seed": d.entropy[0],
            "rng": {"entropy": list(d.entropy), "draw": d.u, "arm_index": d.arm_index},
            "switch_time_s": oe.switch_time_s, "profile_switched": oe.switched, "activate": oe.activate,
            "t_mono_ns": oe.t_boundary_ns, "wall_ns": self.clock.wall_ns(), "late_ms": oe.late_ms,
            "errors": oe.errors,
        }

    def _block_end_record(self, node: NodeRuntime, blk: BlockRuntime, status: str) -> dict:
        return {
            "rec": "block_end", "run_id": self.cfg.run_id, "session": self.session, "node": node.id,
            "block_id": blk.block_id, "status": status,
            "epochs_expected": self.cfg.block_epochs, "epochs_logged": blk.epochs_logged,
            "missed_epochs": blk.missed_epochs, "resumed": blk.resumed, "switches": blk.n_switches,
            "switch_time_total_s": blk.switch_time_total_s, "actuation_failures": blk.n_failures,
            "t_mono_ns": self.clock.mono_ns(), "wall_ns": self.clock.wall_ns(),
        }

    def _run_node(self, node: NodeRuntime, e: int, t_b: int, write_now: bool) -> StepResult:
        res = self._safe_step(node, e, t_b)
        if write_now:
            self._write_results(e, [res])
        return res

    def _write_results(self, e: int, results: list) -> None:
        cloud_nodes: dict = {}
        for r in results:
            if r.epoch_rec:
                self._logs[EPOCHS].append(r.epoch_rec)
            if r.block_end:
                self._logs[BLOCKS].append(r.block_end)
            if r.block_rec:
                self._logs[BLOCKS].append(r.block_rec)
            if r.cloud:
                cloud_nodes.update(r.cloud)
        if cloud_nodes and self.cloud.enabled:
            try:
                self.cloud.notify(cloud_nodes)
            except Exception as exc:
                self._event("cloud_notify_failed", epoch=e, message=str(exc))
        first = self.nodes[0].id
        if (not results or any(r.node == first for r in results)) and (self.grid.is_block_start(e) or e % 10 == 0):
            self._progress(f"epoch {e}/{self.grid.total_epochs} block {self.grid.block_of(e)}/{self.cfg.n_blocks}")


class PreflightError(RuntimeError):
    def __init__(self, problems: list):
        self.problems = problems
        super().__init__("; ".join(problems))


def _plan(cfg: RunConfig, ctx: ContextSchedule, out: Optional[str]) -> int:
    law = make_design_law(cfg)
    grid = BlockGrid(cfg.epoch_ns, cfg.block_epochs, cfg.burn_in_epochs, cfg.n_blocks)
    hours = grid.total_epochs * cfg.epoch_s / 3600.0
    print(f"run {cfg.run_id}: design {law.label}, {len(law.arm_ids)} arms, {len(cfg.nodes)} nodes")
    print(f"  epochs {grid.total_epochs} x {cfg.epoch_s} s = {hours:.2f} h; blocks {cfg.n_blocks} x l={cfg.block_epochs}, "
          f"burn-in b={cfg.burn_in_epochs}; node-blocks {cfg.n_blocks * len(cfg.nodes)}")
    cdef = cfg.context.default
    for a, p in zip(law.arm_ids, law.propensities(cdef)):
        print(f"  e({a}) = {p:.6f}  expected node-blocks {p * cfg.n_blocks * len(cfg.nodes):.1f}")
    draws = []
    counts: dict[str, int] = {}
    for n in cfg.nodes:
        for i in range(cfg.n_blocks):
            cid = ctx.context_at(grid.block_start(i))
            d = law.draw(n.index, i, cid)
            counts[d.arm_id] = counts.get(d.arm_id, 0) + 1
            draws.append({"node": n.id, "block_id": i, "arm_id": d.arm_id, "propensity": d.propensity,
                          "draw": d.u, "context_id": cid})
    print("  planned draws per arm: " + ", ".join(f"{a}={counts.get(a, 0)}" for a in law.arm_ids))
    if out:
        with open(out, "w") as fh:
            json.dump({"run_id": cfg.run_id, "design_hash": cfg.design_hash, "draws": draws}, fh)
        print(f"  wrote {out}")
    for w in cfg.warnings:
        print(f"  warning: {w}")
    return 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="ciara_control.controller", description='The CIARA controller: drives epochs and blocks, draws arms, actuates nodes, logs.')
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "plan", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--run-dir", default=None)
    sub.choices["plan"].add_argument("--out", default=None, help="write all planned draws to this JSON file")
    r = sub.choices["run"]
    r.add_argument("--resume", action="store_true")
    r.add_argument("--dry-run", action="store_true", help="in-process mock endpoints (no network, no GPU)")
    r.add_argument("--real-time", action="store_true", help="dry run on the real clock (default: simulated clock)")
    r.add_argument("--no-prepare", action="store_true", help="skip POST /prepare")
    args = ap.parse_args(argv)

    overrides: dict = {}
    if args.run_dir:
        overrides["run_dir"] = args.run_dir
    try:
        cfg = load_config(args.config, overrides)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return EXIT_CONFIG
    base = os.path.dirname(cfg.source_path) if cfg.source_path else None
    try:
        ctx = ContextSchedule.load(cfg.context.ref, cfg.context.default, base)
    except OSError as exc:
        print(f"context schedule: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    for w in cfg.warnings:
        print(f"warning: {w}", file=sys.stderr)
    bad = ctx.misaligned_segments(cfg.block_epochs)
    if bad:
        print(f"warning: context segments not aligned to blocks: {bad}", file=sys.stderr)
    if args.cmd == "validate":
        print(f"ok: run {cfg.run_id}, design hash {cfg.design_hash[:12]}")
        return 0
    if args.cmd == "plan":
        return _plan(cfg, ctx, args.out)

    if args.no_prepare:
        cfg = dataclasses.replace(cfg, control=dataclasses.replace(cfg.control, prepare_engines=False))
    clock: Clock = RealClock()
    io_factory = None
    cloud = None
    if args.dry_run:
        from .mocks.inproc import InProcCloud, make_io_factory
        clock = RealClock() if args.real_time else FakeClock()
        io_factory = make_io_factory(clock)
        cloud = InProcCloud()
    ctl = Controller(cfg, clock=clock, io_factory=io_factory, cloud_io=cloud, context=ctx)
    signal.signal(signal.SIGTERM, lambda *_: ctl.request_stop())
    signal.signal(signal.SIGINT, lambda *_: ctl.request_stop())
    return ctl.run(resume=args.resume)


if __name__ == "__main__":
    sys.exit(main())
