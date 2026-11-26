from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from typing import Optional

from aiohttp import web

from ..gwcore import JsonlWriter
from .sim import SimNode


def _stamp(rec: dict) -> dict:
    rec["t_mono_ns"] = time.monotonic_ns()
    rec["wall_ns"] = time.time_ns()
    return rec


def make_manager_app(sim: SimNode, log: JsonlWriter, switch_scale: float = 1.0) -> web.Application:
    node = sim.node_id

    async def prepare(request: web.Request) -> web.Response:
        body = await request.json()
        out = sim.prepare(body.get("profiles", []))
        log.write(_stamp({"ev": "prepare", "node": node, "profiles": body.get("profiles")}))
        return web.json_response(out)

    async def status(request: web.Request) -> web.Response:
        return web.json_response(sim.status())

    async def activate(request: web.Request) -> web.Response:
        body = await request.json()
        prof = body.get("profile")
        if not prof:
            return web.json_response({"error": "profile required"}, status=400)
        t_in = time.monotonic_ns()
        st = sim.switch_time() * switch_scale if prof != sim.active else 0.0
        await asyncio.sleep(st)
        out = sim.activate(prof, switch_s=st)
        log.write(_stamp({"ev": "activate", "node": node, "profile": prof, "switch_time_s": st, "t_in_mono_ns": t_in}))
        return web.json_response(out)

    async def metrics(request: web.Request) -> web.Response:
        return web.Response(text=sim.metrics_text(request.match_info["profile"]), content_type="text/plain")

    app = web.Application()
    app.router.add_post("/prepare", prepare)
    app.router.add_get("/status", status)
    app.router.add_post("/activate", activate)
    app.router.add_get("/engine/{profile}/metrics", metrics)
    return app


def make_gateway_app(sim: SimNode, log: JsonlWriter) -> web.Application:
    node = sim.node_id

    async def config(request: web.Request) -> web.Response:
        body = await request.json()
        out = sim.set_config(body)
        log.write(_stamp({"ev": "config", "node": node, "config": body}))
        return web.json_response(out)

    async def state(request: web.Request) -> web.Response:
        return web.json_response(sim.state())

    async def health(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/config", config)
    app.router.add_get("/state", state)
    app.router.add_get("/healthz", health)
    return app


async def serve(args: argparse.Namespace) -> None:
    nodes = [n for n in args.nodes.split(",") if n]
    os.makedirs(args.log_dir, exist_ok=True)
    mlog = JsonlWriter(os.path.join(args.log_dir, "mock-manager.jsonl"))
    glog = JsonlWriter(os.path.join(args.log_dir, "mock-gateway.jsonl"))
    runners = []
    for k, nid in enumerate(nodes):
        mport, gport = args.manager_base_port + k, args.gateway_base_port + k
        sim = SimNode(nid, k + 1, now_s=time.monotonic, seed=args.seed, rate=args.rate,
                      switch_mean_s=args.switch_s, endpoint_base=f"http://127.0.0.1:{mport}")
        for app, port in ((make_manager_app(sim, mlog), mport), (make_gateway_app(sim, glog), gport)):
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", port).start()
            runners.append(runner)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    if args.ready_file:
        with open(args.ready_file, "w") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "nodes": nodes}) + "\n")
    await stop.wait()
    for r in runners:
        await r.cleanup()
    mlog.close()
    glog.close()


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description='Mock engine managers and edge gateway control APIs for the end-to-end dry run.')
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--manager-base-port", type=int, required=True)
    ap.add_argument("--gateway-base-port", type=int, required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--rate", type=float, default=1.0, help="arrivals per second per node")
    ap.add_argument("--switch-s", type=float, default=0.3, help="mean profile switch time, seconds")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--ready-file", default=None)
    args = ap.parse_args(argv)
    asyncio.run(serve(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
