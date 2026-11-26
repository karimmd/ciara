from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from typing import Optional

import aiohttp

from ..util import dumps


async def one(session: aiohttp.ClientSession, url: str, node: str, k: int, out, stats: dict) -> None:
    body = {"model": "mock", "messages": [{"role": "user", "content": f"request {k} from {node}"}],
            "stream": True, "max_tokens": 512, "temperature": 0, "seed": 0, "mock_id": k,
            "chat_template_kwargs": {"enable_thinking": False}}
    hdr = {"X-Ciara-Req": f"{node}-{k}", "X-Ciara-Device": f"dev{k % 9}", "X-Ciara-Class": "ABC"[k % 3],
           "X-Ciara-Family": str(1 + k % 2), "X-Ciara-Item": f"item{k % 50}", "X-Ciara-Deadline-Ms": "10000",
           "X-Ciara-Node": node}
    rec = {"req_id": hdr["X-Ciara-Req"], "node": node, "t_send": time.monotonic_ns(), "status": None,
           "t_first_byte": None, "t_last_byte": None, "bytes": 0, "error": None, "has_usage_event": False}
    try:
        async with session.post(url + "/v1/chat/completions", json=body, headers=hdr) as r:
            rec["status"] = r.status
            async for chunk in r.content.iter_any():
                if rec["t_first_byte"] is None:
                    rec["t_first_byte"] = time.monotonic_ns()
                rec["bytes"] += len(chunk)
                if b'"usage"' in chunk:
                    rec["has_usage_event"] = True
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["t_last_byte"] = time.monotonic_ns()
    stats["ok" if rec["status"] == 200 and not rec["error"] else "failed"] += 1
    out.write(dumps(rec) + "\n")


async def node_loop(session, url, node, rate, duration, rng, out, stats, tasks):
    t_end = time.monotonic() + duration
    k = 0
    while time.monotonic() < t_end:
        await asyncio.sleep(rng.expovariate(rate))
        k += 1
        tasks.append(asyncio.create_task(one(session, url, node, k, out, stats)))


async def amain(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    nodes = args.nodes.split(",")
    stats = {"ok": 0, "failed": 0}
    tasks: list = []
    with open(args.out, "w", buffering=1) as out:
        conn = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=120)) as session:
            loops = [asyncio.create_task(node_loop(session, args.url, n, args.rate, args.duration,
                                                   random.Random(rng.random()), out, stats, tasks)) for n in nodes]
            if args.burst_n:
                await asyncio.sleep(args.burst_at)
                for j in range(args.burst_n):
                    tasks.append(asyncio.create_task(one(session, args.url, nodes[j % len(nodes)], 100000 + j, out, stats)))
            await asyncio.gather(*loops)
            await asyncio.gather(*tasks)
    print(json.dumps(stats))
    return 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description='Streaming load generator that plays the edge gateways towards the cloud gateway (dry run).')
    ap.add_argument("--url", required=True)
    ap.add_argument("--nodes", default="E1,E2,E3,E4,E5")
    ap.add_argument("--rate", type=float, default=3.0, help="requests per second per node")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--burst-at", type=float, default=20.0)
    ap.add_argument("--burst-n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    return asyncio.run(amain(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
