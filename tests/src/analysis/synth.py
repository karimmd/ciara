from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

T0 = 1_000_000_000_000
EPOCH_S = 10
EPOCH_NS = EPOCH_S * 10 ** 9
L, B, K = 6, 2, 3


def ns(epoch: float) -> int:
    return int(T0 + epoch * EPOCH_NS)


def wjsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def schedule_meta() -> Dict[str, Any]:
    return {"type": "meta", "classes": {"A": {"radio": "nbiot"}, "B": {"radio": "ltem"}, "C": {"radio": "nr"}},
            "links": {"nbiot_cl0": {"rate_up_kbps": 66.0, "rate_down_kbps": 26.0},
                      "ltem_good": {"rate_up_kbps": 125.0, "rate_down_kbps": 125.0},
                      "nr_near": {"rate_up_kbps": 100000.0, "rate_down_kbps": 880000.0}}}


LINK = {"A": "nbiot_cl0", "B": "ltem_good", "C": "nr_near"}
DEADLINE_MS = {"A": 60000, "B": 10000, "C": 5000}


def client_rec(req: str, cls: str, family: int, arrival_epoch: float, latency_s: float, item: str = "it-1",
               status: int = 200, error: Optional[str] = None, reply: str = "ok", finish: str = "stop") -> Dict[str, Any]:
    t_arr = ns(arrival_epoch)
    t_send = t_arr + 50_000_000
    t_last = t_arr + int(latency_s * 1e9)
    rec = {"req": req, "device": req.split("-")[0], "node": 1, "class": cls, "family": family, "item": item,
           "deadline_ms": DEADLINE_MS[cls], "link": LINK[cls], "t_arrival_mono_ns": t_arr, "t_send_mono_ns": t_send,
           "status": status, "error": error, "bytes_up": 5000, "bytes_down": 800, "reply_text": reply,
           "finish_reason": finish, "usage": {"prompt_tokens": 1000, "completion_tokens": 20},
           "cpu_prep": {"dev_ms": 20.0}, "cpu_post": {"dev_ms": 2.0}}
    if error is None:
        rec["t_last_byte_mono_ns"] = t_last
        rec["t_done_mono_ns"] = t_last + 1000
    else:
        rec["t_done_mono_ns"] = t_last
    return rec


def edge_rec(req: str, t_arrive_epoch: float, served_by: str = "edge", arm: str = "A1", block: int = 0,
             status: int = 200, router: str = "kappa", **extra: Any) -> Dict[str, Any]:
    t = ns(t_arrive_epoch)
    r = {"req": req, "node": 1, "t_arrive": t, "t_admit": t + 1_000_000, "t_done": t + 2_000_000_000,
         "served_by": served_by, "routed": served_by, "escalated_after_edge": False, "arm_id": arm, "block_id": block,
         "arm_id_arrival": arm, "block_id_arrival": block, "profile": "p1", "batch_cap": 8, "router": router,
         "status": status}
    r.update(extra)
    return r


def cloud_rec(req: str, t_arrive_epoch: float, node: str = "E1") -> Dict[str, Any]:
    t = ns(t_arrive_epoch)
    return {"rec": "request", "req_id": req, "node": node, "arm_id": "A1", "block_id": 0, "t_arrive": t,
            "t_admit": t + 1000, "t_done": t + 1_500_000_000, "http_status": 200, "served_by": "cloud"}


def write_run(run_dir: str, clients: List[Dict[str, Any]], edge: List[Dict[str, Any]], cloud: List[Dict[str, Any]],
              arms: Optional[List[str]] = None, gpu_samples: Optional[List[Dict[str, Any]]] = None) -> None:
    arms = arms or ["A1", "A2", "I2"]
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "manifest.json"), "w") as fh:
        json.dump({"run_id": "synth", "t0_mono_ns": T0, "epoch_ns": EPOCH_NS, "block_epochs": L, "burn_in_epochs": B,
                   "n_blocks": K, "total_epochs": K * L, "design": {"nodes": [{"id": "E1", "index": 1}]}}, fh)
    blocks, epochs = [], []
    for i in range(K):
        arm = arms[i]
        cfg = {"arm_id": arm, "profile": "p1" if i != 1 else "p2", "batch_cap": 8 + 8 * (i % 2),
               "escalate_family2": i == 0, "router": "I2" if arm == "I2" else "kappa", "block_id": i}
        blocks.append({"rec": "block", "run_id": "synth", "node": "E1", "node_index": 1, "block_id": i,
                       "epoch_start": i * L, "epoch_end": (i + 1) * L, "arm_id": arm, "kind": "static",
                       "config": cfg, "propensity": 1 / 3, "context_id": "pooled", "design": "screening", "epsilon": None,
                       "seed": 7, "rng": {"draw": 0.1 * (i + 1)}, "profile_switched": i == 1,
                       "switch_time_s": 1.5 if i == 1 else None})
        blocks.append({"rec": "block_end", "node": "E1", "block_id": i, "status": "complete", "epochs_logged": L,
                       "switches": 1 if i == 1 else 0, "switch_time_total_s": 1.5 if i == 1 else 0.0,
                       "actuation_failures": 0})
        for k in range(L):
            e = i * L + k
            epochs.append({"rec": "epoch", "node": "E1", "epoch": e, "block_id": i, "epoch_in_block": k,
                           "burn_in": k < B, "arm_id": arm, "config": cfg, "config_ok": True,
                           "switched": (i == 1 and k == 0), "switch_time_s": 1.5 if (i == 1 and k == 0) else None,
                           "errors": []})
    wjsonl(os.path.join(run_dir, "blocks.jsonl"), blocks)
    wjsonl(os.path.join(run_dir, "epochs.jsonl"), epochs)
    wjsonl(os.path.join(run_dir, "clients", "n1A000.jsonl"), [c for c in clients if c["class"] == "A"])
    wjsonl(os.path.join(run_dir, "clients", "n1B000.jsonl"), [c for c in clients if c["class"] == "B"])
    wjsonl(os.path.join(run_dir, "clients", "n1C000.jsonl"), [c for c in clients if c["class"] == "C"])
    wjsonl(os.path.join(run_dir, "clients", "client-1-A.monitor.jsonl"), [{"not": "a request"}])
    wjsonl(os.path.join(run_dir, "edge-gw-1.jsonl"), edge)
    wjsonl(os.path.join(run_dir, "cloud-gw.jsonl"), cloud)
    if gpu_samples is not None:
        wjsonl(os.path.join(run_dir, "monitor-gpu.jsonl"), gpu_samples)
    sched = os.path.join(run_dir, "schedule")
    wjsonl(os.path.join(sched, "schedule.jsonl"), [schedule_meta()])
