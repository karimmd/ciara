from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .grid import Grid
from .quality import score_request

NS = 1e9


def _num(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def node_id_map(manifest: Dict[str, Any]) -> Dict[int, str]:
    nodes = ((manifest.get("design") or {}).get("nodes")) or []
    out = {int(n["index"]): str(n["id"]) for n in nodes if "index" in n and "id" in n}
    return out


def block_table(blocks: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    return {(r["node"], int(r["block_id"])): r for r in blocks if r.get("rec") == "block"}


def arm_at(blocks_by_node: Dict[str, List[Dict[str, Any]]], grid: Grid, node: str, t_ns: Optional[int]) -> Optional[str]:
    if t_ns is None:
        return None
    e = grid.epoch_of(t_ns)
    if e < 0 or e >= grid.total_epochs:
        return None
    b = grid.block_of(e)
    for r in blocks_by_node.get(node, []):
        if int(r["block_id"]) == b:
            return r.get("arm_id")
    return None


def build_requests(client: List[Dict[str, Any]], edge: List[Dict[str, Any]], cloud: List[Dict[str, Any]],
                   blocks: List[Dict[str, Any]], manifest: Dict[str, Any],
                   scorers: Dict[int, Callable], energy: Optional[Dict[str, Dict[str, Any]]] = None,
                   latency_targets_s: Optional[Dict[int, float]] = None,
                   truncated_is_wrong: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grid = Grid.from_manifest(manifest)
    nid = node_id_map(manifest)
    bt = block_table(blocks)
    by_node: Dict[str, List[Dict[str, Any]]] = {}
    for (n, _), r in bt.items():
        by_node.setdefault(n, []).append(r)

    edge_by: Dict[str, List[Dict[str, Any]]] = {}
    for r in edge:
        edge_by.setdefault(r["req"], []).append(r)
    cloud_by: Dict[str, List[Dict[str, Any]]] = {}
    for r in cloud:
        if r.get("req_id"):
            cloud_by.setdefault(r["req_id"], []).append(r)
    client_ids: Dict[str, int] = {}
    for r in client:
        client_ids[r["req"]] = client_ids.get(r["req"], 0) + 1

    report: Dict[str, Any] = {
        "n_client": len(client), "n_edge": len(edge), "n_cloud": len(cloud),
        "client_duplicates": sorted(k for k, v in client_ids.items() if v > 1),
        "edge_duplicates": sorted(k for k, v in edge_by.items() if len(v) > 1),
        "cloud_duplicates": sorted(k for k, v in cloud_by.items() if len(v) > 1),
        "edge_without_client": sorted(set(edge_by) - set(client_ids)),
        "cloud_without_client": sorted(set(cloud_by) - set(client_ids)),
        "cloud_without_req_id": sum(1 for r in cloud if not r.get("req_id")),
        "client_without_edge": sorted(set(client_ids) - set(edge_by)),
    }
    rows: List[Dict[str, Any]] = []
    for c in client:
        req = c["req"]
        node_n = int(c["node"])
        node = nid.get(node_n, f"E{node_n}")
        fam = int(c["family"])
        e_list = edge_by.get(req, [])
        e = e_list[0] if e_list else None
        cl_list = cloud_by.get(req, [])
        cl = cl_list[0] if cl_list else None

        t_arr = c.get("t_arrival_mono_ns")
        t_end = c.get("t_last_byte_mono_ns") or c.get("t_done_mono_ns")
        ok = (c.get("error") is None) and c.get("status") == 200 and c.get("t_last_byte_mono_ns") is not None
        lat_s = (t_end - t_arr) / NS if (ok and t_end is not None and t_arr is not None) else None
        loc = grid.locate(t_end)
        loc_arr = grid.locate(t_arr)
        t_admit = e.get("t_admit") if e else None
        loc_adm = grid.locate(t_admit)
        full = e is not None and e.get("status") == 200 and e.get("reply_text") is not None
        q, qstat = score_request(scorers, fam, c.get("item"), e["reply_text"] if full else c.get("reply_text"),
                                 e.get("finish_reason") if full else c.get("finish_reason"), ok, truncated_is_wrong)
        served_by = e.get("served_by") if e else None
        cloud_used = bool((served_by == "cloud") or (cl is not None))
        deadline_s = (c["deadline_ms"] / 1e3) if c.get("deadline_ms") is not None else None
        en = (energy or {}).get(req)
        tgt = (latency_targets_s or {}).get(fam)
        row = {
            "req": req, "device": c.get("device"), "node": node, "node_n": node_n, "class": c.get("class"),
            "family": fam, "item": c.get("item"), "link": c.get("link"), "deadline_s": deadline_s,
            "epoch": loc["epoch"], "block_id": loc["block_id"], "epoch_in_block": loc["epoch_in_block"],
            "phase": loc["phase"],
            "block_id_arrival": loc_arr["block_id"], "epoch_arrival": loc_arr["epoch"],
            "block_id_admit": loc_adm["block_id"], "epoch_admit": loc_adm["epoch"],
            "t_arrival": t_arr, "t_send": c.get("t_send_mono_ns"), "t_complete": t_end,
            "status": c.get("status"), "error": c.get("error"), "ok": ok, "finish_reason": c.get("finish_reason"),
            "latency_s": lat_s, "bytes_up": c.get("bytes_up"), "bytes_down": c.get("bytes_down"),
            "wire": c.get("wire") or "full", "bytes_up_body": c.get("bytes_up_body"),
            "gw_wire": e.get("wire") if e else None, "gw_bytes_up_client": e.get("bytes_up_client") if e else None,
            "compose_ok": e.get("compose_ok") if e else None, "t_compose_ms": e.get("t_compose_ms") if e else None,
            "t_verify_ms": e.get("t_verify_ms") if e else None, "compose_error": e.get("compose_error") if e else None,
            "t_edge_body": e.get("t_body") if e else None,
            "prompt_tokens": (c.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (c.get("usage") or {}).get("completion_tokens"),
            "q": q, "q_status": qstat,
            "exceed": (None if tgt is None else (1.0 if (lat_s is None or lat_s > tgt) else 0.0)),
            "deadline_violation": (None if deadline_s is None else (1.0 if (lat_s is None or lat_s > deadline_s) else 0.0)),
            "energy_mj": en.get("energy_mj") if en else None, "e_norm": en.get("e_norm") if en else None,
            "found_edge": e is not None, "served_by": served_by, "routed": e.get("routed") if e else None,
            "escalated_after_edge": e.get("escalated_after_edge") if e else None,
            "cascade_budget": e.get("cascade_budget") if e else None, "cascade_reason": e.get("cascade_reason") if e else None,
            "edge_attempt_tokens": e.get("edge_attempt_tokens") if e else None, "edge_attempt_ms": e.get("edge_attempt_ms") if e else None,
            "edge_finish_reason": e.get("edge_finish_reason") if e else None,
            "edge_status": e.get("status") if e else None, "gw_arm_id": e.get("arm_id") if e else None,
            "gw_block_id": e.get("block_id") if e else None,
            "gw_arm_id_arrival": e.get("arm_id_arrival") if e else None,
            "gw_block_id_arrival": e.get("block_id_arrival") if e else None,
            "gw_profile": e.get("profile") if e else None, "gw_batch_cap": e.get("batch_cap") if e else None,
            "gw_router": e.get("router") if e else None,
            "queue_wait_ms": e.get("queue_wait_ms") if e else None,
            "delivery": (e.get("delivery") or "stream") if e else None,
            "compact_extracted": e.get("compact_extracted") if e else None,
            "compact_bytes": e.get("compact_bytes") if e else None, "bytes_down_edge": e.get("bytes_down") if e else None,
            "race_probe_s": e.get("race_probe_s") if e else None, "score_u": e.get("score_u") if e else None,
            "router_info": e.get("router_info") if e else None, "router_error": e.get("router_error") if e else None,
            "t_edge_arrive": e.get("t_arrive") if e else None, "t_edge_admit": t_admit,
            "t_edge_done": e.get("t_done") if e else None,
            "found_cloud": cl is not None, "cloud_node": cl.get("node") if cl else None,
            "cloud_arm_id": cl.get("arm_id") if cl else None, "cloud_block_id": cl.get("block_id") if cl else None,
            "cloud_status": cl.get("http_status") if cl else None,
            "t_cloud_arrive": cl.get("t_arrive") if cl else None, "t_cloud_done": cl.get("t_done") if cl else None,
            "cloud_used": cloud_used,
        }
        row["arm_of_block"] = arm_at(by_node, grid, node, t_end)
        row["arm_of_block_at_admit"] = arm_at(by_node, grid, node, t_admit)
        rows.append(row)
    report["n_rows"] = len(rows)
    report["n_found_edge"] = sum(1 for r in rows if r["found_edge"])
    report["n_found_cloud"] = sum(1 for r in rows if r["found_cloud"])
    return rows, report
