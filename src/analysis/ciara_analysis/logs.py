from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, Iterator, List, Optional


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class RunLogs:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir

    def p(self, name: str) -> str:
        return os.path.join(self.run_dir, name)

    def manifest(self) -> Dict[str, Any]:
        m = read_json(self.p("manifest.json"))
        if m is None:
            raise FileNotFoundError(self.p("manifest.json"))
        return m

    def blocks(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.p("blocks.jsonl"))

    def epochs(self) -> List[Dict[str, Any]]:
        return [r for r in read_jsonl(self.p("epochs.jsonl")) if r.get("rec") == "epoch"]

    def events(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.p("events.jsonl"))

    def client_files(self) -> List[str]:
        out = []
        for path in sorted(glob.glob(self.p(os.path.join("clients", "*.jsonl")))):
            if os.path.basename(path).startswith("client-"):
                continue
            out.append(path)
        return out

    def client_records(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for path in self.client_files():
            out.extend(iter_jsonl(path))
        return out

    def edge_gateway_files(self) -> Dict[int, str]:
        out: Dict[int, str] = {}
        for path in glob.glob(self.p("edge-gw-*.jsonl")):
            m = re.fullmatch(r"edge-gw-(\d+)\.jsonl", os.path.basename(path))
            if m:
                out[int(m.group(1))] = path
        return dict(sorted(out.items()))

    def edge_records(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _, path in self.edge_gateway_files().items():
            out.extend(iter_jsonl(path))
        return out

    def edge_events(self) -> Dict[int, List[Dict[str, Any]]]:
        out: Dict[int, List[Dict[str, Any]]] = {}
        for n in self.edge_gateway_files():
            out[n] = read_jsonl(self.p(f"edge-gw-{n}.events.jsonl"))
        return out

    def edge_probes(self) -> Dict[int, List[Dict[str, Any]]]:
        return {n: read_jsonl(self.p(f"edge-gw-{n}.probes.jsonl")) for n in self.edge_gateway_files()}

    def cloud_records(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.p("cloud-gw.jsonl"))

    def manager_records(self) -> Dict[int, List[Dict[str, Any]]]:
        out: Dict[int, List[Dict[str, Any]]] = {}
        for path in glob.glob(self.p("engine-mgr-*.jsonl")):
            m = re.fullmatch(r"engine-mgr-(\d+)\.jsonl", os.path.basename(path))
            if m:
                out[int(m.group(1))] = read_jsonl(path)
        return dict(sorted(out.items()))

    def monitor_gpu(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.p("monitor-gpu.jsonl"))

    def monitor_vllm(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.p("monitor-vllm.jsonl"))
