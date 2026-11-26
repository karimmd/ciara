from __future__ import annotations

import os
from typing import Dict, List, Optional

CLASSES = ("A", "B", "C")
MAX_NODES = 6


def parse_cpu_list(spec: str) -> List[int]:
    out: List[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def client_cpu(node: int, cls: str) -> Optional[int]:
    if not 1 <= node <= MAX_NODES:
        raise ValueError(f"node {node} outside 1..{MAX_NODES}")
    if cls not in CLASSES:
        raise ValueError(f"unknown class {cls!r}")
    cpus = parse_cpu_list(os.environ.get("CIARA_CLIENT_CPUS", ""))
    if not cpus:
        return None
    return cpus[(3 * (node - 1) + CLASSES.index(cls)) % len(cpus)]


def container_name(node: int, cls: str, prefix: str = "client") -> str:
    return f"{prefix}-{node}-{cls}"


def groups(nodes: int, classes: List[str] = ("A", "B", "C")) -> List[Dict[str, object]]:
    return [{"node": n, "class": c, "name": container_name(n, c), "cpu": client_cpu(n, c)} for n in range(1, nodes + 1) for c in classes]
