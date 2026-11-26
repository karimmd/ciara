from __future__ import annotations

import json
import math
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

PORT_BASE = 10240
PORT_BLOCK = 64
MINOR0 = 0x10
PKT_BYTES = 1500


def port_block(idx: int, base: int = PORT_BASE, block: int = PORT_BLOCK) -> Tuple[int, int]:
    first = base + block * idx
    if first % block != 0:
        raise ValueError("port base must be a multiple of the block size")
    return first, 0xFFFF & ~(block - 1)


def stable_seed(*labels: object) -> int:
    import hashlib

    h = hashlib.sha256("|".join(str(x) for x in labels).encode()).digest()
    return (int.from_bytes(h[:8], "big") | 1)


def netem_params(rate_kbps: float, delay_ms: float, jitter_ms: float, loss_pct: float, buffer_ms: float,
                 seed: Optional[int] = None) -> str:
    rate_bps = int(round(rate_kbps * 1000.0))
    limit = max(30, int(math.ceil(1.2 * rate_bps / 8.0 / PKT_BYTES * (delay_ms + buffer_ms) / 1000.0)))
    parts = [f"delay {delay_ms:g}ms"]
    if jitter_ms > 0:
        parts[0] += f" {jitter_ms:g}ms distribution normal"
    if loss_pct > 0:
        parts.append(f"loss random {loss_pct:g}%")
    parts.append(f"rate {rate_bps}bit")
    parts.append(f"limit {limit}")
    if seed is not None:
        parts.append(f"seed {int(seed)}")
    return " ".join(parts)


def _direction(link: Dict[str, Any], d: str) -> str:
    return netem_params(link[f"rate_{d}_kbps"], link[f"delay_{d}_ms"], link[f"jitter_{d}_ms"], link[f"loss_{d}_pct"],
                        link["buffer_ms"], link.get(f"seed_{d}"))


def build_commands(links: Sequence[Dict[str, Any]], dev: str = "eth0", ifb: str = "ifb0") -> List[str]:
    cmds: List[str] = []
    cmds.append(f"qdisc add dev {dev} root handle 1: htb default 0")
    cmds.append(f"qdisc add dev {dev} handle ffff: ingress")
    cmds.append(f"filter add dev {dev} parent ffff: protocol ip prio 1 u32 match u32 0 0 action mirred egress redirect dev {ifb}")
    cmds.append(f"qdisc add dev {ifb} root handle 2: htb default 0")
    for idx, link in enumerate(links):
        if link is None:
            continue
        m = MINOR0 + idx
        first, mask = port_block(idx)
        for d, dv, parent, hmaj, field in (("up", dev, "1", "1", "sport"), ("down", ifb, "2", "2", "dport")):
            cmds.append(f"class add dev {dv} parent {parent}: classid {parent}:{m:x} htb rate 10gbit ceil 10gbit quantum 60000")
            cmds.append(f"qdisc add dev {dv} parent {parent}:{m:x} handle {m:x}: netem {_direction(link, d)}")
            cmds.append(f"filter add dev {dv} parent {parent}: protocol ip prio 1 u32 match ip {field} {first} 0x{mask:04x} flowid {parent}:{m:x}")
    return cmds


def _run(args: Sequence[str], inp: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    cp = subprocess.run(list(args), input=inp, capture_output=True, text=True)
    if check and cp.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed ({cp.returncode}): {cp.stderr.strip()[:2000]}")
    return cp


def apply(links: Sequence[Dict[str, Any]], dev: str = "eth0", ifb: str = "ifb0") -> List[str]:
    clear(dev, ifb)
    _run(["ip", "link", "add", ifb, "type", "ifb"])
    _run(["ip", "link", "set", "dev", ifb, "up"])
    cmds = build_commands(links, dev, ifb)
    _run(["tc", "-batch", "-"], "\n".join(cmds) + "\n")
    return cmds


def clear(dev: str = "eth0", ifb: str = "ifb0") -> None:
    _run(["tc", "qdisc", "del", "dev", dev, "root"], check=False)
    _run(["tc", "qdisc", "del", "dev", dev, "ingress"], check=False)
    _run(["ip", "link", "del", ifb], check=False)


def read_stats(n_devices: int, dev: str = "eth0", ifb: str = "ifb0") -> Dict[int, Dict[str, Dict[str, int]]]:
    out: Dict[int, Dict[str, Dict[str, int]]] = {i: {} for i in range(n_devices)}
    for d, dv in (("up", dev), ("down", ifb)):
        cp = _run(["tc", "-s", "-j", "qdisc", "show", "dev", dv])
        for q in json.loads(cp.stdout or "[]"):
            if q.get("kind") != "netem":
                continue
            minor = int(q["handle"].rstrip(":"), 16) - MINOR0
            if minor in out:
                out[minor][d] = {"bytes": q.get("bytes", 0), "packets": q.get("packets", 0), "drops": q.get("drops", 0),
                                 "overlimits": q.get("overlimits", 0), "backlog": q.get("backlog", 0), "qlen": q.get("qlen", 0)}
    return out


def describe(links: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"idx": i, "up": _direction(l, "up"), "down": _direction(l, "down")} for i, l in enumerate(links) if l]
