from __future__ import annotations

import bisect
import json
import math
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .constants import OUTCOMES


def rho_of(t_bar: float) -> float:
    return math.exp(-1.0 / t_bar)


def beta_dm(t_bar: float, b: float, l: float, tau: float = 0.0) -> float:
    b, l = b - tau, l - tau
    if l - b <= 0:
        raise ValueError("measurement window is empty")
    return math.exp(-b / t_bar) * (1 - math.exp(-(l - b) / t_bar)) / ((1 - math.exp(-1 / t_bar)) * (l - b))


EPOCH_NS = 30 * 10 ** 9
BLOCK_EPOCHS = 50
BURN_IN, WINDOW_END = 25, 50
PLATEAU = (30, 50)
MIN_PLATEAU = 10
MIN_WINDOW = 3
MIN_BLOCKS = 5
MIN_CLUSTERS = 5
Z95 = 1.96
CLASSES = ("A", "B", "C")


def iter_jsonl(path: str, needle: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    with open(path) as fh:
        for line in fh:
            if needle is not None and needle not in line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


class Grid:
    def __init__(self, epoch_rows: Iterable[Dict[str, Any]]):
        per: Dict[str, List[Tuple[int, int, int]]] = {}
        for e in epoch_rows:
            if e.get("rec", "epoch") != "epoch" or e.get("t_start_mono_ns") is None:
                continue
            per.setdefault(e["node"], []).append((int(e["t_start_mono_ns"]), int(e["epoch"]), int(e["block_id"]) * BLOCK_EPOCHS + int(e["epoch_in_block"])))
        self.start: Dict[str, List[int]] = {}
        self.epoch: Dict[str, List[int]] = {}
        self.slot: Dict[str, List[int]] = {}
        for n, v in per.items():
            v.sort()
            self.start[n] = [x[0] for x in v]
            self.epoch[n] = [x[1] for x in v]
            self.slot[n] = [x[2] for x in v]

    def locate(self, node: str, t_ns: float) -> Optional[Tuple[int, int]]:
        s = self.start.get(node)
        if not s:
            return None
        i = bisect.bisect_right(s, t_ns) - 1
        if i < 0 or t_ns >= s[i] + EPOCH_NS * 1.02:
            return None
        return self.epoch[node][i], self.slot[node][i]


def _num(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


_J = {k: j for j, k in enumerate(OUTCOMES)}


def accumulate(requests: Iterable[Dict[str, Any]], grid: Grid) -> Dict[Tuple[str, int], List[float]]:
    acc: Dict[Tuple[str, int], List[float]] = {}

    def add(a: List[float], key: str, v: Optional[float]) -> None:
        if v is not None:
            j = _J[key]
            a[2 * j] += v
            a[2 * j + 1] += 1.0

    for r in requests:
        t = _num(r.get("t_arrival"))
        node = r.get("node")
        if t is None or node is None:
            continue
        loc = grid.locate(node, t)
        if loc is None:
            continue
        a = acc.setdefault((node, loc[0]), [0.0] * (2 * len(OUTCOMES)))
        add(a, "Y_Q", _num(r.get("q")))
        add(a, "Y_D", _num(r.get("deadline_violation")))
        add(a, "Y_C", 1.0 if r.get("cloud_used") else 0.0)
        fam = r.get("family")
        if fam in (1, 2):
            add(a, f"Y_L_f{fam}", _num(r.get("exceed")))
        cls = r.get("class")
        if cls in CLASSES:
            add(a, f"Y_E_{cls}", _num(r.get("e_norm")))
    return acc


def epoch_values(a: Optional[List[float]]) -> Dict[str, Optional[float]]:
    return {k: (a[2 * j] / a[2 * j + 1] if a is not None and a[2 * j + 1] > 0 else None) for k, j in _J.items()}


def switch_blocks(run_dir: str, grid: Grid, acc: Dict[Tuple[str, int], List[float]], k_rounds: Optional[int] = None) -> List[Dict[str, Any]]:
    arm: Dict[Tuple[str, int], str] = {}
    done: Dict[Tuple[str, int], bool] = {}
    for r in iter_jsonl(os.path.join(run_dir, "blocks.jsonl")):
        key = (r.get("node"), int(r.get("block_id", -1)))
        if r.get("rec") == "block":
            arm[key] = r.get("arm_id")
        elif r.get("rec") == "block_end":
            done[key] = r.get("status") == "complete" and int(r.get("epochs_logged") or 0) >= BLOCK_EPOCHS
    return build_blocks(arm, done, grid, acc, k_rounds)


def build_blocks(arm: Dict[Tuple[str, int], str], done: Dict[Tuple[str, int], bool], grid: Grid, acc: Dict[Tuple[str, int], List[float]],
                 k_rounds: Optional[int] = None) -> List[Dict[str, Any]]:
    slot_epoch: Dict[str, Dict[int, int]] = {n: dict(zip(grid.slot[n], grid.epoch[n])) for n in grid.start}
    out: List[Dict[str, Any]] = []
    for (node, b), a in sorted(arm.items(), key=lambda kv: (kv[0][1], str(kv[0][0]))):
        prev = arm.get((node, b - 1))
        if prev is None or prev == a or not done.get((node, b)) or (k_rounds is not None and b >= k_rounds):
            continue
        curves: Dict[str, List[Optional[float]]] = {k: [] for k in OUTCOMES}
        for k in range(BLOCK_EPOCHS):
            e = slot_epoch.get(node, {}).get(b * BLOCK_EPOCHS + k)
            v = epoch_values(acc.get((node, e))) if e is not None else epoch_values(None)
            for o in OUTCOMES:
                curves[o].append(v[o])
        out.append({"node": node, "round": b, "arm": a, "prev": prev, "curves": curves})
    return out


def window_carryover(blocks: Sequence[Dict[str, Any]], outcome: str) -> np.ndarray:
    c = np.full(len(blocks), np.nan)
    for i, b in enumerate(blocks):
        x = np.array([np.nan if v is None else float(v) for v in b["curves"][outcome]])
        pl = x[PLATEAU[0]:PLATEAU[1]]
        wd = x[BURN_IN:PLATEAU[0]]
        if np.isfinite(pl).sum() < MIN_PLATEAU or np.isfinite(wd).sum() < MIN_WINDOW:
            continue
        c[i] = (np.nanmean(wd) - np.nanmean(pl)) * (PLATEAU[0] - BURN_IN) / (WINDOW_END - BURN_IN)
    return c


def clustered_mean_se(values: np.ndarray, rounds: Sequence[int]) -> Tuple[Optional[float], Optional[float], int, int]:
    ok = np.isfinite(values)
    v, r = values[ok], np.asarray(rounds)[ok]
    n = len(v)
    g = len(set(r.tolist()))
    if n < MIN_BLOCKS or g < MIN_CLUSTERS:
        return None, None, n, g
    m = float(v.mean())
    tot: Dict[int, float] = {}
    for x, k in zip(v - m, r.tolist()):
        tot[k] = tot.get(k, 0.0) + float(x)
    se = math.sqrt(g / (g - 1.0) * sum(t * t for t in tot.values())) / n
    return m, se, n, g


def summarize(blocks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rounds = [b["round"] for b in blocks]
    res: Dict[str, Any] = {"definition": "|mean of (epochs 25 to 29 minus plateau of epochs 30 to 49) x 5/25| + 1.96 round-clustered standard errors, over the node-blocks that begin with a switch",
                           "z": Z95, "n_switch_blocks": len(blocks), "n_rounds": len(set(rounds)), "outcomes": {}}
    for o in OUTCOMES:
        m, se, n, g = clustered_mean_se(window_carryover(blocks, o), rounds)
        res["outcomes"][o] = {"mean": m, "se": se, "upper": (abs(m) + Z95 * se) if m is not None else None, "n_blocks": n, "n_rounds": g}
    return res


def measure(run_dir: str, requests_path: str, k_rounds: Optional[int] = None) -> Dict[str, Any]:
    grid = Grid(iter_jsonl(os.path.join(run_dir, "epochs.jsonl"), '"rec":"epoch"'))
    acc = accumulate(iter_jsonl(requests_path), grid)
    blocks = switch_blocks(run_dir, grid, acc, k_rounds)
    res = summarize(blocks)
    res["rounds"] = k_rounds
    return res


def bands_of(res: Optional[Dict[str, Any]]) -> Dict[str, float]:
    return {k: float(v["upper"]) for k, v in ((res or {}).get("outcomes") or {}).items() if v.get("upper") is not None}
