from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

FP16_4B = "q35-4b-fp16"
PROXY = {"q35-0.8b-fp16": "q35-2b-fp16"}


class TauTable:
    def __init__(self, pairs: Dict[str, Dict]):
        self.pairs = pairs
        self.tau: Dict[Tuple[str, str], float] = {}
        self.mode: Dict[Tuple[str, str], str] = {}
        for k, v in pairs.items():
            a, b = [x.strip() for x in k.split("->")]
            self.tau[(a, b)] = float(v["tau_max"])
            self.mode[(a, b)] = v.get("mode", "")
        sleep = [t for p, t in self.tau.items() if self.mode[p] == "sleep"]
        heavy = [t for p, t in self.tau.items() if self.mode[p] != "sleep"]
        self.max_sleep = max(sleep) if sleep else 4.0
        self.max_heavy = max(heavy) if heavy else 16.0

    @staticmethod
    def load(path: str) -> "TauTable":
        with open(path, "r", encoding="utf-8") as fh:
            return TauTable(json.load(fh)["pairs"])

    def pair(self, src: str, dst: str) -> float:
        if src == dst:
            return 1.0
        a, b = PROXY.get(src, src), PROXY.get(dst, dst)
        if a == b:
            return self.max_sleep
        if (a, b) in self.tau:
            return self.tau[(a, b)]
        return self.max_heavy if FP16_4B in (a, b) else self.max_sleep

    def between(self, profs_a: Iterable[str], profs_b: Iterable[str]) -> float:
        out = 0.0
        for p in profs_a:
            for q in profs_b:
                out = max(out, self.pair(p, q), self.pair(q, p))
        return out

    def set_tau(self, profiles: Dict[str, Sequence[str]], members: Sequence[str]) -> float:
        own = [p for m in members for p in profiles[m]]
        t = self.between(own, own) if len(set(own)) > 1 else 0.0
        return t if t > 0 else 1.0

    def arm_tau(self, arm: str, profiles: Dict[str, Sequence[str]], members: Sequence[str]) -> float:
        own = list(profiles[arm])
        t = 0.0
        for m in members:
            if m == arm:
                continue
            t = max(t, self.between(own, profiles[m]))
        if len(set(own)) > 1:
            t = max(t, self.between(own, own) if len(set(own)) > 1 else 0.0)
        return t if t > 0 else 1.0
