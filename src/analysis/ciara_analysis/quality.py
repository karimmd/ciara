from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

DEFAULT_PILOT = "workload"
DEFAULT_SUITES = {1: ("iot", "task_acc"), 2: ("sensor", "answer_acc")}


def pilot_root() -> str:
    env = os.environ.get("CIARA_PILOT_ROOT")
    if env:
        return env
    local = Path(__file__).resolve().parents[3] / "workload"
    if (local / "eval").is_dir():
        return str(local)
    return DEFAULT_PILOT


class SuiteScorer:
    def __init__(self, suite_name: str, metric: str, max_tokens: Optional[int] = None):
        root = pilot_root()
        if root not in sys.path:
            sys.path.insert(0, root)
        from eval.suites.registry import get_suite
        opts: Dict[str, Any] = {}
        if max_tokens:
            opts["max_tokens"] = max_tokens
        self.suite = get_suite(suite_name, opts)
        self.metric = metric
        self.items = {it["id"]: it for it in self.suite.items()}

    def score(self, item_id: str, reply: Optional[str]) -> Tuple[Optional[float], str]:
        item = self.items.get(item_id)
        if item is None:
            return None, "unknown_item"
        out = self.suite.score(item, {"content": reply or ""})
        v = out["scores"].get(self.metric)
        return (None, "not_applicable") if v is None else (float(v), "scored")


def default_scorers() -> Dict[int, Callable[[str, Optional[str]], Tuple[Optional[float], str]]]:
    return {f: SuiteScorer(s, m).score for f, (s, m) in DEFAULT_SUITES.items()}


def score_request(scorers: Dict[int, Callable[[str, Optional[str]], Tuple[Optional[float], str]]], family: int,
                  item_id: str, reply: Optional[str], finish_reason: Optional[str], ok: bool,
                  truncated_is_wrong: bool = True) -> Tuple[Optional[float], str]:
    if not ok:
        return 0.0, "failed"
    if truncated_is_wrong and finish_reason == "length":
        return 0.0, "truncated"
    fn = scorers.get(int(family))
    if fn is None:
        return None, "no_scorer"
    return fn(item_id, reply)
