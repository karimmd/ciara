from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from .outcomes import mean, quantile


def summarize(requests: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cls in sorted({r["class"] for r in requests}):
        sel = [r for r in requests if r["class"] == cls]
        lat = [r["latency_s"] for r in sel if r["latency_s"] is not None]
        out[cls] = {
            "n": len(sel),
            "answered": mean(1.0 if r["ok"] else 0.0 for r in sel),
            "deadline_miss": mean(r["deadline_violation"] for r in sel),
            "timeout_share": mean(1.0 if r["error"] == "timeout" else 0.0 for r in sel),
            "lat_p50_s": quantile(lat, 0.5), "lat_p95_s": quantile(lat, 0.95),
            "bytes_down_mean": mean(r["bytes_down"] for r in sel if r["bytes_down"] is not None),
        }
    return out


def load(analysis_dir: str) -> List[Dict[str, Any]]:
    with open(os.path.join(analysis_dir, "requests.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def markdown(named: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    def f(v: Optional[float], nd: int = 3) -> str:
        return "empty" if v is None else f"{v:.{nd}f}"
    lines = ["| run | class | n | answered | deadline miss | client timeouts | latency p50 (s) | latency p95 (s) | bytes received (mean) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, summ in named.items():
        for cls, s in summ.items():
            lines.append(f"| {name} | {cls} | {s['n']} | {f(s['answered'])} | {f(s['deadline_miss'])} | {f(s['timeout_share'])} | "
                         f"{f(s['lat_p50_s'], 1)} | {f(s['lat_p95_s'], 1)} | {f(s['bytes_down_mean'], 0)} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    dirs = list(argv if argv is not None else sys.argv[1:])
    if not dirs:
        print('usage: python -m ciara_analysis.class_summary RUN_A/analysis RUN_B/analysis ...')
        return 2
    named = {os.path.basename(os.path.dirname(os.path.abspath(d))): summarize(load(d)) for d in dirs}
    print(markdown(named))
    return 0


if __name__ == "__main__":
    sys.exit(main())
