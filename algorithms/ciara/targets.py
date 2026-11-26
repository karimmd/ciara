from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

from .constants import finite, mean

MARGIN = 1.25
MIN_PER_FAMILY = 1000
MAX_FAIL = 0.05
STATIONARITY = (0.5, 2.0)


def p95_with_failures(latencies: Sequence[Optional[float]], q: float = 0.95) -> Optional[float]:
    n = len(latencies)
    if n == 0:
        return None
    xs = sorted(float("inf") if x is None else float(x) for x in latencies)
    k = (n - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    v = xs[lo] + (xs[hi] - xs[lo]) * (k - lo) if xs[lo] != float("inf") and xs[hi] != float("inf") else float("inf")
    return None if v == float("inf") else v


def compute(requests: Sequence[Dict[str, Any]], warm_epochs: int = 20) -> Dict[str, Any]:
    use = [r for r in requests if r.get("epoch") is not None and r["epoch"] >= warm_epochs]
    out: Dict[str, Any] = {"warm_epochs": warm_epochs, "n_used": len(use), "families": {}, "classes": {}, "valid": True, "problems": []}
    for f in (1, 2):
        sel = [r for r in use if int(r["family"]) == f]
        lat = [r["latency_s"] if r.get("ok") and finite(r.get("latency_s")) else None for r in sel]
        n_fail = sum(1 for x in lat if x is None)
        p95 = p95_with_failures(lat)
        sel_sorted = sorted(sel, key=lambda r: r["epoch"])
        half = len(sel_sorted) // 2
        h1 = p95_with_failures([x["latency_s"] if x.get("ok") and finite(x.get("latency_s")) else None for x in sel_sorted[:half]])
        h2 = p95_with_failures([x["latency_s"] if x.get("ok") and finite(x.get("latency_s")) else None for x in sel_sorted[half:]])
        ratio = (h2 / h1) if (h1 and h2) else None
        out["families"][f"f{f}"] = {"n": len(sel), "n_failed": n_fail, "fail_rate": (n_fail / len(sel)) if sel else None, "p95_s": p95,
                                    "p95_first_half_s": h1, "p95_second_half_s": h2, "half_ratio": ratio,
                                    "mean_s": mean([x for x in lat if x is not None]), "target_s": (MARGIN * p95) if p95 is not None else None}
        if len(sel) < MIN_PER_FAMILY:
            out["valid"] = False
            out["problems"].append(f"family {f}: {len(sel)} requests, fewer than {MIN_PER_FAMILY}")
        if p95 is None:
            out["valid"] = False
            out["problems"].append(f"family {f}: the 95th percentile falls on a failed request ({n_fail} of {len(sel)} failed)")
        elif sel and n_fail / len(sel) > MAX_FAIL:
            out["valid"] = False
            out["problems"].append(f"family {f}: failure rate {n_fail / len(sel):.3f} above {MAX_FAIL}")
        if ratio is not None and not (STATIONARITY[0] <= ratio <= STATIONARITY[1]):
            out["valid"] = False
            out["problems"].append(f"family {f}: p95 of the second half over the first half is {ratio:.2f}, outside {STATIONARITY[0]:.2f} to {STATIONARITY[1]:.2f} (no stationary regime)")
    for c in ("A", "B", "C"):
        sel = [r for r in use if r.get("class") == c and finite(r.get("energy_mj"))]
        m = mean([r["energy_mj"] / 1000.0 for r in sel])
        out["classes"][c] = {"n": len(sel), "mean_energy_J": m, "budget_J": (MARGIN * m) if m is not None else None}
        if m is None:
            out["valid"] = False
            out["problems"].append(f"class {c}: no energy rows")
    out["L1_s"] = out["families"]["f1"]["target_s"]
    out["L2_s"] = out["families"]["f2"]["target_s"]
    return out
