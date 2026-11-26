from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from ciara.transient import TauTable

OUT = ("Y_Q", "Y_L_f1", "Y_L_f2", "Y_D", "Y_E_A", "Y_E_B", "Y_E_C", "Y_C")
BASE = {"Y_Q": 0.62, "Y_L_f1": 0.003, "Y_L_f2": 0.003, "Y_D": 0.003, "Y_E_A": 0.25, "Y_E_B": 0.25, "Y_E_C": 0.25, "Y_C": 0.0}
NOISE = {"Y_Q": 0.02, "Y_L_f1": 0.002, "Y_L_f2": 0.002, "Y_D": 0.002, "Y_E_A": 0.02, "Y_E_B": 0.02, "Y_E_C": 0.02, "Y_C": 0.003}
COUNTS = {"n": 390, "n_f1": 200, "n_f2": 190, "n_A": 160, "n_B": 130, "n_C": 100}
ALL_PROFILES = ["q35-4b-fp16", "q35-4b-ih-w8a8", "q35-4b-ih-w4a16", "q35-2b-fp16", "q35-0.8b-fp16"]
PILOT_PROFILES = [("q35-4b-fp16", (8,)), ("q35-4b-ih-w8a8", (8, 16)), ("q35-4b-ih-w4a16", (8, 16)), ("q35-2b-fp16", (8, 16)), ("q35-0.8b-fp16", (8, 16))]
TWO_PROFILES = [("q35-4b-ih-w8a8", (8, 16)), ("q35-4b-ih-w4a16", (8, 16))]


def static_arm(profile: str, cap: int, kappa: Any) -> Dict[str, Any]:
    return {"id": f"{profile}|b{cap}|k{kappa}", "kind": "static", "profile": profile, "batch_cap": cap, "kappa": kappa, "profiles": [profile]}


def regime_arms(base_profile: str, base_cap: int) -> List[Dict[str, Any]]:
    return [
        {"id": "I1-E", "kind": "incumbent", "profile": None, "batch_cap": None, "kappa": None, "profiles": list(ALL_PROFILES)},
        {"id": "I2", "kind": "incumbent", "profile": base_profile, "batch_cap": base_cap, "kappa": None, "profiles": [base_profile]},
        {"id": "I3", "kind": "incumbent", "profile": base_profile, "batch_cap": base_cap, "kappa": None, "profiles": [base_profile]},
    ]


def arms_default() -> List[Dict[str, Any]]:
    stat = [static_arm(p, cap, k) for p, caps in PILOT_PROFILES for cap in caps for k in (0, 1)]
    return stat + regime_arms("q35-4b-ih-w4a16", 16)


def arms_rescreen() -> List[Dict[str, Any]]:
    stat = [static_arm(p, cap, k) for p, caps in TWO_PROFILES for cap in caps for k in (1, "A", "H")]
    return stat + regime_arms("q35-4b-ih-w4a16", 16)


def arms_restricted() -> List[Dict[str, Any]]:
    stat = [static_arm(p, 8, k) for p, _ in TWO_PROFILES for k in (1, "A", "H")]
    return stat + regime_arms("q35-4b-ih-w4a16", 8)


def admission_arm(profile: str, cap: int, kappa: Any, admission: str) -> Dict[str, Any]:
    d = static_arm(profile, cap, kappa)
    d["id"] = f"{d['id']}|{admission}"
    d["admission"] = admission
    return d


def arms_balanced() -> List[Dict[str, Any]]:
    return [admission_arm("q35-4b-ih-w8a8", 8, "H", "fifo"), admission_arm("q35-4b-ih-w8a8", 8, "H", "f1prio"), admission_arm("q35-4b-ih-w8a8", 8, 1, "fifo"),
            regime_arms("q35-4b-ih-w4a16", 8)[0]]


W4, W8 = "q35-4b-ih-w4a16", "q35-4b-ih-w8a8"
A0 = "I1-E"
KH_F, KH_P, K1_F = f"{W8}|b8|kH|fifo", f"{W8}|b8|kH|f1prio", f"{W8}|b8|k1|fifo"
MEM3 = [A0, KH_P, K1_F]
PROF3 = {A0: [W4, W8], KH_P: [W8], K1_F: [W8]}
TRUTH_GO = {A0: {"Y_Q": 0.68, "Y_C": 0.49, "Y_L_f1": 0.02, "Y_L_f2": 0.02, "Y_D": 0.02},
            KH_F: {"Y_Q": 0.69, "Y_C": 0.21, "Y_L_f1": 0.14, "Y_L_f2": 0.12, "Y_D": 0.05},
            KH_P: {"Y_Q": 0.69, "Y_C": 0.22, "Y_L_f1": 0.03, "Y_L_f2": 0.05, "Y_D": 0.04},
            K1_F: {"Y_Q": 0.70, "Y_C": 0.46, "Y_L_f1": 0.03, "Y_L_f2": 0.03, "Y_D": 0.03}}


def tau_table() -> TauTable:
    p = {}
    sleep = ["q35-2b-fp16", "q35-4b-ih-w4a16", "q35-4b-ih-w8a8"]
    for a in sleep:
        for b in sleep:
            if a != b:
                p[f"{a} -> {b}"] = {"tau_max": 1 if (a, b) != ("q35-2b-fp16", "q35-4b-ih-w8a8") else 4, "mode": "sleep"}
    for a in sleep:
        p[f"{a} -> q35-4b-fp16"] = {"tau_max": 10, "mode": "restart"}
        p[f"q35-4b-fp16 -> {a}"] = {"tau_max": 16, "mode": "rebuild-group"}
    return TauTable(p)


def tau_table_flat() -> TauTable:
    p = {}
    sleep = ["q35-2b-fp16", W4, W8]
    for a in sleep:
        for b in sleep:
            if a != b:
                p[f"{a} -> {b}"] = {"tau_max": 1, "mode": "sleep"}
    for a in sleep:
        p[f"{a} -> q35-4b-fp16"] = {"tau_max": 10, "mode": "restart"}
        p[f"q35-4b-fp16 -> {a}"] = {"tau_max": 16, "mode": "rebuild-group"}
    return TauTable(p)


def table_default() -> Dict[str, Any]:
    edge = {}
    for p, kv in (("q35-4b-fp16", 40960), ("q35-4b-ih-w8a8", 90000), ("q35-4b-ih-w4a16", 114000), ("q35-2b-fp16", 410000), ("q35-0.8b-fp16", 560000)):
        edge[p] = {"budget_gib": 10.45, "kv_gib": 3.0 + kv / 1e5, "kv_tokens": kv}
    return {"edge": edge}


def make_rows(truth: Dict[str, Dict[str, float]], seed: int = 1, n_nodes: int = 5, n_blocks: int = 42, arms: Optional[List[Dict[str, Any]]] = None,
              coupling: Optional[Dict[str, float]] = None, cl: Optional[Dict[str, float]] = None, force: Optional[Dict[int, List[str]]] = None,
              block_noise: float = 1.0) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    arms = arms or arms_default()
    ids = [a["id"] for a in arms]
    cl = cl or {}
    rows: List[Dict[str, Any]] = []
    for b in range(n_blocks):
        draw = (force or {}).get(b) or [rng.choice(ids) for _ in range(n_nodes)]
        for n in range(n_nodes):
            arm = draw[n]
            sib = [cl.get(draw[m], 0.0) for m in range(n_nodes) if m != n]
            e = sum(sib) / len(sib)
            row: Dict[str, Any] = {"run_id": "synth", "node": f"E{n + 1}", "node_index": n + 1, "block_id": b, "window": "measure", "arm_id": arm,
                                   "block_status": "complete", "profiles_in_window": ("q35-4b-ih-w4a16+q35-2b-fp16" if arm == "I1-E" else next((a["profiles"][0] for a in arms if a["id"] == arm and a["profiles"]), None))}
            for k in OUT:
                mu = truth.get(arm, {}).get(k, BASE[k])
                if coupling and cl.get(arm, 0.0) > 0.0:
                    mu = mu + coupling.get(k, 0.0) * e
                v = mu + rng.gauss(0, NOISE[k] * block_noise)
                row[k] = min(1.0, max(0.0, v))
            row.update(COUNTS)
            rows.append(row)
    return rows


def balanced_campaign_rows(truth: Dict[str, Dict[str, float]], n_blocks: int, seed: int = 1) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    comp = [A0, A0, KH_P, K1_F]
    force = {}
    for b in range(n_blocks):
        perm = comp[:]
        rng.shuffle(perm)
        force[b] = perm
    return make_rows(truth, seed=seed, n_blocks=n_blocks, n_nodes=4, arms=arms_balanced(), force=force)
