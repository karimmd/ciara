from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import iot_gen
from iot_gen import norm_text


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

REV = "29ac1a80b7185e7b4c2c9e43b7e4fe71531ec434"
SEED_REAL, SEED_SYN_TEST, SEED_SYN_HELD = 20261014, 20261015, 20261016

REAL_QUOTAS = {"test": {"real_onoff": 60, "real_todo": 15, "real_climate": 15, "real_humidifier": 8, "real_vacuum": 8,
                        "real_timer": 8, "real_state_nocall": 36},
               "heldout": {"real_onoff": 45, "real_todo": 12, "real_climate": 12, "real_humidifier": 6, "real_vacuum": 6,
                           "real_timer": 6, "real_state_nocall": 27}}
SYN_QUOTAS = {"test": {"syn_onoff": 30, "syn_lockcover": 25, "syn_brightness": 25, "syn_color": 20, "syn_color_brightness": 10,
                       "syn_position": 15, "syn_volume": 12, "syn_media": 12, "syn_vacuum": 6, "syn_todo": 8, "syn_timer": 8,
                       "syn_climate": 10, "syn_multi": 35, "nocall_ood": 20, "nocall_unsupported": 12, "nocall_state": 12},
              "heldout": {"syn_onoff": 22, "syn_lockcover": 19, "syn_brightness": 19, "syn_color": 15, "syn_color_brightness": 8,
                          "syn_position": 11, "syn_volume": 9, "syn_media": 9, "syn_vacuum": 5, "syn_todo": 6, "syn_timer": 6,
                          "syn_climate": 8, "syn_multi": 26, "nocall_ood": 15, "nocall_unsupported": 9, "nocall_state": 9}}

LINE = re.compile(r"^([a-z_]+)\.([a-z0-9_]+) '(.*)' = (.*)$", re.M)
POLITE = re.compile(r"^(?:\w+ you |please |kindly |hey,? |go ahead and |i (?:want|need) you to |i'd like you to )+")
VERBS = {
    "on": ["turn on", "switch on", "activate", "power on", "power up", "enable", "open", "open up", "unlock", "start"],
    "off": ["turn off", "switch off", "deactivate", "power off", "power down", "disable", "close", "lock", "shut", "shut off"],
    "toggle": ["toggle", "flip"],
}
DOMAIN_VERBS = {
    "open": {"cover"}, "open up": {"cover"}, "unlock": {"lock"}, "close": {"cover"}, "lock": {"lock"}, "shut": {"cover"},
    "start": {"fan", "switch", "light", "media_player"},
}


def domain(eid: str) -> str:
    return eid.split(".")[0]


def parse_devices(prompt: str) -> List[Dict[str, str]]:
    return [{"entity_id": f"{m.group(1)}.{m.group(2)}", "name": m.group(3), "state": m.group(4)} for m in LINE.finditer(prompt)]


def simple_intent(template: str, dom: str) -> Optional[str]:
    t = POLITE.sub("", norm_text(template)).strip()
    t = re.sub(r"\s+(please|now)$", "", t)
    for intent, verbs in VERBS.items():
        for v in verbs:
            if v in DOMAIN_VERBS and dom not in DOMAIN_VERBS[v]:
                continue
            if re.fullmatch(rf"{re.escape(v)} (?:the )?\{{n\}}", t):
                return intent
    m = re.fullmatch(r"(?:turn|switch) (?:the )?\{n\} (on|off)", t)
    if m and dom in ("light", "switch", "fan", "media_player"):
        return m.group(1)
    return None


def unique_target(name: str, devices: List[Dict[str, str]]) -> bool:
    n = norm_text(name)
    return sum(1 for d in devices if n in norm_text(d["name"]) or norm_text(d["name"]) in n) == 1


def real_candidates(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, r in enumerate(rows):
        ms = r["messages"]
        utter = ms[1]["content"][0]["text"].strip()
        prompt = ms[0]["content"][0]["text"]
        devices = parse_devices(prompt)
        if not devices:
            continue
        by_name = {norm_text(d["name"]): d for d in devices}
        by_id = {d["entity_id"]: d for d in devices}
        calls = [tc["function"] for tc in (ms[2].get("tool_calls") or [])]
        base = {"source": "ha_v2_real", "devices": devices, "request": utter, "meta_row": idx}
        if not calls:
            if re.match(r"^(is|are|what|which|how|does|do|has|was|were)\b", utter.lower()) and not re.search(
                    r"\b(turn|open|close|set|start|stop|lock|unlock|toggle|switch)\b", utter.lower()):
                pools["real_state_nocall"].append({**base, "gold_calls": [], "category": "real_state_nocall"})
            continue
        if len(calls) != 1 or len(ms) < 5 or not all("Success" in x["text"] for x in ms[3]["content"]):
            continue
        c = calls[0]
        a = json.loads(c["arguments"])
        tgt = a.get("name", "")
        d = by_name.get(norm_text(tgt)) or by_id.get(tgt)
        if not d or not unique_target(d["name"], devices):
            continue
        u_norm = norm_text(utter)
        n_norm = norm_text(d["name"])
        dom = domain(d["entity_id"])
        tool = c["name"]
        gold = {"name": tool, "arguments": {**a, "name": d["name"]}}
        if tool in ("HassTurnOn", "HassTurnOff", "HassToggle") and set(a) == {"name"}:
            if u_norm.count(n_norm) != 1 or dom not in ("light", "switch", "fan", "cover", "lock", "media_player"):
                continue
            intent = simple_intent(u_norm.replace(n_norm, "{n}"), dom)
            want = {"on": "HassTurnOn", "off": "HassTurnOff", "toggle": "HassToggle"}.get(intent or "")
            if want == tool:
                pools["real_onoff"].append({**base, "gold_calls": [gold], "category": "real_onoff"})
        elif tool == "HassListAddItem" and set(a) == {"name", "item"} and dom == "todo":
            words = re.findall(r"[a-z0-9']+", norm_text(a["item"]))
            if words and all(w in u_norm for w in words) and n_norm in u_norm:
                pools["real_todo"].append({**base, "gold_calls": [gold], "category": "real_todo"})
        elif tool in ("HassClimateSetTemperature", "HassHumidifierSetpoint") and dom == "climate":
            key = "temperature" if tool == "HassClimateSetTemperature" else "humidity"
            if set(a) == {"name", key} and sum(1 for x in devices if domain(x["entity_id"]) == "climate") == 1 \
                    and re.search(rf"(?<![\d.]){re.escape(str(a[key]))}(?![\d])", u_norm):
                cat = "real_climate" if key == "temperature" else "real_humidifier"
                pools[cat].append({**base, "gold_calls": [gold], "category": cat})
        elif tool in ("HassVacuumStart", "HassVacuumReturnToBase") and set(a) == {"name"} and dom == "vacuum" and n_norm in u_norm:
            ret = bool(re.search(r"return|dock|charg|base|home|back", u_norm))
            sta = bool(re.search(r"\b(start|begin|run|clean|vacuum)\b", u_norm))
            if (tool == "HassVacuumReturnToBase" and ret and not sta) or (tool == "HassVacuumStart" and sta and not ret):
                pools["real_vacuum"].append({**base, "gold_calls": [gold], "category": "real_vacuum"})
        elif tool == "HassCancelTimer" and set(a) == {"name"} and dom == "timer" and n_norm in u_norm \
                and re.search(r"\b(cancel|stop|end|clear|abort)\b", u_norm):
            pools["real_timer"].append({**base, "gold_calls": [gold], "category": "real_timer"})
    return pools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="home_assistant_test_english.jsonl of Home-Assistant-Requests V2 at the revision REV")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = args.raw
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    tools = [{"type": "function", "function": t["function"]} for t in rows[0]["tools"]]
    assert all(json.dumps(r["tools"], sort_keys=True) == json.dumps(rows[0]["tools"], sort_keys=True) for r in rows)
    json.dump(tools, open(os.path.join(out_dir, "tools.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    catalog: Dict[str, Dict[str, str]] = {}
    seen_names = set()
    for r in rows:
        for d in parse_devices(r["messages"][0]["content"][0]["text"]):
            k = norm_text(d["name"])
            if d["entity_id"] in catalog or k in seen_names or len(d["name"]) > 60:
                continue
            catalog[d["entity_id"]] = d
            seen_names.add(k)
    cat_list = sorted(catalog.values(), key=lambda d: d["entity_id"])
    json.dump({"source": "Home-Assistant-Requests V2 test_english prompts", "devices": cat_list},
              open(os.path.join(out_dir, "catalog.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    pools = real_candidates(rows)
    rng = random.Random(SEED_REAL)
    taken: set = set()
    split_items: Dict[str, List[Dict[str, Any]]] = {"test": [], "heldout": []}
    avail = {k: len(v) for k, v in pools.items()}
    for cat in sorted(pools):
        pool = pools[cat]
        rng.shuffle(pool)
        for split in ("test", "heldout"):
            want = REAL_QUOTAS[split].get(cat, 0)
            got = 0
            while got < want and pool:
                it = pool.pop()
                key = norm_text(it["request"])
                if key in taken:
                    continue
                taken.add(key)
                split_items[split].append(it)
                got += 1
            if got < want:
                print(f"WARNING: real pool {cat} short for {split}: {got}/{want}")

    for split, seed in (("test", SEED_SYN_TEST), ("heldout", SEED_SYN_HELD)):
        split_items[split].extend(iot_gen.generate(cat_list, seed, SYN_QUOTAS[split], taken))

    final: Dict[str, List[Dict[str, Any]]] = {}
    for split, items in split_items.items():
        r2 = random.Random(hash_seed(split))
        r2.shuffle(items)
        rows_out = []
        for i, it in enumerate(items):
            rows_out.append({
                "id": f"iot-{split}-{i:04d}",
                "split": split,
                "source": it["source"],
                "category": it["category"],
                "request": it["request"],
                "devices": it["devices"],
                "gold_calls": it["gold_calls"],
                "expected_no_call": not it["gold_calls"],
                "source_row": it.get("meta_row"),
            })
        final[split] = rows_out
        write_jsonl(os.path.join(out_dir, f"{split}.jsonl"), rows_out)

    split_info = {
        "description": "Fixed splits. test is the evaluation suite. heldout is reserved for later confirmation runs; it is not used for prompt tuning. "
                       "Requests are unique across splits (case-folded). Training data must come from iot_gen with a different seed "
                       "and must avoid every request in both files.",
        "seeds": {"real": SEED_REAL, "synthetic_test": SEED_SYN_TEST, "synthetic_heldout": SEED_SYN_HELD},
        "counts": {s: len(v) for s, v in final.items()},
        "by_category": {s: dict(sorted(Counter(x["category"] for x in v).items())) for s, v in final.items()},
        "by_source": {s: dict(Counter(x["source"] for x in v)) for s, v in final.items()},
        "no_call_share": {s: round(sum(x["expected_no_call"] for x in v) / len(v), 3) for s, v in final.items()},
        "real_pool_sizes_after_verification": avail,
        "ids": {s: [x["id"] for x in v] for s, v in final.items()},
    }
    json.dump(split_info, open(os.path.join(out_dir, "SPLIT.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: split_info[k] for k in ("counts", "by_source", "no_call_share", "real_pool_sizes_after_verification")}, indent=1))


def hash_seed(s: str) -> int:
    return sum(ord(c) * (31 ** i) for i, c in enumerate(s)) % (2 ** 31)


if __name__ == "__main__":
    main()
