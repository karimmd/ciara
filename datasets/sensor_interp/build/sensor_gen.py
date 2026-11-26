from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple

GEN_VERSION = "sensor-gen-1"
STEP_MIN = 5
MAX_ATTEMPTS = 400


class Chan:
    def __init__(self, name: str, dec: int, desc: str, margin: int, delta: int, dev: int) -> None:
        self.name, self.dec, self.desc = name, dec, desc
        self.margin = margin
        self.delta = delta
        self.dev = dev


CH_ORDER = ["temp_c", "rh_pct", "co2_ppm", "power_w", "occupancy", "vib_mm_s"]
CH: Dict[str, Chan] = {c.name: c for c in [
    Chan("temp_c", 1, "air temperature in degrees Celsius", 1, 10, 10),
    Chan("rh_pct", 0, "relative humidity in percent", 1, 5, 5),
    Chan("co2_ppm", 0, "CO2 concentration in parts per million", 20, 100, 150),
    Chan("power_w", 0, "electric power drawn by the HVAC unit in watts", 60, 200, 300),
    Chan("occupancy", 0, "number of people counted in the room", 2, 2, 2),
    Chan("vib_mm_s", 2, "RMS vibration of the unit's supply fan motor in mm/s", 6, 50, 50),
]}
ANALOG_POOL = ["temp_c", "rh_pct", "co2_ppm", "power_w"]


class Resample(Exception):
    pass


class Rng:
    def __init__(self, seed: int) -> None:
        self._r = random.Random(seed)

    def u(self) -> float:
        return self._r.random()

    def randint(self, lo: int, hi: int) -> int:
        return lo + int(self.u() * (hi - lo + 1))

    def choice(self, seq):
        return seq[self.randint(0, len(seq) - 1)]

    def tri(self, a: int) -> int:
        return self.randint(-a, a) + self.randint(-a, a)

    def shuffle(self, items: List[Any]) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.randint(0, i)
            items[i], items[j] = items[j], items[i]


def _sha_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def item_seed(split_seed: int, split: str, level: int, template: str, index: int) -> int:
    return _sha_int(f"{GEN_VERSION}|{split_seed}|{split}|{level}|{template}|{index}")


def attempt_seed(seed: int, attempt: int) -> int:
    return _sha_int(f"{seed}|{attempt}")


class Knobs:
    def __init__(self, rows: int, n_channels: int, margin: float, distractors: int, closed_set: bool = True,
                 render: str = "table", setting: str = "mid") -> None:
        self.rows, self.n_channels, self.margin = rows, n_channels, margin
        self.distractors, self.closed_set, self.render, self.setting = distractors, closed_set, render, setting

    def as_dict(self) -> Dict[str, Any]:
        return {"rows": self.rows, "n_channels": self.n_channels, "margin": self.margin, "distractors": self.distractors,
                "closed_set": self.closed_set, "render": self.render, "setting": self.setting}


LEVEL_KNOBS: Dict[int, Dict[str, Tuple[int, int, float, int, bool]]] = {
    1: {"easy": (8, 2, 2.0, 0, True), "mid": (12, 2, 1.0, 1, True), "hard": (18, 3, 0.34, 2, False)},
    2: {"easy": (16, 2, 2.0, 0, True), "mid": (24, 3, 1.0, 1, True), "hard": (36, 4, 0.34, 3, False)},
    3: {"easy": (32, 5, 2.0, 1, True), "mid": (48, 5, 1.0, 2, True), "hard": (72, 6, 0.34, 4, False)},
    4: {"easy": (64, 6, 2.0, 1, True), "mid": (96, 6, 1.0, 2, True), "hard": (128, 6, 0.5, 4, False)},
}


def make_knobs(level: int, setting: str = "mid", render: str = "table") -> Knobs:
    rows, nch, mg, dis, closed = LEVEL_KNOBS[level][setting]
    return Knobs(rows, nch, mg, dis, closed, render, setting)


def margin_int(ch: str, mult: float) -> int:
    return max(1, int(CH[ch].margin * mult + 0.5))


def hhmm(minutes: int) -> str:
    return "%02d:%02d" % (minutes // 60, minutes % 60)


def fmt_val(ch: str, v: Optional[int]) -> str:
    if v is None:
        return "NA"
    d = CH[ch].dec
    if d == 0:
        return str(v)
    s = 10 ** d
    return "%d.%0*d" % (v // s, d, v % s)


class Table:
    def __init__(self, t0: int, channels: List[str], data: Dict[str, List[Optional[int]]]) -> None:
        self.t0, self.channels, self.data = t0, list(channels), data
        self.n = len(data[self.channels[0]])

    def tstr(self, i: int) -> str:
        return hhmm(self.t0 + STEP_MIN * i)

    def has_na(self) -> bool:
        return any(v is None for c in self.channels for v in self.data[c])

    def analog(self) -> List[str]:
        return [c for c in self.channels if c != "occupancy"]


def render_readings(tbl: Table, mode: str) -> str:
    if mode == "table":
        lines = ["time | " + " | ".join(tbl.channels)]
        for i in range(tbl.n):
            lines.append(tbl.tstr(i) + " | " + " | ".join(fmt_val(c, tbl.data[c][i]) for c in tbl.channels))
        return "\n".join(lines)
    if mode == "log":
        return "\n".join("%s %s=%s" % (tbl.tstr(i), c, fmt_val(c, tbl.data[c][i]))
                         for i in range(tbl.n) for c in tbl.channels)
    raise ValueError(mode)


def _occupancy_profile(rng: Rng, n: int, evt: bool) -> List[int]:
    occ = [0] * n
    z0 = rng.randint(4, 8) if evt else rng.randint(max(4, n // 8), max(5, n // 4))
    lo, hi = (1, 4) if evt else (1, 8)
    level = rng.randint(lo, min(hi, 4)) if evt else rng.randint(2, 7)
    i = z0
    while i < n:
        d = rng.randint(8, 24)
        for j in range(i, min(n, i + d)):
            occ[j] = level
        i += d
        level = min(hi, max(lo, level + rng.choice([-2, -1, 1, 2])))
    if not evt and rng.u() < 0.5:
        for j in range(n - rng.randint(3, max(3, n // 6)), n):
            occ[j] = 0
    return occ


def _power_profile(rng: Rng, n: int) -> List[int]:
    p = [0] * n
    state = rng.choice([0, 1, 2, 2])
    i = 0
    while i < n:
        d = rng.randint(8, 20)
        lvl = (rng.randint(150, 250), rng.randint(800, 1100), rng.randint(1600, 2300))[state]
        for j in range(i, min(n, i + d)):
            p[j] = lvl
        i += d
        state = rng.choice([s for s in (0, 1, 2) if s != state])
    return p


def _ramp(i: int, start: int, end: int, up: int) -> int:
    if i < start:
        return 0
    if i < end:
        return min(i - start + 1, up)
    return max(0, up - (i - end + 1))


CAUSES = {"A": "occupancy surge", "B": "HVAC fault", "C": "window open", "D": "machine imbalance", "E": "sensor fault"}


def simulate(rng: Rng, n: int, cause: Optional[str] = None, lead: Optional[str] = None
             ) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    def lag(ch: str, hi: int) -> int:
        if lead is None:
            return rng.randint(0, hi)
        return 0 if ch == lead else rng.randint(3, 4)

    evt = cause is not None
    info: Dict[str, Any] = {}
    s = e = 0
    if evt:
        ln = rng.randint(10, 16)
        s = rng.randint(26, n - ln - 18)
        e = s + ln
        info = {"cause": cause, "s": s, "e": e}
    occ = _occupancy_profile(rng, n, evt)
    pw = _power_profile(rng, n)
    if evt:
        v0 = {"A": rng.randint(1, 3), "B": rng.randint(3, 5), "C": rng.randint(4, 6),
              "D": rng.randint(2, 4), "E": rng.randint(2, 4)}[cause]
        on = rng.randint(1100, 2000)
        for j in range(s - 8, min(n, e + 14)):
            occ[j] = v0
            pw[j] = on
    vent_n, vent_d = [1] * n, [6] * n
    d_temp, d_rh = [0] * n, [0] * n
    pw_over: List[Optional[int]] = [None] * n
    pw_mult, vib_mult = [1000] * n, [1000] * n
    vib_over: List[Optional[int]] = [None] * n

    if cause == "A":
        jump = rng.randint(5, 7)
        for j in range(s, e):
            occ[j] += jump
    elif cause == "B":
        rise = rng.randint(22, 35)
        lag_t, lag_p, lag_v, lag_c = lag("temp_c", 3), lag("power_w", 2), lag("vib_mm_s", 2), lag("co2_ppm", 3)
        for i in range(n):
            if s + lag_c <= i < e:
                vent_n[i], vent_d[i] = 1, 24
            if s + lag_p <= i < e:
                pw_over[i] = rng.randint(0, 30)
            if s + lag_v <= i < e:
                vib_over[i] = rng.randint(8, 18)
            up = max(1, e - (s + lag_t))
            d_temp[i] += (rise * _ramp(i, s + lag_t, e, up)) // up
    elif cause == "C":
        dt, dr, sign = rng.randint(22, 35), rng.randint(8, 14), rng.choice([-1, 1])
        lag_v, lag_t, lag_r = lag("co2_ppm", 3), lag("temp_c", 3), lag("rh_pct", 3)
        for i in range(n):
            if s + lag_v <= i < e:
                vent_n[i], vent_d[i] = 1, 2
            d_temp[i] -= (dt * _ramp(i, s + lag_t, e, 3)) // 3
            d_rh[i] += sign * ((dr * _ramp(i, s + lag_r, e, 3)) // 3)
    elif cause == "D":
        mv, mp = rng.randint(2500, 3200), rng.randint(1060, 1120)
        lag_v, lag_p = lag("vib_mm_s", 2), lag("power_w", 2)
        for i in range(n):
            vib_mult[i] = 1000 + ((mv - 1000) * _ramp(i, s + lag_v, e, 2)) // 2
            pw_mult[i] = 1000 + ((mp - 1000) * _ramp(i, s + lag_p, e, 2)) // 2

    co2_state = [0] * n
    c = 420 + 150 * occ[0]
    co2_state[0] = c
    for t in range(1, n):
        c = c + 25 * occ[t] - ((c - 420) * vent_n[t]) // vent_d[t]
        co2_state[t] = c
    t_base = rng.randint(205, 240)
    drift = rng.randint(-4, 6)
    amp = rng.randint(3, 8)
    rh_off = rng.randint(-6, 6)
    vib0 = rng.randint(90, 140)
    out: Dict[str, List[int]] = {k: [0] * n for k in CH_ORDER}
    for t in range(n):
        w = occ[max(0, t - 5):t + 1]
        occ_eff = (4 * sum(w)) // (10 * len(w))
        bump = (2 * amp * min(t, n - 1 - t)) // max(1, n - 1)
        temp_state = t_base + bump + (drift * t) // max(1, n - 1) + occ_eff + d_temp[t]
        out["temp_c"][t] = max(100, temp_state + rng.tri(2))
        rh = 55 - (12 * (temp_state - 220)) // 100 + rh_off + d_rh[t] + rng.tri(1)
        out["rh_pct"][t] = min(85, max(20, rh))
        out["co2_ppm"][t] = co2_state[t] + rng.tri(4)
        p = pw_over[t] if pw_over[t] is not None else ((pw[t] * (1000 + rng.tri(15))) // 1000 * pw_mult[t]) // 1000
        out["power_w"][t] = max(0, p)
        v = vib_over[t] if vib_over[t] is not None else (((vib0 * (1150 if pw[t] >= 1500 else 1000)) // 1000) * vib_mult[t]) // 1000 + rng.tri(4)
        out["vib_mm_s"][t] = max(5, v)
        out["occupancy"][t] = occ[t]
    if cause == "E":
        stuck = rng.choice(["temp_c", "rh_pct", "co2_ppm", "power_w", "vib_mm_s"])
        info["stuck"] = stuck
        val = out[stuck][s - 1]
        for j in range(s, e):
            out[stuck][j] = val
    return out, info


def choose_channels(rng: Rng, level: int, n_channels: int, need: Tuple[str, ...]) -> List[str]:
    if level >= 4:
        return list(CH_ORDER)
    if level == 3:
        base = [c for c in CH_ORDER if c != "vib_mm_s"]
        return base + (["vib_mm_s"] if n_channels >= 6 else [])
    chosen = list(need)
    rest = [c for c in ANALOG_POOL if c not in chosen]
    rng.shuffle(rest)
    chosen += rest[:max(0, n_channels - len(chosen))]
    return [c for c in CH_ORDER if c in chosen]


def make_table(rng: Rng, level: int, kn: Knobs, need: Tuple[str, ...] = (), cause: Optional[str] = None,
               lead: Optional[str] = None) -> Tuple[Table, Dict[str, Any]]:
    n = kn.rows
    chans = choose_channels(rng, level, kn.n_channels, need)
    data_all, info = simulate(rng, n, cause, lead)
    t0 = STEP_MIN * rng.randint(48, min(204, 287 - n))
    return Table(t0, chans, {c: list(data_all[c]) for c in chans}), info


def median_int(vals: List[int]) -> Fraction:
    s = sorted(vals)
    k = len(s)
    return Fraction(s[k // 2]) if k % 2 else Fraction(s[k // 2 - 1] + s[k // 2], 2)


def add_distractors(rng: Rng, tbl: Table, k: int, protected: set) -> int:
    placed = 0
    for _ in range(k):
        kind = rng.choice(["na", "spike"])
        for _try in range(30):
            ch, i = rng.choice(tbl.channels), rng.randint(0, tbl.n - 1)
            if i in protected or tbl.data[ch][i] is None:
                continue
            if kind == "na":
                tbl.data[ch][i] = None
                placed += 1
                break
            if ch in ("co2_ppm", "power_w", "vib_mm_s"):
                med = median_int([v for v in tbl.data[ch] if v is not None])
                tbl.data[ch][i] = int(med * rng.randint(18, 24) / 10)
                placed += 1
                break
    return placed


def valid(vals: List[Optional[int]]) -> List[int]:
    return [v for v in vals if v is not None]


def margin_ok(vals: List[Optional[int]], t2: int, m2: int) -> bool:
    return all(abs(2 * v - t2) >= m2 for v in vals if v is not None)


def count_above(vals: List[Optional[int]], t2: int) -> int:
    return sum(1 for v in vals if v is not None and 2 * v > t2)


def runs_above(vals: List[Optional[int]], t2: int) -> List[Tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(list(vals) + [None]):
        if v is not None and 2 * v > t2:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    return runs


def equal_runs(vals: List[Optional[int]]) -> List[Tuple[int, int]]:
    runs, start = [], 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] is None or vals[i] != vals[start] or vals[start] is None:
            if vals[start] is not None:
                runs.append((start, i))
            start = i
    return runs


def fmt_thr(ch: str, t2: int) -> str:
    if t2 % 2 == 0:
        return fmt_val(ch, t2 // 2)
    d = CH[ch].dec + 1
    x = t2 * 5
    return "%d.%0*d" % (x // 10 ** d, d, x % 10 ** d)


def pick_threshold(rng: Rng, vals: List[Optional[int]], m2: int, lo2: int, hi2: int, pred: Callable[[int], bool]) -> int:
    cands = [t for t in range(max(0, lo2), hi2 + 1) if margin_ok(vals, t, m2) and pred(t)]
    whole = [t for t in cands if t % 2 == 0]
    pool = whole or cands
    if not pool:
        raise Resample()
    return rng.choice(pool)


def min_gap(vals: List[Optional[int]], t2: int) -> float:
    return min(abs(2 * v - t2) for v in vals if v is not None) / 2


def num_answer(ch: str, v: int):
    d = CH[ch].dec
    return v if d == 0 else round(v / 10 ** d, d)


def round_half_up(fr: Fraction) -> int:
    return (fr + Fraction(1, 2)).__floor__()


def dec_text(dec: int) -> str:
    return {0: "the nearest whole number", 1: "1 decimal place", 2: "2 decimal places"}[dec]


class Out:
    def __init__(self, tbl: Table, question: str, answer: Any, atype: str, decimals: int = 0, tol: float = 0,
                 choices: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> None:
        self.tbl, self.question, self.answer, self.atype = tbl, question, answer, atype
        self.decimals, self.tol, self.choices, self.meta = decimals, tol, choices or [], meta or {}


CH_RX = r"[a-z][a-z0-9_]*"
T_RX = r"\d\d:\d\d"
N_RX = r"\d+(?:\.\d+)?"
ROUND_RX = r"(?P<r>the nearest whole number|1 decimal place|2 decimal places)"
T15_OPTIONS = ("A) occupancy surge: many more people entered the room.\n"
               "B) HVAC fault: the HVAC unit stopped delivering air.\n"
               "C) window open: outdoor air entered the room.\n"
               "D) machine imbalance: the fan motor of the unit is out of balance.\n"
               "E) sensor fault: one sensor reports an invalid reading.")
T15_FMT = ("The system flagged the interval from {t1} to {t2} (both readings included) as abnormal. "
           "Which cause best explains the flagged interval?\n" + T15_OPTIONS)
T15_RX = (r"The system flagged the interval from (?P<t1>" + T_RX + r") to (?P<t2>" + T_RX +
          r") \(both readings included\) as abnormal\. Which cause best explains the flagged interval\?\n" + re.escape(T15_OPTIONS))
LIMITS_RX = r"(?P<lims>" + CH_RX + r" " + N_RX + r"(?:, " + CH_RX + r" " + N_RX + r")*)"

QUESTIONS: Dict[str, Tuple[str, str]] = {
    "T01": ("What was the reading of {ch} at {t}?",
            r"What was the reading of (?P<ch>" + CH_RX + r") at (?P<t>" + T_RX + r")\?"),
    "T02": ("What is the {w} value of {ch} in the window?",
            r"What is the (?P<w>highest|lowest) value of (?P<ch>" + CH_RX + r") in the window\?"),
    "T03": ("At what time was {ch} {w}?",
            r"At what time was (?P<ch>" + CH_RX + r") (?P<w>highest|lowest)\?"),
    "T04": ("Did {ch} exceed {L} at any time in the window?",
            r"Did (?P<ch>" + CH_RX + r") exceed (?P<L>" + N_RX + r") at any time in the window\?"),
    "T05": ("How many readings of {ch} are above {L}?",
            r"How many readings of (?P<ch>" + CH_RX + r") are above (?P<L>" + N_RX + r")\?"),
    "T06": ("For how many minutes in total was {ch} above {L}? Count each reading as {step} minutes.",
            r"For how many minutes in total was (?P<ch>" + CH_RX + r") above (?P<L>" + N_RX +
            r")\? Count each reading as (?P<step>\d+) minutes\."),
    "T07": ("What is the mean of {ch} {span}? Round the result to {rd}.",
            r"What is the mean of (?P<ch>" + CH_RX + r") (?:over the whole window|from (?P<t1>" + T_RX + r") to (?P<t2>" +
            T_RX + r") \(both readings included\))\? Round the result to " + ROUND_RX + r"\."),
    "T08": ("At what time does {ch} first exceed {L}? Answer none if no reading exceeds it.",
            r"At what time does (?P<ch>" + CH_RX + r") first exceed (?P<L>" + N_RX + r")\? Answer none if no reading exceeds it\."),
    "T09": ("Between {t1} and {t2}, is {ch} rising, falling or stable? Compare only the readings at {t1} and {t2}: "
            "rising if the reading at {t2} is at least {d} higher than the reading at {t1}, falling if it is at least {d} "
            "lower, otherwise stable.",
            r"Between (?P<t1>" + T_RX + r") and (?P<t2>" + T_RX + r"), is (?P<ch>" + CH_RX + r") rising, falling or stable\? "
            r"Compare only the readings at \1 and \2: rising if the reading at \2 is at least (?P<d>" + N_RX +
            r") higher than the reading at \1, falling if it is at least (?P=d) lower, otherwise stable\."),
    "T10": ("In {ch}, exactly one reading is more than 3 times the median of that channel over the window. "
            "At what time does it occur?",
            r"In (?P<ch>" + CH_RX + r"), exactly one reading is more than 3 times the median of that channel over the "
            r"window\. At what time does it occur\?"),
    "T11": ("Among {chs}, which channel reports exactly the same value in at least 6 consecutive readings? "
            "Answer none if no channel does.",
            r"Among (?P<chs>" + CH_RX + r"(?:, " + CH_RX + r")*), which channel reports exactly the same value in at least "
            r"6 consecutive readings\? Answer none if no channel does\."),
    "T12": ("Limits per channel: {lims}. Count each reading as {step} minutes. Which channel spends the most minutes "
            "above its limit?",
            r"Limits per channel: " + LIMITS_RX + r"\. Count each reading as (?P<step>\d+) minutes\. Which channel spends "
            r"the most minutes above its limit\?"),
    "T13": ("What is the mean of power_w over the readings where occupancy is 0? Round the result to the nearest "
            "whole number.",
            r"What is the mean of power_w over the readings where occupancy is 0\? Round the result to the nearest whole "
            r"number\."),
    "T14": ("An alert is raised at the reading that completes a run of {k} consecutive readings with co2_ppm above {L}. "
            "At what time is the first alert raised? Answer none if no alert is raised.",
            r"An alert is raised at the reading that completes a run of (?P<k>\d+) consecutive readings with co2_ppm above "
            r"(?P<L>" + N_RX + r")\. At what time is the first alert raised\? Answer none if no alert is raised\."),
    "T15": (T15_FMT, T15_RX),
    "T16": ("The flagged interval starts at {t}. For each channel, the baseline is the median of the 6 readings before "
            "{t} (the mean of the two middle values). Deviation limits: {lims}. Starting at {t} and going forward, which "
            "channel is the first to differ from its baseline by more than its deviation limit?",
            r"The flagged interval starts at (?P<t>" + T_RX + r")\. For each channel, the baseline is the median of the 6 "
            r"readings before (?P=t) \(the mean of the two middle values\)\. Deviation limits: " + LIMITS_RX +
            r"\. Starting at (?P=t) and going forward, which channel is the first to differ from its baseline by more than "
            r"its deviation limit\?"),
    "T17": ("An alert is active during every run of at least {k} consecutive readings with co2_ppm above {L}; the alert "
            "covers all readings of the run, from its first to its last. There is exactly one such run in the window. "
            "What is the highest temp_c reading while the alert is active?",
            r"An alert is active during every run of at least (?P<k>\d+) consecutive readings with co2_ppm above "
            r"(?P<L>" + N_RX + r"); the alert covers all readings of the run, from its first to its last\. There is "
            r"exactly one such run in the window\. What is the highest temp_c reading while the alert is active\?"),
}


def _bal(idx: int, mod: int) -> int:
    return idx % mod


def t01(rng: Rng, kn: Knobs, idx: int) -> Out:
    tbl, _ = make_table(rng, 1, kn)
    add_distractors(rng, tbl, kn.distractors, set())
    ch = rng.choice(tbl.channels)
    i = rng.choice([j for j, v in enumerate(tbl.data[ch]) if v is not None])
    return Out(tbl, QUESTIONS["T01"][0].format(ch=ch, t=tbl.tstr(i)), num_answer(ch, tbl.data[ch][i]), "number", CH[ch].dec)


def t02(rng: Rng, kn: Knobs, idx: int) -> Out:
    tbl, _ = make_table(rng, 1, kn)
    add_distractors(rng, tbl, kn.distractors, set())
    ch = rng.choice(tbl.channels)
    w = ("highest", "lowest")[_bal(idx, 2)]
    vals = valid(tbl.data[ch])
    v = max(vals) if w == "highest" else min(vals)
    return Out(tbl, QUESTIONS["T02"][0].format(w=w, ch=ch), num_answer(ch, v), "number", CH[ch].dec)


def t03(rng: Rng, kn: Knobs, idx: int) -> Out:
    tbl, _ = make_table(rng, 1, kn)
    add_distractors(rng, tbl, kn.distractors, set())
    ch = rng.choice(tbl.channels)
    w = ("highest", "lowest")[_bal(idx, 2)]
    vals = tbl.data[ch]
    order = sorted([(v, i) for i, v in enumerate(vals) if v is not None], reverse=(w == "highest"))
    gap = max(1, margin_int(ch, kn.margin) // 2)
    if len(order) < 2 or abs(order[0][0] - order[1][0]) < gap:
        raise Resample()
    return Out(tbl, QUESTIONS["T03"][0].format(ch=ch, w=w), tbl.tstr(order[0][1]), "time", 0,
               meta={"min_gap": abs(order[0][0] - order[1][0])})


def t04(rng: Rng, kn: Knobs, idx: int) -> Out:
    tbl, _ = make_table(rng, 1, kn)
    add_distractors(rng, tbl, kn.distractors, set())
    ch = rng.choice(tbl.channels)
    vals = tbl.data[ch]
    m = margin_int(ch, kn.margin)
    mx2 = 2 * max(valid(vals))
    yes = _bal(idx, 2) == 0
    if yes:
        thr = pick_threshold(rng, vals, m, mx2 - 10 * m, mx2 - m, lambda t: t < mx2)
    else:
        thr = pick_threshold(rng, vals, m, mx2 + m, mx2 + 10 * m, lambda t: t > mx2)
    return Out(tbl, QUESTIONS["T04"][0].format(ch=ch, L=fmt_thr(ch, thr)), "yes" if yes else "no", "yesno",
               meta={"min_gap": min_gap(vals, thr), "margin": m})


def _count_template(rng: Rng, kn: Knobs, idx: int, minutes: bool) -> Out:
    tbl, _ = make_table(rng, 2, kn)
    add_distractors(rng, tbl, kn.distractors, set())
    ch = rng.choice(tbl.channels)
    vals = tbl.data[ch]
    m = margin_int(ch, kn.margin)
    nv = len(valid(vals))
    thr = pick_threshold(rng, vals, m, 2 * min(valid(vals)), 2 * max(valid(vals)),
                         lambda t: 2 <= count_above(vals, t) <= max(3, int(0.6 * nv)))
    cnt = count_above(vals, thr)
    if minutes:
        q = QUESTIONS["T06"][0].format(ch=ch, L=fmt_thr(ch, thr), step=STEP_MIN)
        return Out(tbl, q, cnt * STEP_MIN, "number", 0, meta={"min_gap": min_gap(vals, thr), "margin": m})
    return Out(tbl, QUESTIONS["T05"][0].format(ch=ch, L=fmt_thr(ch, thr)), cnt, "number", 0,
               meta={"min_gap": min_gap(vals, thr), "margin": m})


def t05(rng: Rng, kn: Knobs, idx: int) -> Out:
    return _count_template(rng, kn, idx, False)


def t06(rng: Rng, kn: Knobs, idx: int) -> Out:
    return _count_template(rng, kn, idx, True)


def _mean_guard(total: int, cnt: int) -> int:
    fr = Fraction(total, cnt)
    frac = fr - fr.__floor__()
    if abs(frac - Fraction(1, 2)) < Fraction(1, 10):
        raise Resample()
    return round_half_up(fr)


def t07(rng: Rng, kn: Knobs, idx: int) -> Out:
    tbl, _ = make_table(rng, 2, kn)
    add_distractors(rng, tbl, kn.distractors, set())
    ch = rng.choice(tbl.channels)
    vals = tbl.data[ch]
    dec = CH[ch].dec
    if _bal(idx, 2) == 0:
        span, lo, hi = "over the whole window", 0, tbl.n - 1
    else:
        min_len = max(4, tbl.n // 3)
        lo = rng.randint(0, tbl.n - min_len)
        hi = rng.randint(lo + min_len - 1, tbl.n - 1)
        span = "from %s to %s (both readings included)" % (tbl.tstr(lo), tbl.tstr(hi))
    seg = valid(vals[lo:hi + 1])
    if len(seg) < 3:
        raise Resample()
    g = _mean_guard(sum(seg), len(seg))
    tol = 5 / 10 ** (dec + 1)
    return Out(tbl, QUESTIONS["T07"][0].format(ch=ch, span=span, rd=dec_text(dec)), num_answer(ch, g), "number", dec, tol)


def t08(rng: Rng, kn: Knobs, idx: int) -> Out:
    tbl, _ = make_table(rng, 2, kn)
    add_distractors(rng, tbl, kn.distractors, set())
    ch = rng.choice(tbl.channels)
    vals = tbl.data[ch]
    m = margin_int(ch, kn.margin)
    vv = valid(vals)
    none = _bal(idx, 5) == 0

    def first(t: int) -> int:
        return next((i for i, v in enumerate(vals) if v is not None and 2 * v > t), -1)

    if none:
        thr = pick_threshold(rng, vals, m, 2 * max(vv) + m, 2 * max(vv) + 6 * m, lambda t: first(t) < 0)
        ans = "none"
    else:
        thr = pick_threshold(rng, vals, m, 2 * min(vv), 2 * max(vv), lambda t: first(t) >= 2)
        ans = tbl.tstr(first(thr))
    return Out(tbl, QUESTIONS["T08"][0].format(ch=ch, L=fmt_thr(ch, thr)), ans, "time_or_none", 0,
               meta={"min_gap": min_gap(vals, thr), "margin": m})


def t09(rng: Rng, kn: Knobs, idx: int) -> Out:
    tbl, _ = make_table(rng, 2, kn)
    add_distractors(rng, tbl, kn.distractors, set())
    ch = rng.choice(tbl.channels)
    vals = tbl.data[ch]
    m, d = margin_int(ch, kn.margin), CH[ch].delta
    label = ("rising", "falling", "stable")[_bal(idx, 3)]
    gap = max(3, tbl.n // 4)
    pairs = []
    for i in range(tbl.n):
        for j in range(i + gap, tbl.n):
            if vals[i] is None or vals[j] is None:
                continue
            diff = vals[j] - vals[i]
            if (label == "rising" and diff >= d + m) or (label == "falling" and diff <= -(d + m)) or \
                    (label == "stable" and abs(diff) <= d - m):
                pairs.append((i, j))
    if not pairs:
        raise Resample()
    i, j = rng.choice(pairs)
    q = QUESTIONS["T09"][0].format(t1=tbl.tstr(i), t2=tbl.tstr(j), ch=ch, d=fmt_val(ch, d))
    return Out(tbl, q, label, "label", 0, choices=["rising", "falling", "stable"],
               meta={"diff": vals[j] - vals[i], "margin": m})


def _spike_guard(vals: List[Optional[int]]) -> int:
    vv = valid(vals)
    med = median_int(vv)
    hits = [i for i, v in enumerate(vals) if v is not None and v > 3 * med]
    if len(hits) != 1 or vals[hits[0]] < Fraction(33, 10) * med:
        raise Resample()
    if any(v is not None and i != hits[0] and v > Fraction(255, 100) * med for i, v in enumerate(vals)):
        raise Resample()
    return hits[0]


def t10(rng: Rng, kn: Knobs, idx: int) -> Out:
    tbl, _ = make_table(rng, 3, kn)
    ch = rng.choice(["co2_ppm", "power_w"])
    vals = tbl.data[ch]
    r = rng.randint(2, tbl.n - 3)
    med0 = median_int(valid(vals))
    vals[r] = int(med0 * rng.randint(34, 46) / 10)
    add_distractors(rng, tbl, kn.distractors, {r})
    row = _spike_guard(tbl.data[ch])
    return Out(tbl, QUESTIONS["T10"][0].format(ch=ch), tbl.tstr(row), "time", 0)


def t11(rng: Rng, kn: Knobs, idx: int) -> Out:
    tbl, _ = make_table(rng, 3, kn)
    analog = tbl.analog()
    has = _bal(idx, 5) != 0
    protected = set()
    stuck = None
    if has:
        stuck = rng.choice(analog)
        ln = rng.randint(6, 9)
        a = rng.randint(2, tbl.n - ln - 2)
        val = tbl.data[stuck][a]
        for j in range(a, a + ln):
            tbl.data[stuck][j] = val
        protected.update(range(a, a + ln))
    others = [c for c in analog if c != stuck]
    for _ in range(min(kn.distractors, 2)):
        c = rng.choice(others)
        ln = rng.randint(4, 5)
        a = rng.randint(1, tbl.n - ln - 1)
        if any(j in protected for j in range(a - 1, a + ln + 1)):
            continue
        val = tbl.data[c][a]
        for j in range(a, a + ln):
            tbl.data[c][j] = val
        protected.update(range(a - 1, a + ln + 1))
    add_distractors(rng, tbl, kn.distractors, protected)
    long = [c for c in analog if any(b - a >= 6 for a, b in equal_runs(tbl.data[c]))]
    if has and long != [stuck]:
        raise Resample()
    if not has and long:
        raise Resample()
    choices = analog + ["none"]
    return Out(tbl, QUESTIONS["T11"][0].format(chs=", ".join(analog)), stuck if has else "none", "channel_or_none", 0,
               choices=choices)


def _feasible(vals: List[Optional[int]], m: int) -> List[Tuple[int, int]]:
    vv = valid(vals)
    out = [(t, count_above(vals, t)) for t in range(2 * min(vv), 2 * max(vv) + 4 * m) if margin_ok(vals, t, m)]
    return [x for x in out if x[0] % 2 == 0] + [x for x in out if x[0] % 2 == 1]


def _prefer_whole(pool: List[Tuple[int, int]]) -> List[int]:
    whole = [t for t, _ in pool if t % 2 == 0]
    return whole or [t for t, _ in pool]


def t12(rng: Rng, kn: Knobs, idx: int) -> Out:
    tbl, _ = make_table(rng, 3, kn)
    add_distractors(rng, tbl, kn.distractors, set())
    analog = tbl.analog()
    cap = int(0.6 * tbl.n)
    feas = {c: _feasible(tbl.data[c], margin_int(c, kn.margin)) for c in analog}
    order = [analog[(idx + j) % len(analog)] for j in range(len(analog))]
    target = None
    for c in order:
        top = max([k for _, k in feas[c] if k <= cap] or [0])
        pool = [(t, k) for t, k in feas[c] if max(3, top // 2) <= k <= cap]
        if top >= 3 and pool:
            target = c
            break
    if target is None:
        raise Resample()
    lims = {target: rng.choice(_prefer_whole(pool))}
    cnt = {target: count_above(tbl.data[target], lims[target])}
    for c in analog:
        if c == target:
            continue
        pool = [(t, k) for t, k in feas[c] if 1 <= k <= cnt[target] - 2] or [(t, k) for t, k in feas[c] if k <= cnt[target] - 2]
        if not pool:
            raise Resample()
        lims[c] = rng.choice(_prefer_whole(pool))
        cnt[c] = count_above(tbl.data[c], lims[c])
    text = ", ".join("%s %s" % (c, fmt_thr(c, lims[c])) for c in analog)
    gap = min(min_gap(tbl.data[c], lims[c]) for c in analog)
    return Out(tbl, QUESTIONS["T12"][0].format(lims=text, step=STEP_MIN), target, "channel", 0, choices=list(analog),
               meta={"min_gap": gap})


def t13(rng: Rng, kn: Knobs, idx: int) -> Out:
    tbl, _ = make_table(rng, 3, kn)
    add_distractors(rng, tbl, kn.distractors, set())
    occ, pw = tbl.data["occupancy"], tbl.data["power_w"]
    rows = [i for i in range(tbl.n) if occ[i] == 0 and pw[i] is not None]
    if len(rows) < 4 or sum(1 for v in occ if v is not None and v > 0) < 4:
        raise Resample()
    g = _mean_guard(sum(pw[i] for i in rows), len(rows))
    return Out(tbl, QUESTIONS["T13"][0], g, "number", 0, tol=0.5)


def t14(rng: Rng, kn: Knobs, idx: int) -> Out:
    tbl, _ = make_table(rng, 3, kn)
    add_distractors(rng, tbl, kn.distractors, set())
    vals = tbl.data["co2_ppm"]
    vv = valid(vals)
    m = margin_int("co2_ppm", kn.margin)
    k = rng.randint(3, 4)
    none = _bal(idx, 5) == 0

    def first(t: int) -> int:
        for a, b in runs_above(vals, t):
            if b - a >= k:
                return a + k - 1
        return -1

    def longest(t: int) -> int:
        return max([b - a for a, b in runs_above(vals, t)] or [0])

    if none:
        try:
            thr = pick_threshold(rng, vals, m, 2 * min(vv), 2 * max(vv) + 6 * m,
                                 lambda t: first(t) < 0 and 1 <= longest(t) <= k - 1)
        except Resample:
            thr = pick_threshold(rng, vals, m, 2 * min(vv), 2 * max(vv) + 6 * m, lambda t: first(t) < 0)
        ans = "none"
    else:
        thr = pick_threshold(rng, vals, m, 2 * min(vv), 2 * max(vv), lambda t: first(t) >= 0)
        ans = tbl.tstr(first(thr))
    return Out(tbl, QUESTIONS["T14"][0].format(L=fmt_thr("co2_ppm", thr), k=k), ans, "time_or_none", 0,
               meta={"min_gap": min_gap(vals, thr), "margin": m, "k": k})


def to_frac_cols(tbl: Table) -> Dict[str, List[Optional[Fraction]]]:
    return {c: [None if v is None else Fraction(v, 10 ** CH[c].dec) for v in tbl.data[c]] for c in tbl.channels}


def _med_frac(xs: List[Fraction]) -> Fraction:
    s = sorted(xs)
    k = len(s)
    return s[k // 2] if k % 2 else (s[k // 2 - 1] + s[k // 2]) / 2


def signature_causes(cols: Dict[str, List[Optional[Fraction]]], s: int, e: int) -> List[str]:
    def seg(c: str, a: int, b: int) -> List[Fraction]:
        xs = cols[c][a:b]
        if any(x is None for x in xs):
            raise ValueError("NA in signature window of " + c)
        return xs

    pre = {c: _med_frac(seg(c, s - 6, s)) for c in cols}
    last = {c: _med_frac(seg(c, e - 3, e)) for c in cols}
    d = {c: last[c] - pre[c] for c in cols}
    hit = []
    if d["occupancy"] >= 4 and d["co2_ppm"] >= 250:
        hit.append("A")
    if d["occupancy"] == 0 and last["power_w"] * 5 <= pre["power_w"] and d["temp_c"] >= Fraction(3, 2) and d["co2_ppm"] >= 250:
        hit.append("B")
    if d["temp_c"] <= Fraction(-3, 2) and d["co2_ppm"] <= -250 and abs(d["rh_pct"]) >= 6:
        hit.append("C")
    if pre["vib_mm_s"] > 0 and pre["power_w"] > 0:
        rv, rp = last["vib_mm_s"] / pre["vib_mm_s"], last["power_w"] / pre["power_w"]
        if rv >= 2 and Fraction(103, 100) <= rp <= Fraction(5, 4) and abs(d["temp_c"]) < 1 and abs(d["co2_ppm"]) < 120:
            hit.append("D")
    for c in cols:
        if c == "occupancy":
            continue
        xs = cols[c][s:e]
        best = run = 1
        for i in range(1, len(xs)):
            run = run + 1 if xs[i] == xs[i - 1] else 1
            best = max(best, run)
        if best >= 8:
            hit.append("E")
            break
    return hit


def _event_table(rng: Rng, kn: Knobs, cause: str, lead: Optional[str] = None) -> Tuple[Table, Dict[str, Any], set]:
    tbl, info = make_table(rng, 4, kn, cause=cause, lead=lead)
    protected = set(range(info["s"] - 8, info["e"] + 4))
    return tbl, info, protected


def _no_long_runs(tbl: Table, info: Dict[str, Any]) -> None:
    for c in tbl.analog():
        for a, b in equal_runs(tbl.data[c]):
            if b - a >= 8 and not (info["cause"] == "E" and c == info.get("stuck")):
                raise Resample()


def t15(rng: Rng, kn: Knobs, idx: int) -> Out:
    cause = "ABCDE"[_bal(idx, 5)]
    tbl, info, prot = _event_table(rng, kn, cause)
    add_distractors(rng, tbl, kn.distractors, prot)
    _no_long_runs(tbl, info)
    s, e = info["s"], info["e"]
    try:
        hit = signature_causes(to_frac_cols(tbl), s, e)
    except ValueError:
        raise Resample()
    if hit != [cause]:
        raise Resample()
    q = QUESTIONS["T15"][0].format(t1=tbl.tstr(s), t2=tbl.tstr(e - 1))
    return Out(tbl, q, cause, "letter", 0, choices=list("ABCDE"),
               meta={"event": {"cause": cause, "name": CAUSES[cause], "start": tbl.tstr(s), "end": tbl.tstr(e - 1)}})


def _first_deviation(cols: Dict[str, List[Optional[Fraction]]], s: int, lims: Dict[str, Fraction]) -> List[Tuple[int, List[str]]]:
    base = {c: _med_frac(cols[c][s - 6:s]) for c in cols}
    n = len(next(iter(cols.values())))
    for i in range(s, n):
        ch = [c for c in cols if cols[c][i] is not None and abs(cols[c][i] - base[c]) > lims[c]]
        if ch:
            return [(i, ch)]
    return []


T16_LEADS = [("occupancy", "A"), ("co2_ppm", "C"), ("temp_c", "C"), ("rh_pct", "C"), ("power_w", "B"), ("vib_mm_s", "D")]


def t16(rng: Rng, kn: Knobs, idx: int) -> Out:
    lead, cause = T16_LEADS[_bal(idx, len(T16_LEADS))]
    tbl, info, prot = _event_table(rng, kn, cause, lead)
    add_distractors(rng, tbl, kn.distractors, prot)
    s, e = info["s"], info["e"]
    cols = to_frac_cols(tbl)
    lims = {c: Fraction(CH[c].dev, 10 ** CH[c].dec) for c in tbl.channels}
    try:
        res = _first_deviation(cols, s, lims)
    except TypeError:
        raise Resample()
    if not res or len(res[0][1]) != 1 or res[0][0] >= e + 3:
        raise Resample()
    text = ", ".join("%s %s" % (c, fmt_val(c, CH[c].dev)) for c in tbl.channels)
    q = QUESTIONS["T16"][0].format(t=tbl.tstr(s), lims=text)
    return Out(tbl, q, res[0][1][0], "channel", 0, choices=list(tbl.channels),
               meta={"event": {"cause": cause, "name": CAUSES[cause], "start": tbl.tstr(s), "end": tbl.tstr(e - 1)}})


def t17(rng: Rng, kn: Knobs, idx: int) -> Out:
    cause = "AB"[_bal(idx, 2)]
    tbl, info, prot = _event_table(rng, kn, cause)
    add_distractors(rng, tbl, kn.distractors, prot)
    co2 = tbl.data["co2_ppm"]
    vv = valid(co2)
    m = margin_int("co2_ppm", kn.margin)
    k = rng.randint(3, 5)
    thr = pick_threshold(rng, co2, m, 2 * min(vv), 2 * max(vv),
                         lambda t: len([1 for a, b in runs_above(co2, t) if b - a >= k]) == 1)
    a, b = [r for r in runs_above(co2, thr) if r[1] - r[0] >= k][0]
    temps = valid(tbl.data["temp_c"][a:b])
    if not temps:
        raise Resample()
    q = QUESTIONS["T17"][0].format(k=k, L=fmt_thr("co2_ppm", thr))
    return Out(tbl, q, num_answer("temp_c", max(temps)), "number", 1,
               meta={"min_gap": min_gap(co2, thr), "margin": m, "k": k,
                     "event": {"cause": cause, "name": CAUSES[cause]}})


TEMPLATES: Dict[str, Tuple[int, str, Callable[[Rng, Knobs, int], Out]]] = {
    "T01": (1, "lookup", t01), "T02": (1, "extremum value", t02), "T03": (1, "extremum time", t03),
    "T04": (1, "limit crossed", t04),
    "T05": (2, "count above", t05), "T06": (2, "minutes above", t06), "T07": (2, "mean", t07),
    "T08": (2, "first crossing", t08), "T09": (2, "trend", t09),
    "T10": (3, "spike time", t10), "T11": (3, "stuck channel", t11), "T12": (3, "most minutes above limit", t12),
    "T13": (3, "conditional mean", t13), "T14": (3, "alert rule", t14),
    "T15": (4, "cause", t15), "T16": (4, "first deviating channel", t16), "T17": (4, "alert interval maximum", t17),
}
LEVEL_TEMPLATES = {lv: [t for t, v in TEMPLATES.items() if v[0] == lv] for lv in (1, 2, 3, 4)}


SITES = ["Meeting room 2.14, second floor of an office building", "Classroom B-107 of a secondary school",
         "Open-plan office, zone 3", "Seminar room S4 of a university building", "Library reading room, ground floor",
         "Training lab 5, east wing", "Conference room C-12", "Clinic waiting room, level 1"]
UNITS = ["fan-coil unit FCU-%02d", "air-handling unit AHU-%d", "ventilation unit VU-%02d", "heat-pump unit HP-%02d"]


def format_text(atype: str, decimals: int, choices: List[str], closed: bool) -> str:
    head = 'Answer format: reply with one JSON object and nothing else, in the form {"answer": <value>}, where <value> is '
    if atype == "number":
        body = "a number written without unit (a whole number when the quantity has no decimals)."
    elif atype == "time":
        body = 'a time written as "HH:MM", taken from the time column of the readings.'
    elif atype == "time_or_none":
        body = 'a time written as "HH:MM", taken from the readings, or "none" when there is no such time.'
    elif atype == "yesno":
        body = 'the string "yes" or the string "no".'
    elif atype == "label":
        body = 'one of the strings "rising", "falling", "stable".'
    elif atype in ("channel", "channel_or_none"):
        if closed:
            body = "one of the strings: " + ", ".join('"%s"' % c for c in choices) + "."
        else:
            body = "a channel name written as in the readings" + (', or "none"' if atype == "channel_or_none" else "") + "."
    elif atype == "letter":
        body = 'the letter of the best option, one of "A", "B", "C", "D", "E".'
    else:
        raise ValueError(atype)
    return head + body


def render_system(tbl: Table, site: str, unit: str, render: str, fmt: str) -> str:
    ch_lines = "\n".join("- %s: %s" % (c, CH[c].desc) for c in tbl.channels)
    parts = [
        "You are the analysis assistant of a building monitoring system. Answer the question using only the sensor data "
        "below, and compute from the values exactly as printed.",
        "",
        "Site: %s. Unit: %s." % (site, unit),
        "Sampling: one reading every %d minutes, %d readings from %s to %s." % (STEP_MIN, tbl.n, tbl.tstr(0), tbl.tstr(tbl.n - 1)),
        "Channels:", ch_lines,
    ]
    if tbl.has_na():
        parts.append("NA means that no reading was received. Ignore NA readings in every calculation: an NA reading is not "
                     "above any limit, and it ends a run of consecutive readings.")
    parts.append("")
    parts.append("Readings (one row per time):" if render == "table" else
                 'Readings (one line per reading, written "time channel=value"):')
    parts.append(render_readings(tbl, render))
    parts.append("")
    parts.append(fmt)
    return "\n".join(parts)


def build_item(split: str, split_seed: int, template: str, index: int, setting: str = "mid", render: str = "table") -> Dict[str, Any]:
    level, name, fn = TEMPLATES[template]
    kn = make_knobs(level, setting, render)
    seed = item_seed(split_seed, split, level, template, index)
    for attempt in range(MAX_ATTEMPTS):
        rng = Rng(attempt_seed(seed, attempt))
        try:
            out = fn(rng, kn, index)
        except Resample:
            continue
        site = rng.choice(SITES)
        unit = rng.choice(UNITS)
        unit = unit % rng.randint(1, 12) if "%" in unit else unit
        fmt = format_text(out.atype, out.decimals, out.choices, kn.closed_set)
        item = {
            "id": "sensor-%s-%s-%03d" % (split, template, index),
            "split": split, "level": level, "template": template, "template_name": name, "index": index,
            "seed": seed, "gen_version": GEN_VERSION,
            "system": render_system(out.tbl, site, unit, kn.render, fmt),
            "question": out.question,
            "answer": out.answer, "answer_type": out.atype, "decimals": out.decimals, "tolerance": out.tol,
            "choices": out.choices,
            "meta": dict(out.meta, knobs=kn.as_dict(), attempts=attempt + 1, n_na=sum(
                1 for c in out.tbl.channels for v in out.tbl.data[c] if v is None)),
        }
        ok, msg = verify_item(item)
        if not ok:
            raise AssertionError("verifier disagrees on %s: %s" % (item["id"], msg))
        return item
    raise RuntimeError("no valid item after %d attempts: %s %s" % (MAX_ATTEMPTS, template, index))


SPLITS: Dict[str, Dict[str, Any]] = {
    "test": {"seed": 20261017, "per_level": {1: 100, 2: 100, 3: 100, 4: 100}, "rotate": False},
    "heldout": {"seed": 20261018, "per_level": {1: 75, 2: 75, 3: 75, 4: 75}, "rotate": False},
    "calib": {"seed": 20261019, "per_level": {1: 50, 2: 50, 3: 50, 4: 50}, "rotate": True},
}
ROTATION = ["mid", "easy", "hard"]


def template_counts(split: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for lv, total in SPLITS[split]["per_level"].items():
        tps = LEVEL_TEMPLATES[lv]
        base, rem = divmod(total, len(tps))
        for k, t in enumerate(tps):
            out[t] = base + (1 if k >= len(tps) - rem else 0)
    return out


def generate_split(split: str, render: str = "table") -> List[Dict[str, Any]]:
    cfg = SPLITS[split]
    items = []
    for t, cnt in template_counts(split).items():
        for i in range(cnt):
            setting = ROTATION[i % 3] if cfg["rotate"] else "mid"
            items.append(build_item(split, cfg["seed"], t, i, setting, render))
    items.sort(key=lambda it: (it["index"], it["template"]))
    return items


class Parsed:
    def __init__(self, times: List[int], channels: List[str], cols: Dict[str, List[Optional[Fraction]]]) -> None:
        self.times, self.channels, self.cols = times, channels, cols
        self.n = len(times)

    def row(self, tstr: str) -> int:
        mins = int(tstr[:2]) * 60 + int(tstr[3:])
        return self.times.index(mins)

    def t(self, i: int) -> str:
        return hhmm(self.times[i])


_NUM = re.compile(r"^\d+(?:\.\d+)?$")


def _val(s: str) -> Optional[Fraction]:
    if s == "NA":
        return None
    if not _NUM.match(s):
        raise ValueError("bad value %r" % s)
    return Fraction(s)


def parse_system(system: str) -> Parsed:
    lines = system.split("\n")
    k = next(i for i, ln in enumerate(lines) if ln.startswith("Readings ("))
    table_mode = "one row per time" in lines[k]
    body = []
    for ln in lines[k + 1:]:
        if not ln.strip():
            break
        body.append(ln)
    times: List[int] = []
    if table_mode:
        header = body[0].split(" | ")
        if header[0] != "time":
            raise ValueError("bad header")
        chans = header[1:]
        cols: Dict[str, List[Optional[Fraction]]] = {c: [] for c in chans}
        for ln in body[1:]:
            cells = ln.split(" | ")
            if len(cells) != len(header) or not re.fullmatch(T_RX, cells[0]):
                raise ValueError("bad row %r" % ln)
            times.append(int(cells[0][:2]) * 60 + int(cells[0][3:]))
            for c, v in zip(chans, cells[1:]):
                cols[c].append(_val(v))
    else:
        chans, cols = [], {}
        for ln in body:
            m = re.fullmatch(r"(" + T_RX + r") (" + CH_RX + r")=(\S+)", ln)
            if not m:
                raise ValueError("bad line %r" % ln)
            tm = int(m.group(1)[:2]) * 60 + int(m.group(1)[3:])
            if not times or times[-1] != tm:
                times.append(tm)
            c = m.group(2)
            if c not in cols:
                cols[c] = []
                chans.append(c)
            cols[c].append(_val(m.group(3)))
        if any(len(v) != len(times) for v in cols.values()):
            raise ValueError("ragged log")
    if any(b - a != STEP_MIN for a, b in zip(times, times[1:])):
        raise ValueError("irregular sampling")
    m = re.search(r"one reading every (\d+) minutes, (\d+) readings from (" + T_RX + ") to (" + T_RX + r")\.", system)
    if not m or int(m.group(2)) != len(times) or int(m.group(1)) != STEP_MIN:
        raise ValueError("sampling line disagrees with the readings")
    return Parsed(times, chans, cols)


def _vals(P: Parsed, ch: str, a: int = 0, b: Optional[int] = None) -> List[Fraction]:
    return [x for x in P.cols[ch][a:b] if x is not None]


def _runs(xs: List[Optional[Fraction]], pred: Callable[[Fraction], bool]) -> List[Tuple[int, int]]:
    out, st = [], None
    for i in range(len(xs) + 1):
        ok = i < len(xs) and xs[i] is not None and pred(xs[i])
        if ok and st is None:
            st = i
        if not ok and st is not None:
            out.append((st, i))
            st = None
    return out


def _round_to(fr: Fraction, d: int) -> Fraction:
    sc = 10 ** d
    return Fraction((fr * sc + Fraction(1, 2)).__floor__(), sc)


def _lims(text: str) -> Dict[str, Fraction]:
    return {a: Fraction(b) for a, b in (p.split(" ") for p in text.split(", "))}


def _rd(r: str) -> int:
    return {"the nearest whole number": 0, "1 decimal place": 1, "2 decimal places": 2}[r]


def recompute(template: str, P: Parsed, q: Dict[str, str]) -> Any:
    g = q.get
    if template == "T01":
        return P.cols[g("ch")][P.row(g("t"))]
    if template == "T02":
        v = _vals(P, g("ch"))
        return max(v) if g("w") == "highest" else min(v)
    if template == "T03":
        xs = P.cols[g("ch")]
        best = max if g("w") == "highest" else min
        top = best(x for x in xs if x is not None)
        idx = [i for i, x in enumerate(xs) if x == top]
        if len(idx) != 1:
            raise ValueError("tie for the extremum")
        return P.t(idx[0])
    if template == "T04":
        return "yes" if any(x > Fraction(g("L")) for x in _vals(P, g("ch"))) else "no"
    if template == "T05":
        return sum(1 for x in _vals(P, g("ch")) if x > Fraction(g("L")))
    if template == "T06":
        return int(g("step")) * sum(1 for x in _vals(P, g("ch")) if x > Fraction(g("L")))
    if template == "T07":
        a, b = (P.row(g("t1")), P.row(g("t2"))) if g("t1") else (0, P.n - 1)
        seg = _vals(P, g("ch"), a, b + 1)
        return _round_to(sum(seg) / len(seg), _rd(g("r")))
    if template == "T08":
        thr = Fraction(g("L"))
        for i, x in enumerate(P.cols[g("ch")]):
            if x is not None and x > thr:
                return P.t(i)
        return "none"
    if template == "T09":
        xs = P.cols[g("ch")]
        diff = xs[P.row(g("t2"))] - xs[P.row(g("t1"))]
        d = Fraction(g("d"))
        return "rising" if diff >= d else ("falling" if diff <= -d else "stable")
    if template == "T10":
        xs = P.cols[g("ch")]
        med = _med_frac([x for x in xs if x is not None])
        hits = [i for i, x in enumerate(xs) if x is not None and x > 3 * med]
        if len(hits) != 1:
            raise ValueError("%d readings above 3 times the median" % len(hits))
        return P.t(hits[0])
    if template == "T11":
        names = g("chs").split(", ")
        long = []
        for c in names:
            xs, best, run = P.cols[c], 0, 0
            for i, x in enumerate(xs):
                run = run + 1 if (x is not None and i > 0 and xs[i - 1] == x) else (1 if x is not None else 0)
                best = max(best, run)
            if best >= 6:
                long.append(c)
        if len(long) > 1:
            raise ValueError("several stuck channels")
        return long[0] if long else "none"
    if template == "T12":
        lims = _lims(g("lims"))
        mins = {c: int(g("step")) * sum(1 for x in _vals(P, c) if x > lim) for c, lim in lims.items()}
        top = max(mins.values())
        win = [c for c, v in mins.items() if v == top]
        if len(win) != 1:
            raise ValueError("tie for most minutes")
        return win[0]
    if template == "T13":
        rows = [i for i in range(P.n) if P.cols["occupancy"][i] == 0 and P.cols["power_w"][i] is not None]
        return _round_to(sum(P.cols["power_w"][i] for i in rows) / len(rows), 0)
    if template == "T14":
        k, thr = int(g("k")), Fraction(g("L"))
        for a, b in _runs(P.cols["co2_ppm"], lambda x: x > thr):
            if b - a >= k:
                return P.t(a + k - 1)
        return "none"
    if template == "T15":
        s, e = P.row(g("t1")), P.row(g("t2")) + 1
        hit = signature_causes(P.cols, s, e)
        if len(hit) != 1:
            raise ValueError("signature matches %s" % hit)
        return hit[0]
    if template == "T16":
        s = P.row(g("t"))
        lims = _lims(g("lims"))
        res = _first_deviation({c: P.cols[c] for c in lims}, s, lims)
        if not res or len(res[0][1]) != 1:
            raise ValueError("no unique first channel")
        return res[0][1][0]
    if template == "T17":
        k, thr = int(g("k")), Fraction(g("L"))
        runs = [r for r in _runs(P.cols["co2_ppm"], lambda x: x > thr) if r[1] - r[0] >= k]
        if len(runs) != 1:
            raise ValueError("%d qualifying runs" % len(runs))
        return max(_vals(P, "temp_c", runs[0][0], runs[0][1]))
    raise KeyError(template)


def verify_item(item: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        t = item["template"]
        m = re.fullmatch(QUESTIONS[t][1], item["question"], re.S)
        if not m:
            return False, "question does not match the template pattern"
        P = parse_system(item["system"])
        if not item["system"].endswith(format_text(item["answer_type"], item["decimals"], item["choices"],
                                                   item["meta"]["knobs"]["closed_set"])):
            return False, "format instruction differs from the answer type"
        got = recompute(t, P, m.groupdict())
        gold = item["answer"]
        if isinstance(got, Fraction):
            if isinstance(gold, bool) or not isinstance(gold, (int, float)):
                return False, "numeric gold expected"
            if Fraction(str(gold)) != got:
                return False, "recomputed %s, stored %r" % (got, gold)
        elif got != gold:
            return False, "recomputed %r, stored %r" % (got, gold)
        if item["answer_type"] in ("channel", "channel_or_none", "label", "letter") and \
                gold not in item["choices"]:
            return False, "gold not in choices"
        if item["answer_type"] == "number" and item["tolerance"] < 0:
            return False, "negative tolerance"
        return True, "ok"
    except (ValueError, KeyError, StopIteration, ZeroDivisionError, TypeError) as exc:
        return False, "%s: %s" % (type(exc).__name__, exc)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 sensor_gen.py", description="Show or verify generated sensor items.")
    ap.add_argument("--split", default="test", choices=list(SPLITS))
    ap.add_argument("--template", default="T01", choices=list(TEMPLATES))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--setting", default="mid", choices=["easy", "mid", "hard"])
    ap.add_argument("--render", default="table", choices=["table", "log"])
    ap.add_argument("--verify-split", action="store_true", help="generate the whole split in memory and report verifier agreement")
    a = ap.parse_args(argv)
    if a.verify_split:
        items = generate_split(a.split, a.render)
        bad = [it["id"] for it in items if not verify_item(it)[0]]
        print("%s: %d items, %d verifier disagreements" % (a.split, len(items), len(bad)))
        return 1 if bad else 0
    it = build_item(a.split, SPLITS[a.split]["seed"], a.template, a.index, a.setting, a.render)
    print(it["system"])
    print("\nUSER: " + it["question"])
    print("\nGOLD: %r (%s)  verifier: %s" % (it["answer"], it["answer_type"], verify_item(it)[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
