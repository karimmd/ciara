from __future__ import annotations

import hashlib
import math
import random
from typing import Sequence


def stream(seed: int, *labels: object) -> random.Random:
    key = "|".join([str(int(seed))] + [str(x) for x in labels])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def expo(rng: random.Random, rate: float) -> float:
    return -math.log(1.0 - rng.random()) / rate


def normal(rng: random.Random) -> float:
    u1 = 1.0 - rng.random()
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def gamma_mean1(rng: random.Random, cv: float) -> float:
    if cv <= 0.0:
        return 1.0
    shape = 1.0 / (cv * cv)
    boost = 1.0
    if shape < 1.0:
        boost = rng.random() ** (1.0 / shape)
        shape += 1.0
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = normal(rng)
        v = (1.0 + c * x) ** 3
        if v <= 0.0:
            continue
        u = rng.random()
        if math.log(1.0 - u) < 0.5 * x * x + d - d * v + d * math.log(v):
            return boost * d * v * cv * cv


def pick(rng: random.Random, weights: Sequence[float]) -> int:
    return pick_u(rng.random(), weights)


def pick_u(u: float, weights: Sequence[float]) -> int:
    total = float(sum(weights))
    x = u * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if x < acc:
            return i
    return len(weights) - 1
