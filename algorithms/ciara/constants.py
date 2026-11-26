from __future__ import annotations

import math
from typing import Any, Optional, Sequence

OUTCOMES = ("Y_Q", "Y_L_f1", "Y_L_f2", "Y_D", "Y_E_A", "Y_E_B", "Y_E_C", "Y_C")
BUDGET_KEYS = ("Y_L_f1", "Y_L_f2", "Y_D", "Y_E_A", "Y_E_B", "Y_E_C", "Y_C")
BUDGETS_TIGHT = {"Y_L_f1": 0.05, "Y_L_f2": 0.05, "Y_D": 0.05, "Y_E_A": 0.5, "Y_E_B": 0.5, "Y_E_C": 0.5, "Y_C": 0.30}
BUDGETS = {"Y_L_f1": 0.10, "Y_L_f2": 0.10, "Y_D": 0.10, "Y_E_A": 0.5, "Y_E_B": 0.5, "Y_E_C": 0.5, "Y_C": 0.30}
COUNT_KEY = {"Y_Q": "n", "Y_D": "n", "Y_C": "n", "Y_L_f1": "n_f1", "Y_L_f2": "n_f2", "Y_E_A": "n_A", "Y_E_B": "n_B", "Y_E_C": "n_C"}
DELTA = 0.05
K_OUT = 8
EPOCH_S = 30.0
EPS = 0.5
T_BAR = 5.0
B_EPOCHS, L_EPOCHS = 25, 50
L_DEP = 1200
N_NODES = 4
K7 = 20160 // L_EPOCHS
LOOK_DAYS = (4, 7, 10)
EPOCHS_PER_DAY = 2880


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def mean(xs: Sequence[float]) -> Optional[float]:
    v = [float(x) for x in xs if finite(x)]
    return sum(v) / len(v) if v else None


def variance(xs: Sequence[float]) -> Optional[float]:
    v = [float(x) for x in xs if finite(x)]
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)
