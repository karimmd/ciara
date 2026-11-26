from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

TOL = 1e-9


@dataclass(frozen=True)
class LPResult:
    status: str
    value: float
    p: Optional[List[float]]


def _pivot(tab: List[List[float]], basis: List[int], row: int, col: int) -> None:
    piv = tab[row][col]
    tab[row] = [v / piv for v in tab[row]]
    for r in range(len(tab)):
        if r != row and abs(tab[r][col]) > 0.0:
            f = tab[r][col]
            tr = tab[row]
            tab[r] = [a - f * b for a, b in zip(tab[r], tr)]
    basis[row] = col


def _run(tab: List[List[float]], basis: List[int], cost: List[float], allowed: Sequence[bool]) -> None:
    m = len(tab)
    ncols = len(tab[0]) - 1
    while True:
        enter = -1
        for j in range(ncols):
            if not allowed[j] or j in basis:
                continue
            z = cost[j] - sum(cost[basis[r]] * tab[r][j] for r in range(m))
            if z < -TOL:
                enter = j
                break
        if enter < 0:
            return
        leave = -1
        best = float("inf")
        for r in range(m):
            a = tab[r][enter]
            if a > TOL:
                ratio = tab[r][-1] / a
                if ratio < best - TOL or (abs(ratio - best) <= TOL and basis[r] < basis[leave]):
                    best = ratio
                    leave = r
        if leave < 0:
            raise ArithmeticError("unbounded linear program")
        _pivot(tab, basis, leave, enter)


def solve_distribution_lp(c: Sequence[float], A: Sequence[Sequence[float]], b: Sequence[float]) -> LPResult:
    n = len(c)
    d = len(A)
    if n == 0:
        raise ValueError("no arms")
    if any(len(row) != n for row in A) or len(b) != d:
        raise ValueError("inconsistent LP shapes")
    rows: List[List[float]] = []
    rhs: List[float] = []
    slack_sign: List[float] = []
    for i in range(d):
        row = [float(v) for v in A[i]]
        bi = float(b[i])
        sgn = 1.0
        if bi < 0.0:
            row = [-v for v in row]
            bi = -bi
            sgn = -1.0
        rows.append(row)
        rhs.append(bi)
        slack_sign.append(sgn)
    rows.append([1.0] * n)
    rhs.append(1.0)
    slack_sign.append(0.0)
    m = d + 1
    art_rows = [r for r in range(m) if slack_sign[r] != 1.0]
    n_art = len(art_rows)
    ncols = n + d + n_art
    tab: List[List[float]] = []
    basis: List[int] = []
    for r in range(m):
        line = rows[r] + [0.0] * (d + n_art) + [rhs[r]]
        if r < d:
            line[n + r] = slack_sign[r]
        tab.append(line)
    for k, r in enumerate(art_rows):
        tab[r][n + d + k] = 1.0
    for r in range(m):
        basis.append(n + r if (r < d and slack_sign[r] == 1.0) else n + d + art_rows.index(r))

    if n_art:
        cost1 = [0.0] * (n + d) + [1.0] * n_art
        _run(tab, basis, cost1, [True] * ncols)
        infeas = sum(tab[r][-1] for r in range(m) if basis[r] >= n + d)
        if infeas > 1e-7:
            return LPResult("infeasible", float("nan"), None)
        keep = list(range(m))
        for r in range(m):
            if basis[r] >= n + d:
                col = next((j for j in range(n + d) if abs(tab[r][j]) > 1e-9), -1)
                if col >= 0:
                    _pivot(tab, basis, r, col)
                else:
                    keep.remove(r)
        tab = [tab[r] for r in keep]
        basis = [basis[r] for r in keep]
    cost2 = [-float(v) for v in c] + [0.0] * (d + n_art)
    allowed = [True] * (n + d) + [False] * n_art
    _run(tab, basis, cost2, allowed)
    p = [0.0] * n
    for r in range(len(tab)):
        if basis[r] < n:
            p[basis[r]] = max(0.0, tab[r][-1])
    s = sum(p)
    p = [v / s for v in p] if s > 0 else p
    return LPResult("optimal", sum(ci * pi for ci, pi in zip(c, p)), p)
