from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LENGTHS = (512, 2048, 4096)
CONCS = (1, 4, 8, 16)
NEW_TOKENS = 128
ADMISSIBLE_PROFILES = (
    "q35-0.8b-fp16",
    "q35-2b-fp16",
    "q35-4b-fp16",
    "q35-4b-ih-w8a8",
    "q35-4b-ih-w4a16",
)
CLOUD_CKPT = "q35-9b-ih-w4a16"
REFERENCE_CKPT = "q35-4b-fp16"
NOMINAL_BUDGET_GIB = 0.95 * 11.0
MAX_SEQ_TOKENS = 4096 + 1024


def pilot_dir() -> Path:
    return Path(os.environ.get("CIARA_PILOT_DIR", "pilot"))


def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


@dataclass(frozen=True)
class Cell:
    p95_lat: float
    ttft: float
    dec_rate: float
    agg_tok_s: float


@dataclass(frozen=True)
class FamilyProfile:
    prompt_tokens: float
    out_tokens: Optional[float]
    out_tokens_p95: Optional[float]
    accuracy: Optional[float]
    schema_valid: Optional[float] = None

    def has_workload(self) -> bool:
        return self.out_tokens is not None


@dataclass(frozen=True)
class FamilyCurve:
    conc: Tuple[float, ...]
    ttft: Tuple[float, ...]
    tpot: Tuple[float, ...]
    mean_lat: Tuple[float, ...]
    p95_lat: Tuple[float, ...]
    cycle: Tuple[float, ...]


@dataclass(frozen=True)
class LoadCurve:
    conc: Tuple[float, ...]
    loads: Tuple[float, ...]
    mean_lat: Mapping[str, Tuple[float, ...]]
    p95_lat: Mapping[str, Tuple[float, ...]]
    ttft: Mapping[str, Tuple[float, ...]]
    tpot: Mapping[str, Tuple[float, ...]]

    @property
    def max_load(self) -> float:
        return self.loads[-1]

    def at(self, load: float) -> Optional[Dict[str, Dict[str, float]]]:
        if load > self.max_load * (1.0 + 1e-12):
            return None
        out: Dict[str, Dict[str, float]] = {}
        for f in self.mean_lat:
            out[f] = {
                "mean": _interp(self.loads, self.mean_lat[f], load),
                "p95": _interp(self.loads, self.p95_lat[f], load),
                "ttft": _interp(self.loads, self.ttft[f], load),
                "tpot": _interp(self.loads, self.tpot[f], load),
            }
        return out


def build_load_curve(curves: Mapping[str, FamilyCurve], shares: Mapping[str, float]) -> LoadCurve:
    fams = [f for f in curves if shares.get(f, 0.0) > 0.0]
    if not fams:
        raise ValueError("load curve needs at least one family with a positive share")
    conc = curves[fams[0]].conc
    loads: List[float] = []
    keep: List[int] = []
    best = -1.0
    for i, c in enumerate(conc):
        cycle = sum(shares[f] * curves[f].cycle[i] for f in fams)
        x = c / cycle
        if x > best * (1.0 + 1e-12):
            loads.append(x)
            keep.append(i)
            best = x

    def take(arr: Tuple[float, ...]) -> Tuple[float, ...]:
        return tuple(arr[i] for i in keep)

    return LoadCurve(
        conc=tuple(conc[i] for i in keep),
        loads=tuple(loads),
        mean_lat={f: take(curves[f].mean_lat) for f in fams},
        p95_lat={f: take(curves[f].p95_lat) for f in fams},
        ttft={f: take(curves[f].ttft) for f in fams},
        tpot={f: take(curves[f].tpot) for f in fams},
    )


@dataclass(frozen=True)
class ProfileRow:
    ckpt: str
    site: str
    weights_gib: float
    kv_gib: float
    kv_tokens: int
    budget_gib: float
    cells: Mapping[Tuple[int, int], Cell]
    families: Mapping[str, FamilyProfile]
    max_batch: int = 16

    def kv_gib_per_token(self) -> float:
        return self.kv_gib / self.kv_tokens

    def kv_seqs(self, max_seq_tokens: int = MAX_SEQ_TOKENS) -> int:
        return int(self.kv_tokens // max_seq_tokens)

    def mem_gib(self, batch_cap: int, max_seq_tokens: int = MAX_SEQ_TOKENS) -> float:
        return (self.budget_gib - self.kv_gib) + batch_cap * max_seq_tokens * self.kv_gib_per_token()

    def admissible(self, batch_cap: int, max_seq_tokens: int = MAX_SEQ_TOKENS) -> bool:
        return self.kv_tokens >= batch_cap * max_seq_tokens

    def conc_levels(self, batch_cap: int) -> Tuple[int, ...]:
        levels = tuple(c for c in CONCS if c <= batch_cap)
        if not levels:
            raise ValueError(f"batch cap {batch_cap} is below the smallest grid concurrency")
        return levels

    def cell_at(self, length: float, conc: int) -> Cell:
        lens = [L for L in LENGTHS if (L, conc) in self.cells]
        pick = lambda attr: _interp(lens, [getattr(self.cells[(L, conc)], attr) for L in lens], length)
        return Cell(pick("p95_lat"), pick("ttft"), pick("dec_rate"), pick("agg_tok_s"))

    def interp_cell(self, length: float, conc: float) -> Cell:
        cs = sorted({c for (_, c) in self.cells})
        at = [self.cell_at(length, c) for c in cs]
        pick = lambda attr: _interp(cs, [getattr(x, attr) for x in at], conc)
        return Cell(pick("p95_lat"), pick("ttft"), pick("dec_rate"), pick("agg_tok_s"))

    def curve(self, family: str, batch_cap: int) -> FamilyCurve:
        fp = self.families.get(family)
        if fp is None or not fp.has_workload():
            raise KeyError(f"no workload profile for family {family} on {self.ckpt}")
        n_mean = fp.out_tokens
        n_p95 = fp.out_tokens_p95 if fp.out_tokens_p95 is not None else n_mean
        conc, ttft, tpot, mean, p95, cycle = [], [], [], [], [], []
        for c in self.conc_levels(batch_cap):
            cell = self.cell_at(fp.prompt_tokens, c)
            tp = 1.0 / cell.dec_rate
            m = cell.ttft + max(n_mean - 1.0, 0.0) * tp
            q = max(cell.p95_lat + (n_p95 - NEW_TOKENS) * tp, cell.ttft)
            cyc = max(c * NEW_TOKENS / cell.agg_tok_s + (n_mean - NEW_TOKENS) * tp, m)
            conc.append(float(c))
            ttft.append(cell.ttft)
            tpot.append(tp)
            mean.append(m)
            p95.append(q)
            cycle.append(cyc)
        return FamilyCurve(tuple(conc), tuple(ttft), tuple(tpot), tuple(mean), tuple(p95), tuple(cycle))

    def has_family(self, family: str) -> bool:
        fp = self.families.get(family)
        return fp is not None and fp.has_workload() and fp.accuracy is not None


@dataclass(frozen=True)
class OverheadTable:
    default_s: float = 5.0
    pairs: Mapping[str, float] = field(default_factory=dict)

    def switch_time_s(self, src: str, dst: str) -> float:
        if src == dst:
            return 0.0
        return float(self.pairs.get(f"{src}>{dst}", self.default_s))


def switch_overhead(profiles: Sequence[str] = ADMISSIBLE_PROFILES, restart_only: str = REFERENCE_CKPT) -> OverheadTable:
    pairs = {}
    for a in profiles:
        for b in profiles:
            if a == b:
                continue
            if b == restart_only:
                pairs[f"{a}>{b}"] = 50.0
            elif a == restart_only:
                pairs[f"{a}>{b}"] = 98.0
    return OverheadTable(default_s=1.76, pairs=pairs)


@dataclass
class ProfileTable:
    edge: Dict[str, ProfileRow]
    cloud: Optional[ProfileRow]
    reference: str = REFERENCE_CKPT
    overhead: OverheadTable = field(default_factory=OverheadTable)
    meta: Dict[str, Any] = field(default_factory=dict)

    def reference_accuracy(self, family: str) -> Optional[float]:
        row = self.edge.get(self.reference)
        if row is None or family not in row.families:
            return None
        return row.families[family].accuracy

    def to_dict(self) -> Dict[str, Any]:
        def row_d(r: ProfileRow) -> Dict[str, Any]:
            return {
                "ckpt": r.ckpt,
                "site": r.site,
                "weights_gib": r.weights_gib,
                "kv_gib": r.kv_gib,
                "kv_tokens": r.kv_tokens,
                "budget_gib": r.budget_gib,
                "max_batch": r.max_batch,
                "cells": {f"L{L}_c{c}": vars(cell) for (L, c), cell in sorted(r.cells.items())},
                "families": {f: vars(fp) for f, fp in r.families.items()},
            }

        return {
            "version": 1,
            "reference": self.reference,
            "overhead": {"default_s": self.overhead.default_s, "pairs": dict(self.overhead.pairs)},
            "edge": {k: row_d(r) for k, r in self.edge.items()},
            "cloud": row_d(self.cloud) if self.cloud else None,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ProfileTable":
        def row_f(x: Mapping[str, Any]) -> ProfileRow:
            cells = {}
            for key, v in x["cells"].items():
                L, c = key[1:].split("_c")
                cells[(int(L), int(c))] = Cell(**v)
            fams = {f: FamilyProfile(**fp) for f, fp in x["families"].items()}
            return ProfileRow(
                ckpt=x["ckpt"], site=x["site"], weights_gib=x["weights_gib"], kv_gib=x["kv_gib"],
                kv_tokens=int(x["kv_tokens"]), budget_gib=x["budget_gib"], cells=cells, families=fams,
                max_batch=int(x.get("max_batch", 16)),
            )

        oh = d.get("overhead", {})
        return ProfileTable(
            edge={k: row_f(r) for k, r in d["edge"].items()},
            cloud=row_f(d["cloud"]) if d.get("cloud") else None,
            reference=d.get("reference", REFERENCE_CKPT),
            overhead=OverheadTable(float(oh.get("default_s", 5.0)), dict(oh.get("pairs", {}))),
            meta=dict(d.get("meta", {})),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True))

    @staticmethod
    def load(path: Path) -> "ProfileTable":
        return ProfileTable.from_dict(json.loads(Path(path).read_text()))

    def flat_rows(self, batch_caps: Iterable[int] = (8, 16)) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for ckpt, r in self.edge.items():
            for b in batch_caps:
                rec: Dict[str, Any] = {"ckpt": ckpt, "batch_cap": b, "admissible": r.admissible(b),
                                       "mem_gib": round(r.mem_gib(b), 3), "kv_seqs": r.kv_seqs()}
                for f in ("f1", "f2"):
                    fp = r.families.get(f)
                    rec[f"{f}_acc"] = None if fp is None else fp.accuracy
                    if fp is not None and fp.has_workload():
                        cv = r.curve(f, b)
                        rec[f"{f}_mean_lat_s"] = [round(x, 3) for x in cv.mean_lat]
                        rec[f"{f}_p95_lat_s"] = [round(x, 3) for x in cv.p95_lat]
                rows.append(rec)
        return rows


def describe_columns() -> Dict[str, str]:
    return {
        "ckpt": "Checkpoint key of the engine (for example q35-4b-ih-w8a8).",
        "site": "edge (one whole 11 GB card) or cloud (tensor parallel over two cards).",
        "weights_gib": "Resident weight memory of the engine.",
        "kv_gib, kv_tokens": "KV pool of the engine at gpu_memory_utilization 0.95 and its capacity in tokens.",
        "budget_gib": "Engine memory budget the pool was sized for; nominal 0.95 * 11 GiB.",
        "cells[(L,c)]": "Serving grid, replicate means, 128 output tokens: p95_lat (s), ttft (s), dec_rate "
                        "(tokens/s per request), agg_tok_s (tokens/s over the c requests). Cells are synchronized "
                        "bursts of c requests, not open-loop arrivals.",
        "families[f].prompt_tokens": "Mean prompt tokens of the family workload (family 1 summary for f1; character-based "
                                     "estimate for f2).",
        "families[f].out_tokens": "Mean output tokens of the family on this checkpoint (family 1 summary; calibration "
                                  "table for f2).",
        "families[f].out_tokens_p95": "p95 output tokens (family 1 summary; the output cap for the 4B builds on f2).",
        "families[f].accuracy": "Headline quality metric of the family: task_acc on the family 1 suite (f1), answer "
                                "accuracy on the family 2 calibration split (f2).",
        "mem_gib(b)": "(budget_gib - kv_gib) + b * 5,120 * kv_gib/kv_tokens: weights and engine overhead plus the KV "
                      "pool for b full-length sequences.",
        "curve(f, b)": "ttft, tpot, mean_lat, p95_lat, cycle at grid concurrencies c <= b, derived from the cells at "
                       "the family prompt and output lengths (module docstring).",
        "overhead": "Seconds of paused admission for a profile switch (the sleep and wake test of the edge stack).",
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _cells_from_p25(entry: Mapping[str, Any]) -> Dict[Tuple[int, int], Cell]:
    cells: Dict[Tuple[int, int], Cell] = {}
    for key, v in entry["cells"].items():
        L, c = key[1:].split("_c")
        cells[(int(L), int(c))] = Cell(
            p95_lat=float(v["p95_lat"]), ttft=float(v["ttft"]), dec_rate=float(v["dec_rate"]),
            agg_tok_s=float(v["agg_tok_s"]),
        )
    return cells


def _f1_profile(p3_summary: Mapping[str, Any], metric: str) -> FamilyProfile:
    iot = p3_summary["tasks"]["iot"]
    return FamilyProfile(
        prompt_tokens=float(iot["tokens"]["mean_prompt_tokens"]),
        out_tokens=float(iot["tokens"]["mean_output_tokens"]),
        out_tokens_p95=float(iot["tokens"]["p95_output_tokens"]),
        accuracy=float(iot["metrics"][metric]["mean"]),
        schema_valid=float(iot["metrics"]["schema_valid"]["mean"]),
    )


def build_table(
    pilot: Optional[Path] = None,
    klass: str = "edge",
    ckpts: Sequence[str] = ADMISSIBLE_PROFILES,
    cloud_ckpt: str = CLOUD_CKPT,
    reference: str = REFERENCE_CKPT,
    accuracy_metric: str = "task_acc",
    family2_path: Optional[Path] = None,
    overhead: Optional[OverheadTable] = None,
) -> ProfileTable:
    pilot = Path(pilot) if pilot else pilot_dir()
    p25_path = pilot / "p25" / "results" / "p25-summary.json"
    p25 = json.loads(p25_path.read_text())
    if family2_path is None:
        family2_path = Path(__file__).resolve().parent / "data" / "family2_calibration.json"
    f2: Dict[str, Any] = {}
    f2_prompt = None
    if Path(family2_path).exists():
        f2d = json.loads(Path(family2_path).read_text())
        f2 = f2d["profiles"]
        f2_prompt = float(f2d["prompt_tokens_mean"])

    sources = {"p25-summary.json": _sha256(p25_path)}

    def p3_summary(ckpt: str) -> Mapping[str, Any]:
        p = pilot / "p3" / "results" / "p3" / ckpt / "summary.json"
        sources[f"p3/{ckpt}/summary.json"] = _sha256(p)
        return json.loads(p.read_text())

    def make_row(ckpt: str, site: str, key: str, max_batch: int) -> ProfileRow:
        e = p25[key]
        if not e.get("admissible"):
            raise ValueError(f"{key} is not admissible in the serving-grid summary")
        fams: Dict[str, FamilyProfile] = {"f1": _f1_profile(p3_summary(ckpt), accuracy_metric)}
        if ckpt in f2 and f2_prompt is not None:
            x = f2[ckpt]
            fams["f2"] = FamilyProfile(
                prompt_tokens=f2_prompt,
                out_tokens=float(x["mean_output_tokens"]),
                out_tokens_p95=None if x.get("p95_output_tokens") is None else float(x["p95_output_tokens"]),
                accuracy=float(x["answer_acc"]),
            )
        return ProfileRow(
            ckpt=ckpt, site=site, weights_gib=float(e["weights_gib"]), kv_gib=float(e["kv_gib"]),
            kv_tokens=int(e["kv_tokens"]), budget_gib=NOMINAL_BUDGET_GIB if site == "edge" else float("nan"),
            cells=_cells_from_p25(e), families=fams, max_batch=max_batch,
        )

    edge = {c: make_row(c, "edge", f"{c}:{klass}", 16) for c in ckpts}
    cloud = make_row(cloud_ckpt, "cloud", f"{cloud_ckpt}:cloud", 16)
    cloud = ProfileRow(**{**vars(cloud), "budget_gib": cloud.weights_gib + cloud.kv_gib})
    return ProfileTable(
        edge=edge, cloud=cloud, reference=reference, overhead=overhead or switch_overhead(ckpts),
        meta={
            "klass": klass, "accuracy_metric": accuracy_metric, "nominal_budget_gib": NOMINAL_BUDGET_GIB,
            "max_seq_tokens": MAX_SEQ_TOKENS, "sources_sha256": sources,
            "family2_file": str(family2_path) if f2 else None,
            "note": "Serving-grid cells are synchronized bursts of c requests with 128 output tokens; see module docstring.",
        },
    )


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Build or print the edge profile table.")
    ap.add_argument("--pilot", default=None, help="folder of the profiling outputs (default: CIARA_PILOT_DIR, else ./pilot)")
    ap.add_argument("--klass", default="edge", help="class label that suffixes the checkpoint keys of the serving-grid summary")
    ap.add_argument("--out", default=None, help="write the table as JSON")
    ap.add_argument("--metric", default="task_acc", choices=["task_acc", "schema_valid"])
    args = ap.parse_args(argv)
    tbl = build_table(Path(args.pilot) if args.pilot else None, klass=args.klass, accuracy_metric=args.metric)
    if args.out:
        tbl.save(Path(args.out))
    for r in tbl.flat_rows():
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
