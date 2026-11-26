from __future__ import annotations

from pathlib import Path

from ciara_baselines.profile import CONCS, LENGTHS, NEW_TOKENS, Cell, FamilyProfile, OverheadTable, ProfileRow, ProfileTable

BUDGET = 10.45
SHIPPED_TABLE = Path(__file__).resolve().parents[2] / "baselines" / "profiles" / "edge_profile_table.json"


def make_row(ckpt, site="edge", t0=1.0, d=50.0, acc_f1=0.80, acc_f2=0.50, kv_tokens=100000, kv_gib=1.5,
             weights=8.0, f1=(2000.0, 31.0, 60.0), f2=(1000.0, 400.0, 1000.0), max_batch=16, budget=BUDGET):
    cells = {}
    for L in LENGTHS:
        for c in CONCS:
            ttft = t0 * L / 1000.0
            p95 = ttft + (NEW_TOKENS - 1) / d
            cells[(L, c)] = Cell(p95_lat=p95, ttft=ttft, dec_rate=d, agg_tok_s=c * NEW_TOKENS / p95)
    fams = {}
    if f1 is not None:
        fams["f1"] = FamilyProfile(f1[0], f1[1], f1[2], acc_f1)
    if f2 is not None:
        fams["f2"] = FamilyProfile(f2[0], f2[1], f2[2], acc_f2)
    return ProfileRow(ckpt=ckpt, site=site, weights_gib=weights, kv_gib=kv_gib, kv_tokens=kv_tokens,
                      budget_gib=budget, cells=cells, families=fams, max_batch=max_batch)


def mean_lat(row, fam, c=1):
    fp = row.families[fam]
    cell = row.cell_at(fp.prompt_tokens, c)
    return cell.ttft + (fp.out_tokens - 1.0) / cell.dec_rate


def p95_lat(row, fam, c=1):
    fp = row.families[fam]
    cell = row.cell_at(fp.prompt_tokens, c)
    return cell.ttft + (fp.out_tokens_p95 - 1.0) / cell.dec_rate


def basic_table(overhead_s=5.0):
    big = make_row("big-fp16", t0=1.0, d=50.0, acc_f1=0.80, acc_f2=0.50, kv_tokens=40960, kv_gib=1.5)
    mid = make_row("mid-int8", t0=0.5, d=100.0, acc_f1=0.78, acc_f2=0.49, f2=(1000.0, 420.0, 1000.0))
    tiny = make_row("tiny-fp16", t0=0.1, d=300.0, acc_f1=0.30, acc_f2=0.25, f2=(1000.0, 12.0, 12.0))
    cloud = make_row("cloud-int4", site="cloud", t0=0.2, d=200.0, acc_f1=0.70, acc_f2=0.60,
                     f2=(1000.0, 10.0, 10.0), kv_tokens=200000, kv_gib=4.0, weights=4.8, budget=8.8)
    return ProfileTable(
        edge={r.ckpt: r for r in (big, mid, tiny)}, cloud=cloud, reference="big-fp16",
        overhead=OverheadTable(default_s=overhead_s),
    )


def guard_table(overhead_s=20.0):
    ref = make_row("ref", t0=0.6, d=30.0, acc_f1=0.80, acc_f2=0.50, kv_tokens=100000, kv_gib=1.5)
    slow = make_row("slow", t0=0.4, d=40.0, acc_f1=0.80, acc_f2=0.50, kv_tokens=100000, kv_gib=1.5)
    fast = make_row("fast", t0=0.2, d=100.0, acc_f1=0.80, acc_f2=0.50, kv_tokens=1000000, kv_gib=0.5)
    return ProfileTable(
        edge={r.ckpt: r for r in (ref, slow, fast)}, cloud=None, reference="ref",
        overhead=OverheadTable(default_s=overhead_s),
    )
