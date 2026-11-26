#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${1:-${PYTHON:-python3}}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$ROOT/out/figures" "$ROOT/out/tables" "$ROOT/out/derived"

"$PY" "$HERE/build_figdata.py"
for f in e4_timeseries latency_quantiles cascade_edge_share e1e5_switching e1e5_latency_parts; do
  cmp -s "$ROOT/out/derived/$f.csv" "$ROOT/data/$f.csv" && echo "derived $f.csv: identical to data/$f.csv" || { echo "derived $f.csv: DIFFERS from data/$f.csv"; exit 1; }
done

for g in cert_trajectory cert_heldout e4_trace comparison latency_ccdf validity val_sim load_context operating_curve energy_split energy_device_cmp \
         settling class_cdf supp_extra; do
  "$PY" "$HERE/make_$g.py"
done

for t in tab_methods_heat tab_certificate tab7_methods tab_classc tab_level_a tab_overhead; do
  "$PY" "$HERE/make_$t.py"
done
echo "make_all: figures in $ROOT/out/figures, tables in $ROOT/out/tables"
