import csv, os, sys

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "estimators.csv")
cols = ["dm_bias_quality_inc", "dm_bias_quality_non", "dm_bias_exceed_inc", "dm_bias_exceed_non"]
print(f"{'design':24s} {'t_mix':>5s} {'l':>4s} {'b':>4s}  max |bias|")
for r in csv.DictReader(open(path)):
    print(f"{r['design']:24s} {float(r['t_mix']):5.0f} {int(float(r['l'])):4d} {int(float(r['b'])):4d}  {max(abs(float(r[c])) for c in cols):.5f}")
