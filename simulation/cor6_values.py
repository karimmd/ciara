from math import log, exp, ceil, sqrt
p, d = 0.01, 0.05
for b in (5, 10):
    n = log(1 / d) / (-log(1 - p + exp(-b)))
    print(f"(i) b={b}t: {n:.2f} -> {ceil(n)} rounds; independent nodes N=5: {ceil(n/5)} per node ({ceil(n/5)*50} epochs, {ceil(n/5)*50*30/3600:.1f} h); coupled: {ceil(n)} rounds ({ceil(n)*50} epochs, {ceil(n)*50*30/3600:.0f} h = {ceil(n)*50*30/86400:.1f} days)")
print("(i) no carryover:", log(1 / d) / (-log(1 - p)), " with e^-5 rounded to 0.0067:", log(1 / d) / (-log(1 - p + 0.0067)))
nu, pi, S, eps, N, th, L = 1 / 3, 0.3, 8, 0.5, 5, 0.02, 8.9
q = eps / S
v1 = nu * pi ** 2 * S * L / (2 * N * eps * th ** 2)
print(f"(iii) v1 (Azuma): k >= {v1:.0f}")
thp = th / (nu * pi)
for s in (0.01, 0.03, 0.05):
    n = 8 * s * s * L / thp ** 2 + 2 * L / thp
    kR1 = n / (nu * N * q)
    kC1R = n / (nu * (1 - (1 - q) ** N))
    var_part = 8 * s * s * L / thp ** 2
    lin_part = 2 * L / thp
    print(f"  s={s}: units needed n_a={n:.1f} (variance {var_part:.1f}, linear {lin_part:.1f}); k (R1 pairs) >= {kR1:.0f}; k (C1R rounds) >= {kC1R:.0f}")
for pp in (0.1, 0.2, 0.3, 0.5, 0.8):
    thp = th / (nu * pp)
    n = 8 * 0.0009 * L / thp ** 2 + 2 * L / thp
    print(f"  pi={pp}: n_a={n:.1f}  ratio to pi=0.1: {n / (8*0.0009*L/(th/(nu*0.1))**2 + 2*L/(th/(nu*0.1))):.2f} (pi^2 would give {(pp/0.1)**2:.0f}, pi would give {pp/0.1:.0f})")
print("crossover pi* = theta/(4 s^2 nu)... in units of nu pi:", th / (4 * 0.0009))
