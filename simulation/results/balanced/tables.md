# Balanced allocation and one-sided bounds: simulation tables

**Campaign composition (2,1,1), N = 4:** 36 configurations (4 scenarios x 3 t_mix x 3 coupling levels), 144000 replicated campaigns in all.

**Verdict inputs (zeta true, decision rule of the campaign, looks day 4, 7, 10):**

* Interval coverage, one-sided events of the certificate and the gate, any look: 0 misses in 144000 replicates; largest per-configuration Clopper-Pearson 95% upper bound 0.0007 (criterion 0.05).
* False certification (campaign design, zeta true): 0 in 144000 campaigns; largest per-configuration upper bound over all designs 0.0007 (criterion 0.05).
* With the Stage 1 estimate of zeta (E): 0 false certifications; with zeta = 0 (Z): 0 (reference only; the coupling slack is part of the decision rule).
* Probe composition (1,1,1,1), four arms, interval coverage: 0 misses in 36000 replicates, largest upper bound 0.0007.

**Within-round dependence stress (balanced rounds, target 1/2, outcome-level events, all looks):**

| composition | dependence | replicates | misses, one-sided L | misses, two-sided L |
|---|---|---|---|---|
| 1-1-1-1 | common | 8000 | 2 (0.0003) | 0 (0.0000) |
| 1-1-1-1 | half | 8000 | 23 (0.0029) | 14 (0.0018) |
| 1-1-1-1 | indep | 8000 | 30 (0.0037) | 22 (0.0027) |
| 2-1-1 | common | 8000 | 1 (0.0001) | 1 (0.0001) |
| 2-1-1 | half | 8000 | 19 (0.0024) | 12 (0.0015) |
| 2-1-1 | indep | 8000 | 28 (0.0035) | 19 (0.0024) |

**Width (K3X1, day 7, cost outcomes, all arms and levels):** median radius ratio balanced one-sided to independent draws (two-sided) 0.525 (range 0.486 to 0.918); balanced two-sided to independent draws 0.569.

* arm 0: median ratio 0.907
* arm 1: median ratio 0.501
* arm 2: median ratio 0.517
