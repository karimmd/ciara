import os
import random
import sys
import unittest

import numpy as np

from ciara_monitor import settling as ST
from ciara import band as MB

MAP = {"tail_f1": "Y_L_f1", "tail_f2": "Y_L_f2", "deadline_miss": "Y_D", "cloud_share": "Y_C"}


class Equivalence(unittest.TestCase):
    def test_same_mean_and_standard_error_as_the_settling_measurement(self):
        rng = random.Random(5)
        blocks_st, blocks_mb = [], []
        for rd in range(1, 41):
            for n in ("E1", "E2", "E3"):
                shared = rng.gauss(0, 0.2)
                cur = {s: [shared + rng.gauss(0, 0.5) + (0.3 * np.exp(-k / 4.0) if s != "cloud_share" else 0.0) for k in range(50)] for s in ST.SERIES}
                for s in cur:
                    if rng.random() < 0.1:
                        cur[s][rng.randrange(50)] = None
                blocks_st.append({"node": n, "round": rd, "arm": "a", "prev": "b", "profile_switched": True, "curves": cur})
                blocks_mb.append({"node": n, "round": rd, "arm": "a", "prev": "b", "curves": {**{o: [None] * 50 for o in MB.OUTCOMES}, **{MAP[s]: cur[s] for s in MAP}}})
        starts = ST.cluster_starts([b["round"] for b in blocks_st])
        res = MB.summarize(blocks_mb)["outcomes"]
        for s, o in MAP.items():
            w = ST.window_bias(ST.deviations(blocks_st, s), starts)
            self.assertAlmostEqual(res[o]["mean"], w["mean"], places=12)
            self.assertAlmostEqual(res[o]["se"], w["se"], places=12)
            self.assertAlmostEqual(res[o]["upper"], abs(w["mean"]) + 1.96 * w["se"], places=12)


if __name__ == "__main__":
    unittest.main()
