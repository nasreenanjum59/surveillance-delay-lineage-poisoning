"""
Uncertainty in the classifier recall inputs
===========================================

The surveillance model treats the clean and degraded P.1 recall values as fixed.
They are estimates from 95 P.1 test sequences, so they carry sampling error.
This script propagates that error through the analytical expressions:

    delay = ln(r_clean / r_poisoned) / g        (days)
    size ratio = r_clean / r_poisoned           (dimensionless)

Recall counts: clean 88 of 95 (0.926), degraded 17 of 95 (0.179).
Each recall is given a Beta posterior with a Jeffreys prior, Beta(k + 0.5, n - k + 0.5).

    python recall_uncertainty.py

Author: Nasreen Anjum
"""
import math

import numpy as np
from scipy.stats import beta

N = 95
CLEAN_K, POIS_K = 88, 17
# The manuscript reports point values from the rounded recalls, so the point
# values printed below use the same rounded figures to match it exactly.
R_CLEAN_REPORTED, R_POIS_REPORTED = 0.926, 0.179
DRAWS = 200_000
SEED = 1

rng = np.random.default_rng(SEED)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


for label, k in [("clean", CLEAN_K), ("degraded", POIS_K)]:
    lo, hi = wilson(k, N)
    print(f"{label:9} recall {k}/{N} = {k / N:.3f}, 95% Wilson interval {lo:.3f} to {hi:.3f}")

rc = beta.rvs(CLEAN_K + 0.5, N - CLEAN_K + 0.5, size=DRAWS, random_state=rng)
rp = beta.rvs(POIS_K + 0.5, N - POIS_K + 0.5, size=DRAWS, random_state=rng)

print("\nPropagated through the analytical expressions "
      f"({DRAWS:,} draws, seed {SEED}):")
print(f"{'quantity':36} {'point value':>12} {'median':>8} {'95% interval':>18}")
for name, g in [("delay, reference growth 0.231 /d", 0.231),
                ("delay, BQ.1 growth 0.0513 /d", 0.0513)]:
    point = math.log(R_CLEAN_REPORTED / R_POIS_REPORTED) / g
    d = np.log(rc / rp) / g
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{name:36} {point:12.2f} {np.median(d):8.1f} {lo:8.1f} to {hi:5.1f}")
ratio = rc / rp
lo, hi = np.percentile(ratio, [2.5, 97.5])
print(f"{'size ratio':36} {(R_CLEAN_REPORTED / R_POIS_REPORTED):12.2f} {np.median(ratio):8.2f} {lo:8.2f} to {hi:5.2f}")
