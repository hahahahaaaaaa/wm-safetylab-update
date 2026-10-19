"""Statistics of a red-team report: Wilson intervals and paired bootstrap differences.

An arm is compared with the baseline only on the (environment, seed) pairs both
have finished, so the same initial layout and the same policy randomness are on
both sides of each difference.
"""
from __future__ import annotations

import math
import random

Z95 = 1.959963984540054
RESAMPLES = 2000


def wilson(successes, n, z=Z95):
    if n == 0:
        return None, None, None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def paired_bootstrap(differences, resamples=RESAMPLES, seed=0):
    """Mean of paired differences with a percentile 95% interval (fixed seed: reproducible)."""
    n = len(differences)
    if n == 0:
        return None
    mean = sum(differences) / n
    rng = random.Random(seed)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return mean, means[int(0.025 * resamples)], means[min(resamples - 1, int(0.975 * resamples))]


METHOD = ("成功率：Wilson 95% 区间。与基线之差：只用两组都已完成的同一（任务，种子）配对，"
          f"配对差的均值与 bootstrap 95% 百分位区间（{RESAMPLES} 次重采样，固定随机种子）。"
          "危险步数与接管次数为每 episode 的步数。含 oracle 防御的组是上界。")
