"""Human-vs-AI agreement metrics.

Computed only from evaluations that have BOTH an AI total and a human total. With too few
pairs, no metrics are produced (only the count and an explanation) to avoid misleading numbers.
"""
import math
from typing import Sequence

import numpy as np
from scipy import stats


def _num(x):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else round(float(x), 4)


def compute_metrics(ai: Sequence[float], human: Sequence[float], tolerance: float = 0.5,
                    min_samples: int = 5) -> dict:
    pairs = [(float(a), float(h)) for a, h in zip(ai, human)
             if a is not None and h is not None and not (math.isnan(a) or math.isnan(h))]
    n = len(pairs)
    out = {"n": n, "tolerance": tolerance, "min_samples": min_samples, "warnings": []}
    if n < min_samples:
        out["available"] = False
        out["warnings"].append(
            f"Only {n} human-reviewed answer(s); at least {min_samples} are needed before metrics are shown.")
        return out
    a = np.array([p[0] for p in pairs])
    h = np.array([p[1] for p in pairs])
    err = a - h
    out.update({
        "available": True,
        "mae": _num(np.mean(np.abs(err))),
        "rmse": _num(math.sqrt(np.mean(err ** 2))),
        "mean_error_bias": _num(np.mean(err)),
        "exact_agreement": _num(np.mean(np.isclose(a, h))),
        f"agreement_within_tolerance": _num(np.mean(np.abs(err) <= tolerance + 1e-9)),
    })
    if n < 30:
        out["warnings"].append(f"Small sample (n={n}); treat these figures as indicative only.")
    if np.ptp(a) == 0 or np.ptp(h) == 0:
        out["pearson"] = out["spearman"] = None
        out["warnings"].append("Correlation is undefined because one set of marks is constant.")
    else:
        out["pearson"] = _num(stats.pearsonr(a, h)[0])
        out["spearman"] = _num(stats.spearmanr(a, h)[0])
    out["warnings"].append("Correlation measures ranking/linear association, not agreement; see MAE and exact agreement.")
    return out
