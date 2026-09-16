from .base import BaseRatingStrategy
from .criteria import CRITERIA, DEFAULT_WEIGHTS, REFERENCE
from .weighted import WeightedSumStrategy
from .topsis import TopsisStrategy


def build_rating(results, weights=None, method="weighted", use_reference=False):
    if method == "topsis":
        strategy = TopsisStrategy()
    else:
        strategy = WeightedSumStrategy(use_reference=use_reference)
    return strategy.rank(results, weights)


def pairwise_diff(rated, key="score"):
    rows = []
    for i in range(len(rated)):
        for j in range(i + 1, len(rated)):
            a, b = rated[i], rated[j]
            diff = {
                "A": a["camera"],
                "B": b["camera"],
                "Δ score": round(a[key] - b[key], 2),
            }
            for k in CRITERIA:
                diff[f"Δ {k}"] = round(a[k] - b[k], 4)
            rows.append(diff)
    return rows


__all__ = [
    "BaseRatingStrategy",
    "WeightedSumStrategy",
    "TopsisStrategy",
    "CRITERIA",
    "DEFAULT_WEIGHTS",
    "REFERENCE",
    "build_rating",
    "pairwise_diff",
]