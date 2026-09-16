import numpy as np

from .base import BaseRatingStrategy
from .criteria import CRITERIA, DEFAULT_WEIGHTS, REFERENCE


def normalize_minmax(values, invert=False):
    arr = np.array(values, dtype=float)
    if arr.max() - arr.min() < 1e-9:
        return [50.0] * len(arr)
    norm = (arr - arr.min()) / (arr.max() - arr.min()) * 100
    return (100 - norm if invert else norm).tolist()


def normalize_reference(values, ref, higher_better=True):
    out = []
    for v in values:
        if ref <= 0:
            out.append(50.0)
            continue
        ratio = v / ref if higher_better else ref / max(v, 1e-6)
        out.append(float(np.clip(ratio * 100, 0, 150)))
    return out


class WeightedSumStrategy(BaseRatingStrategy):

    def __init__(self, use_reference: bool = False):
        self.use_reference = use_reference

    def score(self, results, weights=None):
        weights = weights or DEFAULT_WEIGHTS
        if not results:
            return []

        norm = {}
        for key, higher_better in CRITERIA.items():
            vals = [r[key] for r in results]
            if self.use_reference:
                norm[key] = normalize_reference(vals, REFERENCE[key], higher_better)
            else:
                norm[key] = normalize_minmax(vals, invert=not higher_better)

        out = []
        for i, r in enumerate(results):
            score = sum(weights[k] * norm[k][i] for k in CRITERIA)
            out.append({**r, "score": round(score, 2)})
        return out