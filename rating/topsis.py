import numpy as np

from .base import BaseRatingStrategy
from .criteria import CRITERIA, DEFAULT_WEIGHTS


class TopsisStrategy(BaseRatingStrategy):

    def score(self, results, weights=None):
        weights = weights or DEFAULT_WEIGHTS
        if not results:
            return []

        keys = list(CRITERIA.keys())
        X = np.array([[r[k] for k in keys] for r in results], dtype=float)

        norm = np.sqrt((X ** 2).sum(axis=0))
        norm[norm == 0] = 1e-9
        R = X / norm

        w = np.array([weights[k] for k in keys])
        V = R * w

        ideal, anti = [], []
        for j, k in enumerate(keys):
            if CRITERIA[k]:
                ideal.append(V[:, j].max())
                anti.append(V[:, j].min())
            else:
                ideal.append(V[:, j].min())
                anti.append(V[:, j].max())
        ideal, anti = np.array(ideal), np.array(anti)

        d_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
        d_neg = np.sqrt(((V - anti) ** 2).sum(axis=1))

        closeness = d_neg / (d_pos + d_neg + 1e-9)
        scores = (closeness * 100).round(2)

        return [{**r, "score": float(s)} for r, s in zip(results, scores)]