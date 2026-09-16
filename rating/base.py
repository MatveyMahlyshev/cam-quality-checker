from abc import ABC, abstractmethod


class BaseRatingStrategy(ABC):

    @abstractmethod
    def score(self, results: list[dict], weights: dict | None = None) -> list[dict]:
        ...

    def rank(self, results: list[dict], weights: dict | None = None) -> list[dict]:
        scored = self.score(results, weights)
        scored.sort(key=lambda x: x["score"], reverse=True)
        for rank, item in enumerate(scored, 1):
            item["rank"] = rank
        return scored