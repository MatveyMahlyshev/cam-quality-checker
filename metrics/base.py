from abc import ABC, abstractmethod
import numpy as np


class BaseMetric(ABC):

    name: str = ""

    @abstractmethod
    def compute(self, img: np.ndarray) -> float:
        ...

    def __call__(self, img: np.ndarray) -> float:
        return self.compute(img)