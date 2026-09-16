from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ValidationResult:
    ok: bool
    message: str = "OK"
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


class BaseValidator(ABC):

    @abstractmethod
    def _check(self, img: np.ndarray, filename: str) -> ValidationResult:
        ...

    def validate(self, img: np.ndarray, filename: str = "") -> ValidationResult:
        if img is None or img.size == 0:
            return ValidationResult(False, "Пустое или некорректное изображение")
        return self._check(img, filename)