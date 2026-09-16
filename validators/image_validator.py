import cv2
import numpy as np

from metrics import to_gray, noise_estimate
from .base import BaseValidator, ValidationResult


class DefaultImageValidator(BaseValidator):

    MIN_SHORT_SIDE = 480
    RECOMMENDED_LONG_SIDE = 1920
    MAX_CLIP_RATIO = 0.5
    WARN_CLIP_RATIO = 0.3
    MIN_LAPLACIAN_VAR = 10.0

    def __init__(
        self,
        min_short_side: int | None = None,
        recommended_long_side: int | None = None,
        max_clip_ratio: float | None = None,
        warn_clip_ratio: float | None = None,
        min_laplacian_var: float | None = None,
    ):
        self.min_short_side = min_short_side or self.MIN_SHORT_SIDE
        self.recommended_long_side = recommended_long_side or self.RECOMMENDED_LONG_SIDE
        self.max_clip_ratio = max_clip_ratio or self.MAX_CLIP_RATIO
        self.warn_clip_ratio = warn_clip_ratio or self.WARN_CLIP_RATIO
        self.min_laplacian_var = min_laplacian_var or self.MIN_LAPLACIAN_VAR

    def _check_size(self, h: int, w: int) -> ValidationResult | None:
        if min(h, w) < self.min_short_side:
            return ValidationResult(
                False,
                f"Слишком маленькое разрешение: {w}×{h}. "
                f"Минимум {self.min_short_side}px по короткой стороне.",
            )
        return None

    def _check_clipping(self, gray: np.ndarray) -> tuple[ValidationResult | None, float]:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        total = hist.sum()
        if total == 0:
            return ValidationResult(False, "Пустое изображение"), 0.0
        clipped = (hist[:5].sum() + hist[250:].sum()) / total
        if clipped > self.max_clip_ratio:
            return ValidationResult(
                False,
                f"Слишком много клиппинга ({clipped:.0%}). "
                "Сцена не подходит — выберите кадр с меньшим контрастом.",
            ), clipped
        return None, clipped

    def _check_texture(self, gray: np.ndarray) -> ValidationResult | None:
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < self.min_laplacian_var:
            return ValidationResult(
                False,
                "Слишком однородная сцена — нет текстур. "
                "Снимите что-то с деталями: листва, кирпич, текст.",
            )
        return None

    def _check_noise_estimable(self, img: np.ndarray) -> ValidationResult | None:
        if noise_estimate(img) < 0:
            return ValidationResult(
                False,
                "Недостаточно однородных областей для оценки шума. "
                "В кадре нужны и текстуры, и ровные участки (стена, небо).",
            )
        return None

    def _check(self, img: np.ndarray, filename: str) -> ValidationResult:
        h, w = img.shape[:2]

        if (res := self._check_size(h, w)) is not None:
            return res

        gray = to_gray(img)

        clip_res, clipped = self._check_clipping(gray)
        if clip_res is not None:
            return clip_res

        if (res := self._check_texture(gray)) is not None:
            return res

        if (res := self._check_noise_estimable(img)) is not None:
            return res

        warnings: list[str] = []

        if max(h, w) < self.recommended_long_side:
            warnings.append(
                f"Разрешение {w}×{h} ниже рекомендуемого "
                f"({self.recommended_long_side}px по длинной стороне). "
                "Оценка резкости может быть менее точной."
            )

        if clipped > self.warn_clip_ratio:
            warnings.append(
                f"Много клиппинга ({clipped:.0%}) — динамический диапазон занижен."
            )

        if filename.lower().endswith((".heic", ".heif")):
            warnings.append(
                "HEIC часто уже пересжат — лучше загрузить JPEG/PNG оригинал."
            )

        return ValidationResult(True, "OK", warnings)