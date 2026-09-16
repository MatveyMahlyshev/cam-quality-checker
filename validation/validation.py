import cv2
import numpy as np
from metrics.metrics import to_gray, noise_estimate


MIN_SHORT_SIDE = 480
RECOMMENDED_LONG_SIDE = 1920
MAX_CLIP_RATIO = 0.5
WARN_CLIP_RATIO = 0.3
MIN_LAPLACIAN_VAR = 10.0


def validate_image(img: np.ndarray, filename: str = "") -> tuple[bool, str, list[str]]:
    h, w = img.shape[:2]
    warnings: list[str] = []

    if min(h, w) < MIN_SHORT_SIDE:
        return False, (
            f"Слишком маленькое разрешение: {w}×{h}. "
            f"Минимум {MIN_SHORT_SIDE}px по короткой стороне."
        ), []

    gray = to_gray(img)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total == 0:
        return False, "Пустое изображение", []
    clipped = (hist[:5].sum() + hist[250:].sum()) / total
    if clipped > MAX_CLIP_RATIO:
        return False, (
            f"Слишком много клиппинга ({clipped:.0%}). "
            "Сцена не подходит — выберите кадр с меньшим контрастом."
        ), []

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < MIN_LAPLACIAN_VAR:
        return False, (
            "Слишком однородная сцена — нет текстур. "
            "Снимите что-то с деталями: листва, кирпич, текст."
        ), []

    if noise_estimate(img) < 0:
        return False, (
            "Недостаточно однородных областей для оценки шума. "
            "В кадре нужны и текстуры, и ровные участки (стена, небо)."
        ), []

    if max(h, w) < RECOMMENDED_LONG_SIDE:
        warnings.append(
            f"Разрешение {w}×{h} ниже рекомендуемого "
            f"({RECOMMENDED_LONG_SIDE}px по длинной стороне). "
            "Оценка резкости может быть менее точной."
        )

    if clipped > WARN_CLIP_RATIO:
        warnings.append(
            f"Много клиппинга ({clipped:.0%}) — динамический диапазон занижен."
        )

    if filename.lower().endswith((".heic", ".heif")):
        warnings.append(
            "HEIC часто уже пересжат — лучше загрузить JPEG/PNG оригинал."
        )

    return True, "OK", warnings