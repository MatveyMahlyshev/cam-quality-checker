import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_size(img: np.ndarray, long_side: int = 1920):
    h, w = img.shape[:2]
    cur = max(h, w)
    if cur <= long_side:
        return img, False
    scale = long_side / cur
    new_size = (int(round(w * scale)), int(round(h * scale)))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return resized, True