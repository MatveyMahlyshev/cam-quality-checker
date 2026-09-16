import cv2
import numpy as np
from skimage.measure import shannon_entropy


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


def sharpness(img: np.ndarray) -> float:
    gray = to_gray(img).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = gx ** 2 + gy ** 2
    var = gray.var() + 1e-6
    return float(grad.mean() / var)


def noise_estimate(img: np.ndarray, block: int = 32, min_blocks: int = 20) -> float:
    gray = to_gray(img).astype(np.float32)
    h, w = gray.shape
    stds = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            b = gray[y:y + block, x:x + block]
            stds.append(b.std())
    stds = np.array(stds)
    if len(stds) < min_blocks:
        return -1.0
    k = max(min_blocks, int(len(stds) * 0.1))
    return float(np.mean(np.sort(stds)[:k]))


def dynamic_range(img: np.ndarray) -> float:
    gray = to_gray(img)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total == 0:
        return 0.0
    hist_norm = hist / total
    ent = shannon_entropy(hist_norm, base=2)
    clipped = (hist[:5].sum() + hist[250:].sum()) / total
    return float(ent * (1.0 - clipped))


def exposure_accuracy(img: np.ndarray) -> float:
    gray = to_gray(img).astype(np.float32)
    mean = gray.mean()
    deviation = abs(mean - 128.0) / 128.0
    return float(max(0.0, 1.0 - deviation))


def compute_all(img: np.ndarray) -> dict:
    n = noise_estimate(img)
    resized, was_resized = normalize_size(img, long_side=1920)

    return {
        "sharpness": sharpness(resized),
        "noise": n,
        "dynamic_range": dynamic_range(resized),
        "exposure": exposure_accuracy(resized),
        "original_size": (img.shape[1], img.shape[0]),
        "analyzed_size": (resized.shape[1], resized.shape[0]),
        "was_resized": was_resized,
    }