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
    cn = chroma_noise(img)
    resized, was_resized = normalize_size(img, long_side=1920)

    return {
        "sharpness": sharpness(resized),
        "noise": n,
        "dynamic_range": dynamic_range(resized),
        "exposure": exposure_accuracy(resized),
        "contrast": contrast_rms(resized),
        "colorfulness": colorfulness(resized),
        "chroma_noise": cn,
        "highlight_rec": highlight_recovery(resized),
        "shadow_detail": shadow_detail(resized),
        "original_size": (img.shape[1], img.shape[0]),
        "analyzed_size": (resized.shape[1], resized.shape[0]),
        "was_resized": was_resized,
    }


def contrast_rms(img: np.ndarray) -> float:
    gray = to_gray(img).astype(np.float32)
    return float(gray.std() / 255.0)


def colorfulness(img: np.ndarray) -> float:
    if len(img.shape) == 2:
        return 0.0
    b, g, r = cv2.split(img.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    return float(np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))


def chroma_noise(img: np.ndarray, block: int = 32) -> float:
    if len(img.shape) == 2:
        return 0.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[:, :, 1], lab[:, :, 2]
    h, w = a.shape
    stds = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            ba = a[y:y + block, x:x + block]
            bb = b[y:y + block, x:x + block]
            stds.append(np.sqrt(ba.var() + bb.var()))
    stds = np.array(stds)
    if len(stds) < 20:
        return -1.0
    k = max(20, int(len(stds) * 0.1))
    return float(np.mean(np.sort(stds)[:k]))


def highlight_recovery(img: np.ndarray) -> float:
    gray = to_gray(img).astype(np.float32)
    threshold = np.percentile(gray, 95)
    bright = gray >= threshold
    if bright.sum() < 100:
        return 0.5

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    texture = float(lap[bright].std())
    baseline = float(lap.std()) + 1e-6
    ratio = texture / baseline
    return float(np.clip(ratio, 0.0, 1.5) / 1.5)


def shadow_detail(img: np.ndarray) -> float:
    gray = to_gray(img).astype(np.float32)
    threshold = np.percentile(gray, 5)
    dark = gray <= threshold
    if dark.sum() < 100:
        return 0.5

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    texture = float(lap[dark].std())
    baseline = float(lap.std()) + 1e-6
    ratio = texture / baseline
    return float(np.clip(ratio, 0.0, 1.5) / 1.5)