import cv2
import numpy as np
from skimage.measure import shannon_entropy

from .base import BaseMetric
from .utils import to_gray


class SharpnessMetric(BaseMetric):

    name = "sharpness"

    def compute(self, img: np.ndarray) -> float:
        gray = to_gray(img).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = gx ** 2 + gy ** 2
        var = gray.var() + 1e-6
        return float(grad.mean() / var)


class NoiseMetric(BaseMetric):

    name = "noise"

    def __init__(self, block: int = 32, min_blocks: int = 20):
        self.block = block
        self.min_blocks = min_blocks

    def compute(self, img: np.ndarray) -> float:
        gray = to_gray(img).astype(np.float32)
        h, w = gray.shape
        stds = []
        for y in range(0, h - self.block, self.block):
            for x in range(0, w - self.block, self.block):
                b = gray[y:y + self.block, x:x + self.block]
                stds.append(b.std())
        stds = np.array(stds)
        if len(stds) < self.min_blocks:
            return -1.0
        k = max(self.min_blocks, int(len(stds) * 0.1))
        return float(np.mean(np.sort(stds)[:k]))


class DynamicRangeMetric(BaseMetric):

    name = "dynamic_range"

    def compute(self, img: np.ndarray) -> float:
        gray = to_gray(img)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        total = hist.sum()
        if total == 0:
            return 0.0
        hist_norm = hist / total
        ent = shannon_entropy(hist_norm, base=2)
        clipped = (hist[:5].sum() + hist[250:].sum()) / total
        return float(ent * (1.0 - clipped))


class ExposureMetric(BaseMetric):

    name = "exposure"

    def compute(self, img: np.ndarray) -> float:
        gray = to_gray(img).astype(np.float32)
        mean = gray.mean()
        deviation = abs(mean - 128.0) / 128.0
        return float(max(0.0, 1.0 - deviation))


class ContrastMetric(BaseMetric):

    name = "contrast"

    def compute(self, img: np.ndarray) -> float:
        gray = to_gray(img).astype(np.float32)
        return float(gray.std() / 255.0)


class ColorfulnessMetric(BaseMetric):

    name = "colorfulness"

    def compute(self, img: np.ndarray) -> float:
        if len(img.shape) == 2:
            return 0.0
        b, g, r = cv2.split(img.astype(np.float32))
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        std_rg, std_yb = rg.std(), yb.std()
        mean_rg, mean_yb = rg.mean(), yb.mean()
        return float(
            np.sqrt(std_rg ** 2 + std_yb ** 2)
            + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
        )


class ChromaNoiseMetric(BaseMetric):

    name = "chroma_noise"

    def __init__(self, block: int = 32, min_blocks: int = 20):
        self.block = block
        self.min_blocks = min_blocks

    def compute(self, img: np.ndarray) -> float:
        if len(img.shape) == 2:
            return 0.0
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        a, b = lab[:, :, 1], lab[:, :, 2]
        h, w = a.shape
        stds = []
        for y in range(0, h - self.block, self.block):
            for x in range(0, w - self.block, self.block):
                ba = a[y:y + self.block, x:x + self.block]
                bb = b[y:y + self.block, x:x + self.block]
                stds.append(np.sqrt(ba.var() + bb.var()))
        stds = np.array(stds)
        if len(stds) < self.min_blocks:
            return -1.0
        k = max(self.min_blocks, int(len(stds) * 0.1))
        return float(np.mean(np.sort(stds)[:k]))


class HighlightRecoveryMetric(BaseMetric):

    name = "highlight_rec"

    def compute(self, img: np.ndarray) -> float:
        gray = to_gray(img).astype(np.float32)
        threshold = np.percentile(gray, 95)
        bright = gray >= threshold
        if bright.sum() < 100:
            return 0.5

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        baseline = float(lap.std())

        if baseline < 1e-3:
            return 0.5

        texture = float(lap[bright].std())
        ratio = texture / baseline
        return float(np.clip(ratio, 0.0, 1.5) / 1.5)


class ShadowDetailMetric(BaseMetric):

    name = "shadow_detail"

    def compute(self, img: np.ndarray) -> float:
        gray = to_gray(img).astype(np.float32)
        threshold = np.percentile(gray, 5)
        dark = gray <= threshold
        if dark.sum() < 100:
            return 0.5

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        baseline = float(lap.std())

        if baseline < 1e-3:
            return 0.5

        texture = float(lap[dark].std())
        ratio = texture / baseline
        return float(np.clip(ratio, 0.0, 1.5) / 1.5)