import numpy as np

from .utils import normalize_size
from .basic import (
    SharpnessMetric,
    NoiseMetric,
    DynamicRangeMetric,
    ExposureMetric,
    ContrastMetric,
    ColorfulnessMetric,
    ChromaNoiseMetric,
    HighlightRecoveryMetric,
    ShadowDetailMetric,
)


class MetricsPipeline:

    def __init__(self, long_side: int = 1920):
        self.long_side = long_side

        self.on_original = [
            NoiseMetric(),
            ChromaNoiseMetric(),
        ]

        self.on_resized = [
            SharpnessMetric(),
            DynamicRangeMetric(),
            ExposureMetric(),
            ContrastMetric(),
            ColorfulnessMetric(),
            HighlightRecoveryMetric(),
            ShadowDetailMetric(),
        ]

    def compute(self, img: np.ndarray) -> dict:
        result = {}

        for metric in self.on_original:
            result[metric.name] = metric.compute(img)

        resized, was_resized = normalize_size(img, long_side=self.long_side)

        for metric in self.on_resized:
            result[metric.name] = metric.compute(resized)

        result["original_size"] = (img.shape[1], img.shape[0])
        result["analyzed_size"] = (resized.shape[1], resized.shape[0])
        result["was_resized"] = was_resized

        return result


_default_pipeline = MetricsPipeline()


def compute_all(img: np.ndarray) -> dict:
    return _default_pipeline.compute(img)