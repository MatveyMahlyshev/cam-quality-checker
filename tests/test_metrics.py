import cv2
import numpy as np
import pytest

from metrics.basic import (
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


def test_sharpness_sharp_greater_than_blurred(sharp_image, blurred_image):
    m = SharpnessMetric()
    assert m.compute(sharp_image) > m.compute(blurred_image)


def test_sharpness_returns_finite(random_image):
    m = SharpnessMetric()
    v = m.compute(random_image)
    assert np.isfinite(v)
    assert v >= 0


def test_noise_on_flat_low(flat_image):
    m = NoiseMetric()
    assert m.compute(flat_image) < 1.0


def test_noise_returns_minus_one_on_tiny_image(small_image):
    m = NoiseMetric()
    assert m.compute(small_image) == -1.0


def test_dynamic_range_uniform_low(flat_image):
    m = DynamicRangeMetric()
    assert m.compute(flat_image) < 0.5


def test_dynamic_range_random_higher_than_flat(random_image, flat_image):
    m = DynamicRangeMetric()
    assert m.compute(random_image) > m.compute(flat_image)


def test_exposure_perfect_on_gray_128(flat_image):
    m = ExposureMetric()
    assert m.compute(flat_image) == pytest.approx(1.0, abs=1e-6)


def test_exposure_low_on_dark(dark_image):
    m = ExposureMetric()
    assert m.compute(dark_image) < 0.2


def test_exposure_low_on_overexposed(overexposed_image):
    m = ExposureMetric()
    assert m.compute(overexposed_image) < 0.2


def test_contrast_zero_on_flat(flat_image):
    m = ContrastMetric()
    assert m.compute(flat_image) == pytest.approx(0.0, abs=1e-6)


def test_contrast_positive_on_random(random_image):
    m = ContrastMetric()
    assert m.compute(random_image) > 0.0


def test_colorfulness_zero_on_grayscale():
    gray = np.full((600, 800), 128, dtype=np.uint8)
    m = ColorfulnessMetric()
    assert m.compute(gray) == 0.0


def test_colorfulness_positive_on_random(random_image):
    m = ColorfulnessMetric()
    assert m.compute(random_image) > 0.0


def test_chroma_noise_on_gray_returns_zero():
    gray = np.full((600, 800), 128, dtype=np.uint8)
    m = ChromaNoiseMetric()
    assert m.compute(gray) == 0.0


def test_chroma_noise_on_flat_low(flat_image):
    m = ChromaNoiseMetric()
    v = m.compute(flat_image)
    assert v >= 0
    assert v < 1.0


def test_highlight_recovery_neutral_on_flat(flat_image):
    m = HighlightRecoveryMetric()
    assert m.compute(flat_image) == 0.5


def test_shadow_detail_neutral_on_flat(flat_image):
    m = ShadowDetailMetric()
    assert m.compute(flat_image) == 0.5


def test_highlight_recovery_range(random_image):
    m = HighlightRecoveryMetric()
    v = m.compute(random_image)
    assert 0.0 <= v <= 1.0


def test_shadow_detail_range(random_image):
    m = ShadowDetailMetric()
    v = m.compute(random_image)
    assert 0.0 <= v <= 1.0


def test_highlight_recovers_texture(sharp_image):
    m = HighlightRecoveryMetric()
    v = m.compute(sharp_image)
    assert v > 0.5