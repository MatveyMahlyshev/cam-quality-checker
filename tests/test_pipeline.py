import numpy as np

from metrics import compute_all, MetricsPipeline
from metrics.utils import normalize_size, to_gray


def test_to_gray_color_image(random_image):
    g = to_gray(random_image)
    assert g.ndim == 2


def test_to_gray_grayscale_passthrough():
    gray = np.zeros((100, 100), dtype=np.uint8)
    assert to_gray(gray) is gray


def test_normalize_size_downscales():
    img = np.zeros((3000, 4000, 3), dtype=np.uint8)
    out, resized = normalize_size(img, long_side=1920)
    assert resized
    assert max(out.shape[:2]) == 1920


def test_normalize_size_no_upscale():
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    out, resized = normalize_size(img, long_side=1920)
    assert not resized
    assert out is img


def test_compute_all_has_all_keys(random_image):
    m = compute_all(random_image)
    expected = {
        "sharpness", "noise", "dynamic_range", "exposure",
        "contrast", "colorfulness", "chroma_noise",
        "highlight_rec", "shadow_detail",
        "original_size", "analyzed_size", "was_resized",
    }
    assert expected.issubset(m.keys())


def test_compute_all_original_size(random_image):
    m = compute_all(random_image)
    assert m["original_size"] == (800, 600)


def test_compute_all_resized_flag():
    big = np.zeros((3000, 4000, 3), dtype=np.uint8)
    m = compute_all(big)
    assert m["was_resized"] is True


def test_pipeline_custom_long_side(random_image):
    p = MetricsPipeline(long_side=512)
    m = p.compute(random_image)
    assert max(m["analyzed_size"]) <= 512