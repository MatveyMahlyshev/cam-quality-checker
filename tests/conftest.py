import numpy as np
import pytest


@pytest.fixture
def random_image():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)


@pytest.fixture
def sharp_image():
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    for y in range(0, 600, 20):
        img[y:y + 10, :, :] = 255
    return img


@pytest.fixture
def blurred_image(sharp_image):
    import cv2
    return cv2.GaussianBlur(sharp_image, (31, 31), 10)


@pytest.fixture
def flat_image():
    return np.full((600, 800, 3), 128, dtype=np.uint8)


@pytest.fixture
def dark_image():
    return np.full((600, 800, 3), 10, dtype=np.uint8)


@pytest.fixture
def overexposed_image():
    return np.full((600, 800, 3), 250, dtype=np.uint8)


@pytest.fixture
def small_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)