import numpy as np
import pytest

from validators import DefaultImageValidator, ValidationResult


def test_validation_result_bool_true():
    r = ValidationResult(True, "OK")
    assert bool(r) is True


def test_validation_result_bool_false():
    r = ValidationResult(False, "fail")
    assert bool(r) is False


def test_validator_rejects_small(small_image):
    v = DefaultImageValidator()
    r = v.validate(small_image)
    assert not r
    assert "разрешение" in r.message.lower()


def test_validator_rejects_overexposed(overexposed_image):
    v = DefaultImageValidator()
    r = v.validate(overexposed_image)
    assert not r


def test_validator_rejects_flat(flat_image):
    v = DefaultImageValidator()
    r = v.validate(flat_image)
    assert not r


def test_validator_accepts_random(random_image):
    v = DefaultImageValidator()
    r = v.validate(random_image)
    assert r
    assert r.message == "OK"


def test_validator_none_input():
    v = DefaultImageValidator()
    r = v.validate(None)
    assert not r


def test_validator_custom_min_side():
    v = DefaultImageValidator(min_short_side=1000)
    img = np.random.default_rng(0).integers(0, 256, (800, 800, 3), dtype=np.uint8)
    r = v.validate(img)
    assert not r


def test_validator_warns_on_small_long_side():
    img = np.random.default_rng(0).integers(0, 256, (600, 800, 3), dtype=np.uint8)
    v = DefaultImageValidator()
    r = v.validate(img)
    assert r
    assert any("рекомендуем" in w.lower() or "разрешени" in w.lower() for w in r.warnings)


def test_validator_warns_on_heic_extension(random_image):
    v = DefaultImageValidator()
    r = v.validate(random_image, filename="photo.heic")
    assert r
    assert any("heic" in w.lower() for w in r.warnings)