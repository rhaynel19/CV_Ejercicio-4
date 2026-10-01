import numpy as np
import pytest

from inf8239_u02_cv.data import audit_splits, normalize_images, validate_images


def test_normalize_adds_channel_and_scales_pixels():
    raw = np.zeros((2, 28, 28), dtype=np.uint8)
    raw[1] = 255
    result = normalize_images(raw)
    assert result.shape == (2, 28, 28, 1)
    assert result.dtype == np.float32
    assert result.min() == 0
    assert result.max() == 1


def test_normalize_rejects_wrong_shape_and_range():
    with pytest.raises(ValueError, match="Expected image shape"):
        normalize_images(np.zeros((2, 27, 28), dtype=np.uint8))
    invalid = np.zeros((2, 28, 28), dtype=np.int16)
    invalid[0, 0, 0] = 256
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        normalize_images(invalid)


def test_validate_rejects_shape_label_and_pixel_errors():
    with pytest.raises(ValueError, match="Unexpected image shape"):
        validate_images(np.zeros((2, 28, 28)), np.zeros(2, dtype=int))
    with pytest.raises(ValueError, match="counts must match"):
        validate_images(np.zeros((2, 28, 28, 1)), np.zeros(1, dtype=int))
    invalid = np.full((1, 28, 28, 1), 1.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_images(invalid, np.zeros(1, dtype=int))


def test_audit_records_counts_and_hashes():
    images = np.zeros((10, 28, 28), dtype=np.uint8)
    labels = np.arange(10, dtype=np.uint8)
    audit = audit_splits({"train": (images, labels)})
    assert audit["splits"]["train"]["count"] == 10
    assert sum(audit["splits"]["train"]["class_counts"].values()) == 10
    assert len(audit["splits"]["train"]["images_sha256"]) == 64


def test_audit_rejects_misaligned_data():
    with pytest.raises(ValueError, match="counts must match"):
        audit_splits({"train": (np.zeros((2, 28, 28)), np.zeros(1, dtype=int))})