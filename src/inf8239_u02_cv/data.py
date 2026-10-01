from __future__ import annotations

import hashlib

import numpy as np


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Scale uint8 Fashion-MNIST pixels and add the grayscale channel."""
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"Expected image shape (n, 28, 28), received {images.shape}.")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
        raise ValueError("Raw image pixels must be finite and in [0, 255].")
    return images.astype("float32")[..., np.newaxis] / 255.0


def validate_images(images: np.ndarray, labels: np.ndarray) -> None:
    """Validate model input shape, labels, and normalized pixel range."""
    if images.ndim != 4 or images.shape[1:] != (28, 28, 1):
        raise ValueError(f"Unexpected image shape: {images.shape}.")
    if labels.ndim != 1 or len(images) != len(labels):
        raise ValueError("Image and label counts must match.")
    if len(images) == 0:
        raise ValueError("The image dataset cannot be empty.")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Normalized images must be finite and in [0, 1].")
    if not np.isin(labels, np.arange(len(CLASS_NAMES))).all():
        raise ValueError("Labels must be integers from 0 to 9.")


def array_sha256(array: np.ndarray) -> str:
    """Return a SHA-256 fingerprint for a contiguous NumPy array."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def audit_splits(splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Summarize split shapes, class counts, ranges, and array fingerprints."""
    summary = {"dataset": "Fashion-MNIST", "source": "keras.datasets.fashion_mnist", "splits": {}}
    for split_name, (images, labels) in splits.items():
        if images.ndim != 3 or images.shape[1:] != (28, 28):
            raise ValueError(f"{split_name} images must have shape (n, 28, 28).")
        if labels.ndim != 1 or len(images) != len(labels):
            raise ValueError(f"{split_name} image and label counts must match.")
        if not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
            raise ValueError(f"{split_name} pixels must be finite and in [0, 255].")
        if not np.isin(labels, np.arange(len(CLASS_NAMES))).all():
            raise ValueError(f"{split_name} labels must be integers from 0 to 9.")
        counts = np.bincount(labels.astype(int), minlength=len(CLASS_NAMES))
        summary["splits"][split_name] = {
            "count": int(len(labels)),
            "image_shape": list(images.shape[1:]),
            "pixel_range": [int(images.min()), int(images.max())],
            "class_counts": {
                CLASS_NAMES[index]: int(count) for index, count in enumerate(counts)
            },
            "images_sha256": array_sha256(images),
            "labels_sha256": array_sha256(labels),
        }
    return summary