import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from inf8239_u02_cv.models import build_cnn


def test_cnn_output_is_ten_finite_probabilities():
    model = build_cnn(tf)
    probabilities = model(np.zeros((2, 28, 28, 1), dtype="float32")).numpy()
    assert probabilities.shape == (2, 10)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)