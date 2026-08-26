import numpy as np
import pytest


cv2 = pytest.importorskip("cv2")

from Make_Datafiles import TANH_KAPPA, bounded_tanh_normalize


def test_bounded_tanh_matches_paper_equation():
    roi = np.arange(100, dtype=float).reshape(10, 10)
    mean = roi.mean()
    sigma = roi.std()
    expected = (mean - sigma) + 2 * sigma * (
        (np.tanh(TANH_KAPPA * (roi - mean)) + 1) / 2
    )
    assert bounded_tanh_normalize(roi) == pytest.approx(expected)


def test_saturated_subtraction_matches_max_x_minus_dark_zero():
    captured = np.asarray([[5, 20]], dtype=np.uint8)
    dark = np.asarray([[10, 3]], dtype=np.uint8)
    assert cv2.subtract(captured, dark).tolist() == [[0, 17]]
