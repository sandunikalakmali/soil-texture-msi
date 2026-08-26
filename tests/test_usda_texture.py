import numpy as np
import pytest

from usda_texture import (
    USDA_CLASS_ORDER,
    classify_usda_texture,
    project_to_compositional_simplex,
)


@pytest.mark.parametrize(
    ("expected", "clay", "silt", "sand"),
    [
        ("Sand", 5, 5, 90),
        ("Loamy Sand", 5, 10, 85),
        ("Sandy Loam", 10, 20, 70),
        ("Loam", 18, 40, 42),
        ("Silt Loam", 10, 65, 25),
        ("Silt", 5, 90, 5),
        ("Sandy Clay Loam", 25, 10, 65),
        ("Clay Loam", 32, 34, 34),
        ("Silty Clay Loam", 32, 55, 13),
        ("Sandy Clay", 38, 5, 57),
        ("Silty Clay", 45, 45, 10),
        ("Clay", 50, 20, 30),
    ],
)
def test_all_twelve_usda_class_interiors(expected, clay, silt, sand):
    assert classify_usda_texture(clay, silt, sand) == expected


def test_class_order_contains_exactly_the_twelve_regions():
    assert len(USDA_CLASS_ORDER) == 12
    assert len(set(USDA_CLASS_ORDER)) == 12


@pytest.mark.parametrize(
    ("composition", "expected"),
    [
        ((0, 15, 85), "Loamy Sand"),
        ((7, 16, 77), "Sandy Loam"),
        ((20, 20, 60), "Sandy Clay Loam"),
        ((27, 53, 20), "Silty Clay Loam"),
        ((40, 40, 20), "Silty Clay"),
    ],
)
def test_boundary_conditions(composition, expected):
    assert classify_usda_texture(*composition) == expected


@pytest.mark.parametrize("composition", [[40, 30, 30], [-2, 52, 51], [25, 25, 25]])
def test_eq14_projection_is_nonnegative_and_sums_to_100(composition):
    projected = project_to_compositional_simplex(composition)
    assert np.all(projected >= 0)
    assert projected.sum() == pytest.approx(100.0)


def test_eq14_clips_negative_before_normalizing():
    projected = project_to_compositional_simplex([-2, 52, 51])
    assert projected[0] == 0
    assert projected[1:] == pytest.approx([52 / 103 * 100, 51 / 103 * 100])


@pytest.mark.parametrize("composition", [(-1, 50, 51), (np.nan, 50, 50), (0, 0, 0)])
def test_invalid_unprocessed_usda_inputs_raise(composition):
    with pytest.raises(ValueError):
        classify_usda_texture(*composition)


def test_zero_vector_cannot_be_projected():
    with pytest.raises(ValueError):
        project_to_compositional_simplex([0, 0, 0])
