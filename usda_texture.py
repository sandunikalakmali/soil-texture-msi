"""Canonical USDA soil texture triangle rules used throughout the repository.

The public functions use the scientifically conventional component order
``(clay, silt, sand)``. Regression predictions must first be passed through
``project_to_compositional_simplex`` (paper Eq. 14); ground-truth compositions
may be classified directly because this function normalizes small rounding
departures from 100 percent.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


USDA_CLASS_ORDER = (
    "Sand",
    "Loamy Sand",
    "Sandy Loam",
    "Loam",
    "Silt Loam",
    "Silt",
    "Sandy Clay Loam",
    "Clay Loam",
    "Silty Clay Loam",
    "Sandy Clay",
    "Silty Clay",
    "Clay",
)


def project_to_compositional_simplex(values: Iterable[float] | np.ndarray) -> np.ndarray:
    """Apply Eq. (14): clip negative components and rescale each row to 100%.

    This operation is solely for indirect USDA classification. It must not be
    applied before calculating regression R2 or RMSE. A row containing no
    positive component has no defined composition and therefore raises an
    error instead of being silently mapped to a texture class.
    """

    array = np.asarray(values, dtype=float)
    one_dimensional = array.ndim == 1
    if one_dimensional:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Composition values must have shape (3,) or (n, 3).")
    if not np.isfinite(array).all():
        raise ValueError("Composition values must all be finite.")

    clipped = np.clip(array, 0.0, None)
    totals = clipped.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("A composition must contain at least one positive component.")
    projected = clipped * (100.0 / totals)
    return projected[0] if one_dimensional else projected


def classify_usda_texture(clay: float, silt: float, sand: float) -> str:
    """Return one of the 12 USDA texture classes for clay/silt/sand percentages.

    Inputs must be finite and nonnegative. They are rescaled to 100% to account
    for ordinary rounding in laboratory-derived ground truth. The inequality
    order intentionally resolves points lying exactly on USDA boundaries.
    """

    composition = np.asarray([clay, silt, sand], dtype=float)
    if not np.isfinite(composition).all():
        raise ValueError("USDA texture inputs must all be finite.")
    if np.any(composition < 0.0):
        raise ValueError(
            "USDA texture inputs must be nonnegative; project regression "
            "predictions with project_to_compositional_simplex first."
        )
    total = float(composition.sum())
    if total <= 0.0:
        raise ValueError("USDA texture inputs must have a positive total.")
    clay, silt, sand = composition * (100.0 / total)

    if (silt + 1.5 * clay) < 15.0:
        return "Sand"
    if (silt + 2.0 * clay) < 30.0:
        return "Loamy Sand"
    if (
        (7.0 <= clay < 20.0 and sand > 52.0)
        or (clay < 7.0 and silt < 50.0)
    ):
        return "Sandy Loam"
    if 7.0 <= clay < 27.0 and 28.0 <= silt < 50.0 and sand <= 52.0:
        return "Loam"
    if (
        (50.0 <= silt and 12.0 <= clay < 27.0)
        or (50.0 <= silt < 80.0 and clay < 12.0)
    ):
        return "Silt Loam"
    if silt >= 80.0 and clay < 12.0:
        return "Silt"
    if 20.0 <= clay < 35.0 and silt < 28.0 and sand > 45.0:
        return "Sandy Clay Loam"
    if 27.0 <= clay < 40.0 and 20.0 < sand <= 45.0:
        return "Clay Loam"
    if 27.0 <= clay < 40.0 and sand <= 20.0:
        return "Silty Clay Loam"
    if clay >= 35.0 and sand > 45.0:
        return "Sandy Clay"
    if clay >= 40.0 and silt >= 40.0:
        return "Silty Clay"
    if clay >= 40.0 and sand <= 45.0 and silt < 40.0:
        return "Clay"

    raise ValueError(
        f"Composition did not match a USDA region after normalization: "
        f"clay={clay:.8g}, silt={silt:.8g}, sand={sand:.8g}."
    )


def classify_usda_rows(values: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
    """Classify an ``(n, 3)`` array ordered as clay, silt, sand."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Composition values must have shape (n, 3).")
    return np.asarray(
        [classify_usda_texture(clay, silt, sand) for clay, silt, sand in array],
        dtype=object,
    )
