"""Shared, paper-aligned data validation and specimen-level operations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler

from usda_texture import classify_usda_rows


RANDOM_SEED = 42
N_SPLITS = 5
BLOCKS_PER_SPECIMEN = 100
WAVELENGTHS_NM = (365, 405, 473, 530, 575, 621, 660, 735, 770, 830, 850, 890, 940)
FEATURE_COLUMNS = tuple(str(value) for value in WAVELENGTHS_NM)
TARGET_COLUMNS = ("Clay", "Silt", "Sand")
LABEL_COLUMN = "Soil_Type"
SPECIMEN_COLUMN = "Specimen_ID"
LDA_POWER_TO_KEEP = 0.99
EXPECTED_LDA_COMPONENTS = 5

EXPECTED_DATASETS = {
    "training/testing": {"rows": 44_000, "specimens": 440, "composition_groups": 22},
    "external validation": {"rows": 8_400, "specimens": 84, "composition_groups": 7},
}


def require_finite(values: np.ndarray, name: str = "features") -> np.ndarray:
    """Reject NaN/Inf explicitly; the published pipeline has no imputation stage."""

    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        bad = np.argwhere(~np.isfinite(array))
        first = tuple(int(item) for item in bad[0]) if bad.size else "unknown"
        raise ValueError(f"{name} contains NaN or Inf (first invalid position: {first}).")
    return array


def lda_components_for_power(explained_ratios: Sequence[float], power: float = LDA_POWER_TO_KEEP) -> int:
    """Return the smallest LDA dimension meeting the cumulative-power target."""

    ratios = require_finite(np.asarray(explained_ratios, dtype=float), "LDA power ratios")
    if ratios.ndim != 1 or len(ratios) == 0:
        raise ValueError("LDA power ratios must be a nonempty one-dimensional sequence.")
    if not 0.0 < power <= 1.0:
        raise ValueError("LDA cumulative-power target must lie in (0, 1].")
    return min(len(ratios), int(np.searchsorted(np.cumsum(ratios), power, side="left") + 1))


def specimen_ids_from_dataframe(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    blocks_per_specimen: int = BLOCKS_PER_SPECIMEN,
) -> np.ndarray:
    """Use explicit specimen IDs, with a validated legacy contiguous-row fallback."""

    if SPECIMEN_COLUMN in frame.columns:
        if frame[SPECIMEN_COLUMN].isna().any():
            raise ValueError(f"{dataset_name}: {SPECIMEN_COLUMN} contains missing values.")
        local_ids = frame[SPECIMEN_COLUMN].astype(str).to_numpy()
    else:
        if len(frame) % blocks_per_specimen:
            raise ValueError(
                f"{dataset_name}: {len(frame)} rows is not divisible by "
                f"{blocks_per_specimen} blocks per specimen."
            )
        warnings.warn(
            f"{dataset_name}: legacy CSV has no {SPECIMEN_COLUMN}; inferring one "
            f"physical specimen per consecutive {blocks_per_specimen} rows. "
            "Regenerate the CSV with Make_Datafiles.py to store explicit IDs.",
            UserWarning,
            stacklevel=2,
        )
        local_ids = np.repeat(
            np.arange(len(frame) // blocks_per_specimen).astype(str),
            blocks_per_specimen,
        )
    # Namespacing prevents accidental collisions when datasets are combined.
    return np.char.add(f"{dataset_name}:", local_ids.astype(str))


def composition_group_labels(frame: pd.DataFrame) -> np.ndarray:
    """Create stable labels for unique ground-truth clay/silt/sand triplets."""

    missing = [column for column in TARGET_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing composition columns: {missing}")
    values = require_finite(frame[list(TARGET_COLUMNS)].to_numpy(), "composition targets")
    return np.asarray(
        ["Cl{:.12g}-Si{:.12g}-Sa{:.12g}".format(*row) for row in values],
        dtype=object,
    )


def validate_block_dataset(
    frame: pd.DataFrame,
    groups: Sequence[str],
    *,
    dataset_name: str,
    expected: dict[str, int] | None = None,
) -> None:
    """Validate dimensions, finiteness, block counts, and within-specimen truth."""

    required = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS) + [LABEL_COLUMN]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{dataset_name}: missing required columns: {missing}")
    groups_array = np.asarray(groups)
    if len(groups_array) != len(frame):
        raise ValueError(f"{dataset_name}: specimen ID count does not match row count.")
    require_finite(frame[list(FEATURE_COLUMNS)].to_numpy(), f"{dataset_name} spectral features")
    require_finite(frame[list(TARGET_COLUMNS)].to_numpy(), f"{dataset_name} targets")

    work = frame[list(TARGET_COLUMNS) + [LABEL_COLUMN]].copy()
    work[SPECIMEN_COLUMN] = groups_array
    counts = work.groupby(SPECIMEN_COLUMN, sort=False).size()
    bad_counts = counts[counts != BLOCKS_PER_SPECIMEN]
    if not bad_counts.empty:
        raise ValueError(
            f"{dataset_name}: every physical specimen must have exactly "
            f"{BLOCKS_PER_SPECIMEN} blocks; invalid counts: {bad_counts.head().to_dict()}"
        )
    for column in list(TARGET_COLUMNS) + [LABEL_COLUMN]:
        varying = work.groupby(SPECIMEN_COLUMN, sort=False)[column].nunique(dropna=False)
        if (varying != 1).any():
            raise ValueError(f"{dataset_name}: {column} varies within a physical specimen.")

    specimen_truth = work.groupby(SPECIMEN_COLUMN, sort=False).first()
    canonical = classify_usda_rows(specimen_truth[list(TARGET_COLUMNS)].to_numpy())
    stored = specimen_truth[LABEL_COLUMN].astype(str).to_numpy()
    if not np.array_equal(canonical, stored):
        mismatches = specimen_truth.loc[canonical != stored].copy()
        mismatches["Canonical_USDA"] = canonical[canonical != stored]
        raise ValueError(
            f"{dataset_name}: stored {LABEL_COLUMN} disagrees with the canonical USDA "
            f"rules for {len(mismatches)} specimen(s):\n{mismatches.head()}"
        )

    if expected is not None:
        actual = {
            "rows": len(frame),
            "specimens": len(counts),
            "composition_groups": len(np.unique(composition_group_labels(specimen_truth))),
        }
        differences = {key: (expected[key], actual[key]) for key in expected if actual[key] != expected[key]}
        if differences:
            raise ValueError(f"{dataset_name}: official dataset dimensions do not match: {differences}")


def assert_no_group_leakage(train_groups: Iterable[str], test_groups: Iterable[str], fold: int) -> None:
    overlap = set(train_groups).intersection(test_groups)
    if overlap:
        raise AssertionError(f"Fold {fold}: {len(overlap)} specimen(s) occur in train and test.")


def assert_complete_oof(test_group_sets: Sequence[Sequence[str]], all_groups: Sequence[str]) -> None:
    """Assert every physical specimen occurs in exactly one held-out fold."""

    held_out = np.concatenate([np.unique(values) for values in test_group_sets])
    unique, counts = np.unique(held_out, return_counts=True)
    if set(unique) != set(np.unique(all_groups)) or np.any(counts != 1):
        raise AssertionError("Out-of-fold coverage is incomplete or duplicates specimens.")


def specimen_first(values: Sequence, groups: Sequence[str]) -> pd.Series:
    table = pd.DataFrame({SPECIMEN_COLUMN: groups, "value": values})
    uniqueness = table.groupby(SPECIMEN_COLUMN)["value"].nunique(dropna=False)
    if (uniqueness != 1).any():
        raise ValueError("Ground truth varies within at least one physical specimen.")
    return table.groupby(SPECIMEN_COLUMN, sort=True)["value"].first()


def aggregate_prediction_to_specimen(
    block_predictions: np.ndarray,
    groups: Sequence[str],
) -> pd.DataFrame:
    """Average raw block regressions only; never perform Eq. (14) here."""

    predictions = require_finite(block_predictions, "block-level regression predictions")
    if predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("Regression predictions must have shape (n_blocks, 3).")
    if len(predictions) != len(groups):
        raise ValueError("Prediction and specimen ID counts differ.")
    table = pd.DataFrame(predictions, columns=TARGET_COLUMNS)
    table[SPECIMEN_COLUMN] = np.asarray(groups)
    return table.groupby(SPECIMEN_COLUMN, sort=True)[list(TARGET_COLUMNS)].mean()


def aggregate_probabilities_to_specimen(
    probabilities: np.ndarray,
    groups: Sequence[str],
    model_classes: Sequence[int],
    all_classes: Sequence[int],
) -> pd.DataFrame:
    """Average block class-probability vectors for each physical specimen."""

    probabilities = require_finite(probabilities, "classification probabilities")
    full = np.zeros((len(probabilities), len(all_classes)), dtype=float)
    positions = {label: index for index, label in enumerate(all_classes)}
    for local_index, label in enumerate(model_classes):
        full[:, positions[label]] = probabilities[:, local_index]
    table = pd.DataFrame(full, columns=all_classes)
    table[SPECIMEN_COLUMN] = np.asarray(groups)
    return table.groupby(SPECIMEN_COLUMN, sort=True)[list(all_classes)].mean()


@dataclass
class FoldPreprocessor:
    """Training-only min-max scaling followed by training-only supervised LDA."""

    supervisory_labels: str
    expected_components: int | None = EXPECTED_LDA_COMPONENTS
    power_to_keep: float = LDA_POWER_TO_KEEP

    def __post_init__(self) -> None:
        self.scaler = MinMaxScaler()
        self.lda: LinearDiscriminantAnalysis | None = None
        self.n_components_: int | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, labels: Sequence) -> "FoldPreprocessor":
        x_train = require_finite(x_train, "training spectral features")
        labels = np.asarray(labels)
        if len(labels) != len(x_train):
            raise ValueError("LDA label and training-feature counts differ.")
        scaled = self.scaler.fit_transform(x_train)
        n_classes = len(np.unique(labels))
        max_components = min(scaled.shape[1], n_classes - 1)
        if max_components < 1:
            raise ValueError("LDA requires at least two supervisory groups.")
        self.lda = LinearDiscriminantAnalysis(n_components=max_components)
        transformed = self.lda.fit_transform(scaled, labels)
        ratios = np.asarray(self.lda.explained_variance_ratio_, dtype=float)
        self.n_components_ = min(
            transformed.shape[1], lda_components_for_power(ratios, self.power_to_keep)
        )
        self.explained_variance_ratio_ = ratios
        if self.expected_components is not None and self.n_components_ != self.expected_components:
            raise AssertionError(
                f"{self.supervisory_labels} LDA retained {self.n_components_} components "
                f"at {self.power_to_keep:.0%}; the official dataset is expected to retain "
                f"{self.expected_components}."
            )
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.lda is None or self.n_components_ is None:
            raise RuntimeError("FoldPreprocessor must be fitted before transform.")
        scaled = self.scaler.transform(require_finite(values, "spectral features to transform"))
        return self.lda.transform(scaled)[:, : self.n_components_]

    def fit_transform(self, values: np.ndarray, labels: Sequence) -> np.ndarray:
        return self.fit(values, labels).transform(values)


def grouped_specimen_truth(frame: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    """Return one ground-truth row and mean spectral signature per specimen."""

    columns = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS) + [LABEL_COLUMN]
    work = frame[columns].copy()
    work[SPECIMEN_COLUMN] = np.asarray(groups)
    aggregations = {column: "mean" for column in FEATURE_COLUMNS}
    aggregations.update({column: "first" for column in list(TARGET_COLUMNS) + [LABEL_COLUMN]})
    return work.groupby(SPECIMEN_COLUMN, sort=True).agg(aggregations)
