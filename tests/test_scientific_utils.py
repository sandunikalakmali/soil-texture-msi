import numpy as np
import pandas as pd
import pytest

from scientific_utils import (
    BLOCKS_PER_SPECIMEN,
    FEATURE_COLUMNS,
    SPECIMEN_COLUMN,
    aggregate_prediction_to_specimen,
    assert_no_group_leakage,
    lda_components_for_power,
    specimen_ids_from_dataframe,
    validate_block_dataset,
)


def specimen_frame():
    rows = []
    for specimen_id, composition, texture in [
        ("A", (5, 5, 90), "Sand"),
        ("B", (50, 20, 30), "Clay"),
    ]:
        for block in range(BLOCKS_PER_SPECIMEN):
            row = {column: float(block) for column in FEATURE_COLUMNS}
            row.update({
                SPECIMEN_COLUMN: specimen_id,
                "Clay": composition[0], "Silt": composition[1], "Sand": composition[2],
                "Soil_Type": texture,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def test_expected_thirteen_wavelength_features():
    assert FEATURE_COLUMNS == (
        "365", "405", "473", "530", "575", "621", "660",
        "735", "770", "830", "850", "890", "940",
    )


def test_explicit_specimen_ids_and_100_block_validation():
    frame = specimen_frame()
    groups = specimen_ids_from_dataframe(frame, dataset_name="test")
    validate_block_dataset(frame, groups, dataset_name="test")
    counts = pd.Series(groups).value_counts()
    assert counts.tolist() == [100, 100]


def test_legacy_grouping_is_warned_and_validated():
    frame = specimen_frame().drop(columns=SPECIMEN_COLUMN)
    with pytest.warns(UserWarning, match="legacy CSV"):
        groups = specimen_ids_from_dataframe(frame, dataset_name="test")
    validate_block_dataset(frame, groups, dataset_name="test")


def test_invalid_specimen_block_count_is_rejected():
    frame = specimen_frame().iloc[:-1]
    groups = specimen_ids_from_dataframe(frame, dataset_name="test")
    with pytest.raises(ValueError, match="exactly 100 blocks"):
        validate_block_dataset(frame, groups, dataset_name="test")


def test_group_leakage_detection():
    with pytest.raises(AssertionError, match="train and test"):
        assert_no_group_leakage(["A", "B"], ["B", "C"], fold=1)
    assert_no_group_leakage(["A"], ["B"], fold=1)


def test_regression_aggregation_does_not_project_to_simplex():
    predictions = np.tile([20.0, 20.0, 20.0], (100, 1))
    aggregated = aggregate_prediction_to_specimen(predictions, ["A"] * 100)
    assert aggregated.loc["A"].to_numpy() == pytest.approx([20, 20, 20])
    assert aggregated.loc["A"].sum() == pytest.approx(60)


def test_99_percent_lda_selection_logic_returns_five_components():
    ratios = [0.50, 0.20, 0.15, 0.08, 0.061, 0.009]
    assert lda_components_for_power(ratios, 0.99) == 5
