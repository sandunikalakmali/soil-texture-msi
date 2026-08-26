import numpy as np
import pandas as pd

from Direct_Soil_Classification import (
    MODEL_ORDER as CLASSIFIER_ORDER,
    run_classification_statistics,
)
from Soil_Composition_Prediction_and_Indirect_Classification import (
    MODEL_ORDER as REGRESSOR_ORDER,
    indirect_statistics,
    regression_statistics,
)


def test_direct_post_hoc_is_skipped_when_cochran_q_is_not_significant():
    predictions = []
    metrics = []
    pattern = np.asarray([0, 1] * 6)
    for model_index, model in enumerate(CLASSIFIER_ORDER):
        values = np.roll(pattern, model_index)
        predictions.extend(
            {"Model": model, "Specimen_ID": str(index), "Correct": int(value)}
            for index, value in enumerate(values)
        )
        metrics.extend(
            {"Model": model, "Fold": fold, "Accuracy": float(values.mean())}
            for fold in range(5)
        )
    omnibus, pairwise, summary = run_classification_statistics(
        pd.DataFrame(predictions), pd.DataFrame(metrics)
    )
    assert not bool(omnibus.loc[0, "Significant"])
    assert pairwise.empty
    assert not bool(summary.loc[0, "Post_Hoc_Performed"])


def test_regression_post_hoc_is_skipped_when_friedman_is_not_significant():
    records = []
    base = np.arange(len(REGRESSOR_ORDER), dtype=float)
    for specimen in range(len(REGRESSOR_ORDER)):
        rotated = np.roll(base, specimen)
        records.extend(
            {
                "Model": model,
                "Specimen_ID": str(specimen),
                "Composition_Error": rotated[model_index],
            }
            for model_index, model in enumerate(REGRESSOR_ORDER)
        )
    omnibus, pairwise, summary = regression_statistics(pd.DataFrame(records))
    assert not bool(omnibus.loc[0, "Significant"])
    assert pairwise.empty
    assert not bool(summary.loc[0, "Post_Hoc_Performed"])


def test_indirect_post_hoc_is_skipped_when_cochran_q_is_not_significant():
    predictions = []
    metrics = []
    pattern = np.asarray([0, 1] * 6)
    for model_index, model in enumerate(REGRESSOR_ORDER):
        values = np.roll(pattern, model_index)
        predictions.extend(
            {"Model": model, "Specimen_ID": str(index), "Correct_Indirect": int(value)}
            for index, value in enumerate(values)
        )
        metrics.extend(
            {"Model": model, "Fold": fold, "Accuracy": float(values.mean())}
            for fold in range(5)
        )
    omnibus, pairwise, summary = indirect_statistics(
        pd.DataFrame(predictions), pd.DataFrame(metrics)
    )
    assert not bool(omnibus.loc[0, "Significant"])
    assert pairwise.empty
    assert not bool(summary.loc[0, "Post_Hoc_Performed"])
