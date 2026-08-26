"""Benchmark paper-style KNN model-level inference on external specimens.

Timed direct inference includes ``predict_proba`` for 100 already-transformed
blocks, probability averaging, and argmax. Timed indirect regression includes
``predict`` and block averaging; USDA Eq. (14) projection plus triangle mapping
is timed separately. Disk I/O, preprocessing, model fitting, and plotting are
excluded. Results depend on hardware and installed package versions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

from Direct_Soil_Classification import apply_training_smote
from scientific_utils import (
    EXPECTED_DATASETS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    RANDOM_SEED,
    TARGET_COLUMNS,
    FoldPreprocessor,
    composition_group_labels,
    specimen_ids_from_dataframe,
    validate_block_dataset,
)
from usda_texture import USDA_CLASS_ORDER, classify_usda_texture, project_to_compositional_simplex


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data" / "Data_files"
DEFAULT_TRAIN_CSV = DATA_DIR / (
    "dataAll_Soil_Composition_Training_Histogram_No_Augmentation_"
    "Adjusted_Files_1crop_0_03_sigma_subOperator_SuperPixel.csv"
)
DEFAULT_VALIDATION_CSV = DATA_DIR / (
    "dataAll_Soil_Composition_Validation_Histogram_No_Augmentation_"
    "Adjusted_Files_1crop_0_03_sigma_subOperator_SuperPixel.csv"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "inference_benchmark.json"


def elapsed_ms(function, argument) -> float:
    start = time.perf_counter_ns()
    function(argument)
    return (time.perf_counter_ns() - start) / 1_000_000.0


def run(train_csv: Path, validation_csv: Path, output: Path, repeats: int, warmups: int) -> dict:
    train = pd.read_csv(train_csv).reset_index(drop=True)
    validation = pd.read_csv(validation_csv).reset_index(drop=True)
    train_groups = specimen_ids_from_dataframe(train, dataset_name="training/testing")
    validation_groups = specimen_ids_from_dataframe(validation, dataset_name="external validation")
    validate_block_dataset(
        train, train_groups, dataset_name="training/testing",
        expected=EXPECTED_DATASETS["training/testing"],
    )
    validate_block_dataset(
        validation, validation_groups, dataset_name="external validation",
        expected=EXPECTED_DATASETS["external validation"],
    )
    x_train = train[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    x_validation = validation[list(FEATURE_COLUMNS)].to_numpy(dtype=float)

    encoder = LabelEncoder().fit(list(USDA_CLASS_ORDER))
    direct_labels = encoder.transform(train[LABEL_COLUMN].astype(str))
    direct_preprocessor = FoldPreprocessor("USDA texture class").fit(x_train, direct_labels)
    direct_train = direct_preprocessor.transform(x_train)
    direct_validation = direct_preprocessor.transform(x_validation)
    direct_train, direct_labels = apply_training_smote(direct_train, direct_labels)
    direct_model = KNeighborsClassifier(
        n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30,
        p=2, metric="minkowski",
    ).fit(direct_train, direct_labels)

    composition_labels = composition_group_labels(train)
    regression_preprocessor = FoldPreprocessor("composition group").fit(
        x_train, composition_labels
    )
    regression_train = regression_preprocessor.transform(x_train)
    regression_validation = regression_preprocessor.transform(x_validation)
    regression_model = KNeighborsRegressor(
        n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30,
        p=2, metric="minkowski",
    ).fit(regression_train, train[list(TARGET_COLUMNS)].to_numpy(dtype=float))

    specimen_blocks = [
        np.flatnonzero(validation_groups == specimen_id)
        for specimen_id in np.sort(np.unique(validation_groups))
    ]

    def direct_inference(indices):
        return int(np.argmax(direct_model.predict_proba(direct_validation[indices]).mean(axis=0)))

    def regression_inference(indices):
        return regression_model.predict(regression_validation[indices]).mean(axis=0)

    def mapping_inference(raw_prediction):
        clay, silt, sand = project_to_compositional_simplex(raw_prediction)
        return classify_usda_texture(clay, silt, sand)

    for _ in range(warmups):
        for indices in specimen_blocks:
            direct_inference(indices)
            mapping_inference(regression_inference(indices))

    direct_times, regression_times, mapping_times = [], [], []
    for _ in range(repeats):
        for indices in specimen_blocks:
            direct_times.append(elapsed_ms(direct_inference, indices))
            raw = regression_inference(indices)
            regression_times.append(elapsed_ms(regression_inference, indices))
            mapping_times.append(elapsed_ms(mapping_inference, raw))

    result = {
        "definition": "model-level inference on one physical specimen (100 preprocessed/LDA blocks)",
        "excluded": ["disk I/O", "image preprocessing", "min-max transform", "LDA transform", "training", "plotting"],
        "benchmark_specimens": len(specimen_blocks),
        "repeats": repeats,
        "random_seed": RANDOM_SEED,
        "direct_knn_ms_per_specimen": float(np.mean(direct_times)),
        "indirect_knn_regression_ms_per_specimen": float(np.mean(regression_times)),
        "indirect_usda_mapping_ms_per_specimen": float(np.mean(mapping_times)),
        "indirect_total_ms_per_specimen": float(np.mean(regression_times) + np.mean(mapping_times)),
        "paper_hardware": "AMD Ryzen 7 6800HS, 16 GB RAM",
        "paper_reference_ms": {
            "direct_knn": 1.33,
            "indirect_knn_regression": 1.15,
            "indirect_usda_mapping": 0.009,
            "indirect_total": 1.159,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.train_csv, arguments.validation_csv, arguments.output, arguments.repeats, arguments.warmups)
