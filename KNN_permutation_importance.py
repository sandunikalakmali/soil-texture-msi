"""Figure 16: grouped specimen-pattern permutation importance for KNN.

One held-out wavelength is permuted by exchanging its complete 100-block
pattern between physical specimens. Training data and every fitted scaler/LDA
remain unchanged. The original result-generating script used 10 deterministic
permutations per band per fold; that otherwise-unpublished detail is retained
and is made explicit in outputs and README for reproducibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

from Direct_Soil_Classification import apply_training_smote
from scientific_utils import (
    EXPECTED_DATASETS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    N_SPLITS,
    RANDOM_SEED,
    TARGET_COLUMNS,
    FoldPreprocessor,
    aggregate_prediction_to_specimen,
    aggregate_probabilities_to_specimen,
    assert_complete_oof,
    assert_no_group_leakage,
    composition_group_labels,
    grouped_specimen_truth,
    specimen_first,
    specimen_ids_from_dataframe,
    validate_block_dataset,
)
from usda_texture import USDA_CLASS_ORDER


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = (
    SCRIPT_DIR / "Data" / "Data_files"
    / "dataAll_Soil_Composition_Training_Histogram_No_Augmentation_"
    "Adjusted_Files_1crop_0_03_sigma_subOperator_SuperPixel.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "04_Permutation_Band_Importance"
N_PERMUTATION_REPEATS = 10


def exchange_specimen_band_patterns(
    features: np.ndarray,
    groups: np.ndarray,
    band_index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exchange one full wavelength pattern between same-sized specimens."""
    permuted = features.copy()
    specimen_ids = np.sort(np.unique(groups))
    source_ids = rng.permutation(specimen_ids)
    for destination, source in zip(specimen_ids, source_ids):
        destination_rows = np.flatnonzero(groups == destination)
        source_rows = np.flatnonzero(groups == source)
        if len(destination_rows) != len(source_rows):
            raise ValueError("Specimen-pattern permutation requires equal block counts.")
        permuted[destination_rows, band_index] = features[source_rows, band_index]
    return permuted


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_regression_importance(
    frame: pd.DataFrame, features: np.ndarray, groups: np.ndarray, splits
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    composition_labels = composition_group_labels(frame)
    all_compositions = set(composition_labels)
    truth = grouped_specimen_truth(frame, groups)
    baseline_records = []
    permutation_records = []
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        train_groups, test_groups = groups[train_index], groups[test_index]
        assert_no_group_leakage(train_groups, test_groups, fold)
        missing_compositions = all_compositions - set(composition_labels[train_index])
        if missing_compositions:
            raise AssertionError(
                f"Fold {fold}: regression-importance training specimens omit "
                f"composition groups: {sorted(missing_compositions)}"
            )
        preprocessor = FoldPreprocessor("composition group")
        x_train = preprocessor.fit_transform(features[train_index], composition_labels[train_index])
        x_test = preprocessor.transform(features[test_index])
        model = KNeighborsRegressor(
            n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30,
            p=2, metric="minkowski",
        ).fit(x_train, targets[train_index])
        specimen_ids = np.sort(np.unique(test_groups))
        y_true = truth.loc[specimen_ids, list(TARGET_COLUMNS)].to_numpy(dtype=float)
        # Raw specimen averages: no Eq. (14) in regression importance.
        baseline_predictions = aggregate_prediction_to_specimen(
            model.predict(x_test), test_groups
        ).loc[specimen_ids].to_numpy(dtype=float)
        baseline = rmse(y_true, baseline_predictions)
        baseline_records.append({
            "Fold": fold, "Baseline_RMSE": baseline,
            "Test_Specimens": len(specimen_ids),
            "LDA_Components": preprocessor.n_components_,
        })
        for band_index, wavelength in enumerate(FEATURE_COLUMNS):
            for repeat in range(1, N_PERMUTATION_REPEATS + 1):
                rng = np.random.default_rng(
                    RANDOM_SEED + fold * 10_000 + band_index * 100 + repeat
                )
                permuted_raw = exchange_specimen_band_patterns(
                    features[test_index], test_groups, band_index, rng
                )
                permuted_predictions = aggregate_prediction_to_specimen(
                    model.predict(preprocessor.transform(permuted_raw)), test_groups
                ).loc[specimen_ids].to_numpy(dtype=float)
                permuted_score = rmse(y_true, permuted_predictions)
                item = {
                    "Fold": fold, "Repeat": repeat, "Wavelength_nm": int(wavelength),
                    "Baseline_RMSE": baseline, "Permuted_RMSE": permuted_score,
                    "RMSE_Increase": permuted_score - baseline,
                }
                for component_index, component in enumerate(TARGET_COLUMNS):
                    base_component = rmse(y_true[:, component_index], baseline_predictions[:, component_index])
                    perm_component = rmse(y_true[:, component_index], permuted_predictions[:, component_index])
                    item[f"{component}_RMSE_Increase"] = perm_component - base_component
                permutation_records.append(item)
    return pd.DataFrame(baseline_records), pd.DataFrame(permutation_records)


def run_classification_importance(
    frame: pd.DataFrame, features: np.ndarray, groups: np.ndarray, splits
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoder = LabelEncoder().fit(list(USDA_CLASS_ORDER))
    encoded = encoder.transform(frame[LABEL_COLUMN].astype(str))
    all_encoded = encoder.transform(list(USDA_CLASS_ORDER))
    baseline_records = []
    permutation_records = []
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        train_groups, test_groups = groups[train_index], groups[test_index]
        assert_no_group_leakage(train_groups, test_groups, fold)
        preprocessor = FoldPreprocessor("USDA texture class")
        x_train = preprocessor.fit_transform(features[train_index], encoded[train_index])
        x_test = preprocessor.transform(features[test_index])
        x_train, y_train = apply_training_smote(x_train, encoded[train_index])
        model = KNeighborsClassifier(
            n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30,
            p=2, metric="minkowski",
        ).fit(x_train, y_train)
        true = specimen_first(encoded[test_index], test_groups)
        specimen_ids = true.index.to_numpy()

        def specimen_predictions(transformed: np.ndarray) -> np.ndarray:
            probabilities = aggregate_probabilities_to_specimen(
                model.predict_proba(transformed), test_groups, model.classes_, all_encoded
            )
            return probabilities.loc[specimen_ids].idxmax(axis=1).to_numpy(dtype=int)

        baseline_predictions = specimen_predictions(x_test)
        baseline = float(accuracy_score(true.to_numpy(dtype=int), baseline_predictions))
        baseline_records.append({
            "Fold": fold, "Baseline_Accuracy": baseline,
            "Test_Specimens": len(specimen_ids), "LDA_Components": preprocessor.n_components_,
        })
        for band_index, wavelength in enumerate(FEATURE_COLUMNS):
            for repeat in range(1, N_PERMUTATION_REPEATS + 1):
                rng = np.random.default_rng(
                    RANDOM_SEED + fold * 20_000 + band_index * 100 + repeat
                )
                permuted_raw = exchange_specimen_band_patterns(
                    features[test_index], test_groups, band_index, rng
                )
                permuted_predictions = specimen_predictions(preprocessor.transform(permuted_raw))
                permuted_score = float(
                    accuracy_score(true.to_numpy(dtype=int), permuted_predictions)
                )
                permutation_records.append({
                    "Fold": fold, "Repeat": repeat, "Wavelength_nm": int(wavelength),
                    "Baseline_Accuracy": baseline, "Permuted_Accuracy": permuted_score,
                    "Accuracy_Drop": baseline - permuted_score,
                })
    return pd.DataFrame(baseline_records), pd.DataFrame(permutation_records)


def save_figure(regression: pd.DataFrame, classification: pd.DataFrame, path: Path) -> None:
    regression = regression.sort_values("Wavelength_nm")
    classification = classification.sort_values("Wavelength_nm")
    wavelengths = classification["Wavelength_nm"].to_numpy()
    width = float(np.median(np.diff(wavelengths)) * 0.25)
    figure, axes = plt.subplots(2, 1, figsize=(10, 9))
    axes[0].bar(wavelengths, classification["Mean_Accuracy_Drop"], width=width, color="tab:blue")
    axes[0].set_ylabel("Mean specimen accuracy drop")
    axes[0].set_title("(a) Direct KNN classification")
    axes[1].bar(wavelengths, regression["Mean_RMSE_Increase"], width=width, color="tab:red")
    axes[1].set_ylabel("Mean raw-composition RMSE increase")
    axes[1].set_title("(b) KNN composition regression")
    for axis in axes:
        axis.set_xlabel("Wavelength (nm)")
        axis.set_xticks(wavelengths)
        axis.tick_params(axis="x", rotation=90)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run(train_csv: Path, output_dir: Path, strict_dimensions: bool = True) -> None:
    frame = pd.read_csv(train_csv).reset_index(drop=True)
    groups = specimen_ids_from_dataframe(frame, dataset_name="training/testing")
    validate_block_dataset(
        frame, groups, dataset_name="training/testing",
        expected=EXPECTED_DATASETS["training/testing"] if strict_dimensions else None,
    )
    features = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    labels = frame[LABEL_COLUMN].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(splitter.split(features, labels, groups=groups))
    assert_complete_oof([groups[test] for _, test in splits], groups)
    regression_baseline, regression_repeats = run_regression_importance(
        frame, features, groups, splits
    )
    classification_baseline, classification_repeats = run_classification_importance(
        frame, features, groups, splits
    )
    regression_summary = regression_repeats.groupby("Wavelength_nm", as_index=False).agg(
        Mean_RMSE_Increase=("RMSE_Increase", "mean"),
        Std_RMSE_Increase=("RMSE_Increase", "std"),
        Mean_Clay_RMSE_Increase=("Clay_RMSE_Increase", "mean"),
        Mean_Silt_RMSE_Increase=("Silt_RMSE_Increase", "mean"),
        Mean_Sand_RMSE_Increase=("Sand_RMSE_Increase", "mean"),
    )
    classification_summary = classification_repeats.groupby("Wavelength_nm", as_index=False).agg(
        Mean_Accuracy_Drop=("Accuracy_Drop", "mean"),
        Std_Accuracy_Drop=("Accuracy_Drop", "std"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    regression_baseline.to_csv(output_dir / "regression_baseline_by_fold.csv", index=False)
    classification_baseline.to_csv(output_dir / "classification_baseline_by_fold.csv", index=False)
    regression_repeats.to_csv(output_dir / "regression_permutation_repeats.csv", index=False)
    classification_repeats.to_csv(output_dir / "classification_permutation_repeats.csv", index=False)
    regression_summary.to_csv(output_dir / "regression_permutation_importance.csv", index=False)
    classification_summary.to_csv(output_dir / "classification_permutation_importance.csv", index=False)
    save_figure(
        regression_summary, classification_summary,
        output_dir / "Figure_16_KNN_Spectral_Band_Importance.png",
    )
    print(f"Saved permutation-importance outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-nonofficial-dimensions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.train_csv, arguments.output_dir, not arguments.allow_nonofficial_dimensions)
