"""Soil composition regression and separate indirect USDA classification.

Regression metrics use raw specimen-level averages of 100 block predictions.
Only the indirect branch applies Eq. (14) before the canonical USDA mapping.
Figure 13 pools every held-out specimen across all five grouped folds; no
favorable-fold selection is performed. Figure 14 uses external-validation
predictions from models fitted on the complete training/testing dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, r2_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

from scientific_utils import (
    EXPECTED_DATASETS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    N_SPLITS,
    RANDOM_SEED,
    SPECIMEN_COLUMN,
    TARGET_COLUMNS,
    FoldPreprocessor,
    aggregate_prediction_to_specimen,
    assert_complete_oof,
    assert_no_group_leakage,
    composition_group_labels,
    grouped_specimen_truth,
    specimen_ids_from_dataframe,
    validate_block_dataset,
)
from usda_texture import USDA_CLASS_ORDER, classify_usda_rows, project_to_compositional_simplex


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
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
MODEL_ORDER = (
    "Linear Regression", "KNN", "Random Forest", "Decision Tree", "XGBoost", "CatBoost"
)


def slug(model_name: str) -> str:
    return model_name.replace(" ", "_")


def build_models() -> dict[str, object]:
    """Construct the six published regressors without changing original tuning."""
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("Install requirements.txt; CatBoost is a published model.") from exc
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install requirements.txt; XGBoost is a published model.") from exc
    # CatBoost/XGBoost parameters left unspecified here were likewise library
    # defaults in the original result-generating scripts. Historical package
    # versions are unavailable, so guessing explicit values would risk drift.
    return {
        "Linear Regression": LinearRegression(fit_intercept=True, positive=False),
        "KNN": KNeighborsRegressor(
            n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30,
            p=2, metric="minkowski",
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, max_depth=15, min_samples_leaf=5,
            min_samples_split=10, max_features="sqrt", bootstrap=True,
            n_jobs=-1, random_state=RANDOM_SEED,
        ),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_SEED),
        "XGBoost": MultiOutputRegressor(XGBRegressor(
            objective="reg:squarederror", random_state=RANDOM_SEED, n_jobs=-1,
        )),
        "CatBoost": MultiOutputRegressor(CatBoostRegressor(
            loss_function="RMSE", random_seed=RANDOM_SEED,
            verbose=False, allow_writing_files=False,
        )),
    }


def regression_metrics(y_true: np.ndarray, y_pred_raw: np.ndarray) -> dict[str, float]:
    """Calculate paper R2/RMSE from raw averaged specimen predictions."""
    row: dict[str, float] = {}
    for index, component in enumerate(TARGET_COLUMNS):
        row[f"{component}_R2"] = float(r2_score(y_true[:, index], y_pred_raw[:, index]))
        row[f"{component}_RMSE"] = float(
            np.sqrt(mean_squared_error(y_true[:, index], y_pred_raw[:, index]))
        )
    errors = np.sqrt(np.mean((y_true - y_pred_raw) ** 2, axis=1))
    row["Mean_Specimen_Composition_RMSE"] = float(errors.mean())
    return row


def indirect_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1_score(
            y_true, y_pred, labels=USDA_CLASS_ORDER, average="macro", zero_division=0
        )),
        "Macro_Recall": float(recall_score(
            y_true, y_pred, labels=USDA_CLASS_ORDER, average="macro", zero_division=0
        )),
    }


def prediction_records(
    model_name: str,
    fold: int | str,
    specimen_ids: np.ndarray,
    truth: pd.DataFrame,
    raw_predictions: np.ndarray,
) -> list[dict[str, object]]:
    """Keep raw regression and Eq. (14) indirect values explicitly separate."""
    simplex = project_to_compositional_simplex(raw_predictions)
    predicted_textures = classify_usda_rows(simplex)
    records = []
    for index, specimen_id in enumerate(specimen_ids):
        item: dict[str, object] = {
            "Model": model_name,
            "Fold": fold,
            SPECIMEN_COLUMN: specimen_id,
            "True_Texture": truth.loc[specimen_id, LABEL_COLUMN],
            "Predicted_Texture_Indirect": predicted_textures[index],
        }
        for component_index, component in enumerate(TARGET_COLUMNS):
            item[f"True_{component}"] = truth.loc[specimen_id, component]
            item[f"Pred_{component}_Raw"] = raw_predictions[index, component_index]
            item[f"Pred_{component}_Simplex"] = simplex[index, component_index]
        item["Correct_Indirect"] = int(item["True_Texture"] == item["Predicted_Texture_Indirect"])
        records.append(item)
    return records


def save_composition_plot(predictions: pd.DataFrame, path: Path, title: str) -> None:
    """Plot raw specimen regression estimates for clay, silt, and sand."""
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for panel, (axis, component) in enumerate(zip(axes, TARGET_COLUMNS), start=1):
        true = predictions[f"True_{component}"].to_numpy(dtype=float)
        predicted = predictions[f"Pred_{component}_Raw"].to_numpy(dtype=float)
        lower = min(true.min(), predicted.min())
        upper = max(true.max(), predicted.max())
        axis.scatter(true, predicted, alpha=0.55)
        axis.plot([lower, upper], [lower, upper], "r--", linewidth=1)
        axis.set_xlabel(f"True {component.lower()} (%)")
        axis.set_ylabel(f"Predicted {component.lower()} (%)")
        axis.set_title(f"({chr(96 + panel)})")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_indirect_confusion(predictions: pd.DataFrame, path: Path) -> None:
    matrix = confusion_matrix(
        predictions["True_Texture"], predictions["Predicted_Texture_Indirect"],
        labels=USDA_CLASS_ORDER, normalize="true",
    )
    figure, axis = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="Blues", square=True,
        xticklabels=USDA_CLASS_ORDER, yticklabels=USDA_CLASS_ORDER, ax=axis,
    )
    axis.set_xlabel("Predicted USDA soil texture class")
    axis.set_ylabel("True USDA soil texture class")
    axis.set_title("KNN indirect classification: pooled out-of-fold specimens")
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def regression_statistics(errors: pd.DataFrame, alpha: float = 0.05):
    """Friedman omnibus test followed conditionally by Wilcoxon/Holm tests."""
    matrix = errors.pivot(index=SPECIMEN_COLUMN, columns="Model", values="Composition_Error")
    matrix = matrix.sort_index().loc[:, list(MODEL_ORDER)]
    if matrix.isna().any().any():
        raise AssertionError("Regression paired tests require all specimens for all models.")
    means = matrix.mean().sort_values()
    best = str(means.index[0])
    statistic, p_value = friedmanchisquare(*[matrix[model] for model in MODEL_ORDER])
    significant = bool(p_value < alpha)
    omnibus = pd.DataFrame([{
        "Test": "Friedman", "Unit": "physical specimen",
        "Error": "sqrt(mean squared error across clay, silt, sand)",
        "Statistic": float(statistic), "p_value": float(p_value),
        "alpha": alpha, "Significant": significant,
    }])
    records = []
    if significant:
        for other in MODEL_ORDER:
            if other == best:
                continue
            try:
                result = wilcoxon(
                    matrix[best], matrix[other], alternative="less", zero_method="wilcox"
                )
                test_statistic, test_p = float(result.statistic), float(result.pvalue)
            except ValueError:
                test_statistic, test_p = np.nan, 1.0
            records.append({
                "Lowest_RMSE_Model": best, "Other_Model": other,
                "Alternative": f"{best} error < {other} error",
                "Statistic": test_statistic, "Raw_p_value": test_p,
            })
    pairwise = pd.DataFrame(records, columns=[
        "Lowest_RMSE_Model", "Other_Model", "Alternative", "Statistic", "Raw_p_value"
    ])
    if not pairwise.empty:
        reject, adjusted, _, _ = multipletests(pairwise["Raw_p_value"], alpha=alpha, method="holm")
        pairwise["Holm_Adjusted_p_value"] = adjusted
        pairwise["Significant_After_Holm"] = reject
    else:
        pairwise["Holm_Adjusted_p_value"] = pd.Series(dtype=float)
        pairwise["Significant_After_Holm"] = pd.Series(dtype=bool)
    selection = pd.DataFrame([{
        "Lowest_Mean_Error_Model": best, "Mean_Specimen_Composition_Error": float(means.iloc[0]),
        "Friedman_Significant": significant, "Post_Hoc_Performed": significant,
    }])
    return omnibus, pairwise, selection


def indirect_statistics(predictions: pd.DataFrame, fold_metrics: pd.DataFrame, alpha: float = 0.05):
    """Cochran Q followed conditionally by best-vs-rest McNemar/Holm tests."""
    means = fold_metrics.groupby("Model")["Accuracy"].mean().sort_values(ascending=False)
    best = str(means.index[0])
    matrix = predictions.pivot(
        index=SPECIMEN_COLUMN, columns="Model", values="Correct_Indirect"
    ).sort_index().loc[:, list(MODEL_ORDER)]
    if matrix.isna().any().any():
        raise AssertionError("Indirect paired tests require all specimens for all models.")
    result = cochrans_q(matrix.to_numpy(dtype=int))
    significant = bool(result.pvalue < alpha)
    omnibus = pd.DataFrame([{
        "Test": "Cochran Q", "Unit": "physical specimen correctness",
        "Statistic": float(result.statistic), "p_value": float(result.pvalue),
        "alpha": alpha, "Significant": significant,
    }])
    records = []
    if significant:
        best_correct = matrix[best].to_numpy(dtype=int)
        for other in MODEL_ORDER:
            if other == best:
                continue
            other_correct = matrix[other].to_numpy(dtype=int)
            b = int(np.sum((best_correct == 0) & (other_correct == 1)))
            c = int(np.sum((best_correct == 1) & (other_correct == 0)))
            exact = b + c < 25
            table = [[np.sum((best_correct == 0) & (other_correct == 0)), b],
                     [c, np.sum((best_correct == 1) & (other_correct == 1))]]
            test = mcnemar(table, exact=exact, correction=not exact)
            records.append({
                "Highest_Accuracy_Model": best, "Other_Model": other,
                "Best_Wrong_Other_Correct": b, "Best_Correct_Other_Wrong": c,
                "Exact_McNemar": exact, "Statistic": float(test.statistic),
                "Raw_p_value": float(test.pvalue),
            })
    pairwise = pd.DataFrame(records, columns=[
        "Highest_Accuracy_Model", "Other_Model", "Best_Wrong_Other_Correct",
        "Best_Correct_Other_Wrong", "Exact_McNemar", "Statistic", "Raw_p_value",
    ])
    if not pairwise.empty:
        reject, adjusted, _, _ = multipletests(pairwise["Raw_p_value"], alpha=alpha, method="holm")
        pairwise["Holm_Adjusted_p_value"] = adjusted
        pairwise["Significant_After_Holm"] = reject
    else:
        pairwise["Holm_Adjusted_p_value"] = pd.Series(dtype=float)
        pairwise["Significant_After_Holm"] = pd.Series(dtype=bool)
    selection = pd.DataFrame([{
        "Highest_Accuracy_Model": best, "Mean_CV_Accuracy": float(means.iloc[0]),
        "Cochran_Q_Significant": significant, "Post_Hoc_Performed": significant,
    }])
    return omnibus, pairwise, selection


def run(train_csv: Path, validation_csv: Path, output_root: Path, strict_dimensions: bool = True) -> None:
    train = pd.read_csv(train_csv).reset_index(drop=True)
    validation = pd.read_csv(validation_csv).reset_index(drop=True)
    train_groups = specimen_ids_from_dataframe(train, dataset_name="training/testing")
    validation_groups = specimen_ids_from_dataframe(validation, dataset_name="external validation")
    validate_block_dataset(
        train, train_groups, dataset_name="training/testing",
        expected=EXPECTED_DATASETS["training/testing"] if strict_dimensions else None,
    )
    validate_block_dataset(
        validation, validation_groups, dataset_name="external validation",
        expected=EXPECTED_DATASETS["external validation"] if strict_dimensions else None,
    )
    if set(train_groups).intersection(validation_groups):
        raise AssertionError("Training/testing and external-validation specimen IDs overlap.")

    x = train[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = train[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    x_validation = validation[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    composition_labels = composition_group_labels(train)
    all_compositions = set(composition_labels)
    texture_labels = train[LABEL_COLUMN].astype(str).to_numpy()
    train_truth = grouped_specimen_truth(train, train_groups)
    validation_truth = grouped_specimen_truth(validation, validation_groups)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(splitter.split(x, texture_labels, groups=train_groups))
    assert_complete_oof([train_groups[test] for _, test in splits], train_groups)

    fold_metric_records = []
    indirect_metric_records = []
    all_predictions = []
    error_records = []
    lda_records = []
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        fold_train_groups, fold_test_groups = train_groups[train_index], train_groups[test_index]
        assert_no_group_leakage(fold_train_groups, fold_test_groups, fold)
        fold_compositions = set(composition_labels[train_index])
        if fold_compositions != all_compositions:
            missing = sorted(all_compositions - fold_compositions)
            raise AssertionError(f"Fold {fold}: training specimens omit composition groups: {missing}")
        preprocessor = FoldPreprocessor("composition group")
        x_train = preprocessor.fit_transform(x[train_index], composition_labels[train_index])
        x_test = preprocessor.transform(x[test_index])
        lda_records.append({
            "Dataset": "grouped CV", "Fold": fold,
            "Supervision": "composition group", "Composition_Groups_In_Training": len(fold_compositions),
            "Power_Threshold": preprocessor.power_to_keep,
            "Components_Retained": preprocessor.n_components_,
        })
        specimen_ids = np.sort(np.unique(fold_test_groups))
        truth = train_truth.loc[specimen_ids]
        y_true = truth[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        for model_name, model in build_models().items():
            model.fit(x_train, y[train_index])
            raw = aggregate_prediction_to_specimen(
                model.predict(x_test), fold_test_groups
            ).loc[specimen_ids].to_numpy(dtype=float)
            metrics = regression_metrics(y_true, raw)
            metrics.update({"Model": model_name, "Fold": fold})
            fold_metric_records.append(metrics)
            records = prediction_records(model_name, fold, specimen_ids, truth, raw)
            all_predictions.extend(records)
            true_textures = truth[LABEL_COLUMN].to_numpy(dtype=str)
            predicted_textures = np.asarray([item["Predicted_Texture_Indirect"] for item in records])
            cls = indirect_metrics(true_textures, predicted_textures)
            cls.update({"Model": model_name, "Fold": fold})
            indirect_metric_records.append(cls)
            errors = np.sqrt(np.mean((y_true - raw) ** 2, axis=1))
            error_records.extend({
                "Model": model_name, "Fold": fold, SPECIMEN_COLUMN: specimen_id,
                "Composition_Error": error,
            } for specimen_id, error in zip(specimen_ids, errors))

    fold_metrics = pd.DataFrame(fold_metric_records)
    indirect_fold_metrics = pd.DataFrame(indirect_metric_records)
    oof_predictions = pd.DataFrame(all_predictions)
    errors = pd.DataFrame(error_records)

    # External validation: every fitted transformation and model sees only the
    # complete 440-specimen training/testing dataset, never validation data.
    external_preprocessor = FoldPreprocessor("composition group")
    x_all = external_preprocessor.fit_transform(x, composition_labels)
    x_external = external_preprocessor.transform(x_validation)
    lda_records.append({
        "Dataset": "external validation", "Fold": "all training/testing specimens",
        "Supervision": "composition group", "Composition_Groups_In_Training": len(all_compositions),
        "Power_Threshold": external_preprocessor.power_to_keep,
        "Components_Retained": external_preprocessor.n_components_,
    })
    external_ids = np.sort(np.unique(validation_groups))
    external_truth = validation_truth.loc[external_ids]
    external_metric_records = []
    external_prediction_records = []
    for model_name, model in build_models().items():
        model.fit(x_all, y)
        raw = aggregate_prediction_to_specimen(
            model.predict(x_external), validation_groups
        ).loc[external_ids].to_numpy(dtype=float)
        metrics = regression_metrics(
            external_truth[list(TARGET_COLUMNS)].to_numpy(dtype=float), raw
        )
        metrics["Model"] = model_name
        external_metric_records.append(metrics)
        external_prediction_records.extend(
            prediction_records(model_name, "external validation", external_ids, external_truth, raw)
        )
    external_metrics = pd.DataFrame(external_metric_records)
    external_predictions = pd.DataFrame(external_prediction_records)

    regression_dir = output_root / "02_Soil_Composition_Regression"
    indirect_dir = output_root / "03_Indirect_Soil_Classification"
    regression_dir.mkdir(parents=True, exist_ok=True)
    indirect_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(regression_dir / "cross_validation_fold_metrics.csv", index=False)
    external_metrics.to_csv(regression_dir / "external_validation_metrics.csv", index=False)
    pd.DataFrame(lda_records).to_csv(regression_dir / "lda_diagnostics.csv", index=False)
    fold_metrics.groupby("Model")[[f"{c}_{m}" for c in TARGET_COLUMNS for m in ("R2", "RMSE")]].agg(
        ["mean", "std"]
    ).to_csv(regression_dir / "cross_validation_summary.csv")
    for model_name in MODEL_ORDER:
        model_oof = oof_predictions[oof_predictions["Model"] == model_name].sort_values(SPECIMEN_COLUMN)
        model_external = external_predictions[external_predictions["Model"] == model_name].sort_values(SPECIMEN_COLUMN)
        model_oof.to_csv(regression_dir / f"testing_out_of_fold_{slug(model_name)}.csv", index=False)
        model_external.to_csv(regression_dir / f"external_validation_{slug(model_name)}.csv", index=False)

    regression_omnibus, regression_pairwise, regression_selection = regression_statistics(errors)
    regression_omnibus.to_csv(regression_dir / "friedman_test.csv", index=False)
    regression_pairwise.to_csv(regression_dir / "conditional_wilcoxon_holm_tests.csv", index=False)
    regression_selection.to_csv(regression_dir / "statistical_summary.csv", index=False)

    indirect_fold_metrics.to_csv(indirect_dir / "fold_metrics.csv", index=False)
    indirect_fold_metrics.groupby("Model")[["Accuracy", "Macro_F1", "Macro_Recall"]].agg(
        ["mean", "std"]
    ).to_csv(indirect_dir / "summary_metrics.csv")
    oof_predictions[[
        "Model", "Fold", SPECIMEN_COLUMN, "True_Texture",
        "Predicted_Texture_Indirect", "Correct_Indirect",
    ]].to_csv(indirect_dir / "out_of_fold_specimen_predictions.csv", index=False)
    indirect_omnibus, indirect_pairwise, indirect_selection = indirect_statistics(
        oof_predictions, indirect_fold_metrics
    )
    indirect_omnibus.to_csv(indirect_dir / "cochran_q_test.csv", index=False)
    indirect_pairwise.to_csv(indirect_dir / "conditional_mcnemar_holm_tests.csv", index=False)
    indirect_selection.to_csv(indirect_dir / "statistical_summary.csv", index=False)

    knn_oof = oof_predictions[oof_predictions["Model"] == "KNN"].sort_values(SPECIMEN_COLUMN)
    knn_external = external_predictions[external_predictions["Model"] == "KNN"].sort_values(SPECIMEN_COLUMN)
    if strict_dimensions and (len(knn_oof) != 440 or len(knn_external) != 84):
        raise AssertionError("Figures 13/14 require 440 OOF and 84 external specimens.")
    save_composition_plot(
        knn_oof, regression_dir / "Figure_13_KNN_Out_of_Fold_Composition.png",
        "KNN grouped-CV out-of-fold predictions",
    )
    save_composition_plot(
        knn_external, regression_dir / "Figure_14_KNN_External_Validation_Composition.png",
        "KNN external-validation predictions",
    )
    save_indirect_confusion(
        knn_oof, indirect_dir / "Figure_15_KNN_Indirect_Classification_Confusion_Matrix.png"
    )
    print(f"Saved regression outputs to {regression_dir}")
    print(f"Saved indirect-classification outputs to {indirect_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--allow-nonofficial-dimensions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.train_csv, arguments.validation_csv, arguments.output_root,
        not arguments.allow_nonofficial_dimensions,
    )
