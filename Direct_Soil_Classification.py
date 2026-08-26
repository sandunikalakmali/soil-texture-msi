"""Paper-aligned direct USDA soil texture classification.

Five-fold stratified grouped cross-validation is performed at physical-
specimen level. Scaling and USDA-class-supervised LDA are fitted on each
training fold only, followed by training-only SMOTE. Final predictions and all
metrics are specimen-level averages of the 100 block probability vectors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

from scientific_utils import (
    EXPECTED_DATASETS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    N_SPLITS,
    RANDOM_SEED,
    SPECIMEN_COLUMN,
    FoldPreprocessor,
    aggregate_probabilities_to_specimen,
    assert_complete_oof,
    assert_no_group_leakage,
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "01_Direct_Soil_Classification"
MODEL_ORDER = (
    "KNN", "Random Forest", "Decision Tree", "CatBoost", "XGBoost",
    "Logistic Regression",
)


def build_models(number_of_classes: int) -> dict[str, object]:
    """Construct the six published classifiers with explicit parameters."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("Install requirements.txt; CatBoost is a published model.") from exc
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install requirements.txt; XGBoost is a published model.") from exc
    return {
        "KNN": KNeighborsClassifier(
            n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30,
            p=2, metric="minkowski",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, criterion="gini", max_depth=None,
            min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
            bootstrap=True, random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=None,
            min_samples_split=2, min_samples_leaf=1, random_state=RANDOM_SEED,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            loss_function="MultiClass", random_seed=RANDOM_SEED,
            verbose=False, allow_writing_files=False,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", num_class=number_of_classes,
            eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=5000,
            random_state=RANDOM_SEED,
        ),
    }


def apply_training_smote(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE after splitting/scaling/LDA, and only to a training fold."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError("Install requirements.txt; direct classification uses SMOTE.") from exc
    minimum_count = int(np.unique(labels, return_counts=True)[1].min())
    if minimum_count < 2:
        raise ValueError("SMOTE requires at least two training observations per class.")
    return SMOTE(
        random_state=RANDOM_SEED, k_neighbors=min(5, minimum_count - 1)
    ).fit_resample(features, labels)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = list(USDA_CLASS_ORDER)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "Macro_Recall": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
    }


def save_normalized_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=USDA_CLASS_ORDER, normalize="true")
    figure, axis = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="Blues", square=True,
        xticklabels=USDA_CLASS_ORDER, yticklabels=USDA_CLASS_ORDER, ax=axis,
    )
    axis.set_xlabel("Predicted USDA soil texture class")
    axis.set_ylabel("True USDA soil texture class")
    axis.set_title("KNN direct classification: pooled out-of-fold specimens")
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_classification_statistics(
    predictions: pd.DataFrame, fold_metrics: pd.DataFrame, *, alpha: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Cochran Q, then conditional best-vs-rest McNemar/Holm tests."""
    mean_accuracy = fold_metrics.groupby("Model")["Accuracy"].mean().sort_values(ascending=False)
    best_model = str(mean_accuracy.index[0])
    correctness = (
        predictions.pivot(index=SPECIMEN_COLUMN, columns="Model", values="Correct")
        .sort_index().loc[:, list(MODEL_ORDER)]
    )
    if correctness.isna().any().any():
        raise AssertionError("Paired tests require every specimen for every model.")
    correctness = correctness.astype(int)
    result = cochrans_q(correctness.to_numpy())
    significant = bool(result.pvalue < alpha)
    omnibus = pd.DataFrame([{
        "Test": "Cochran Q", "Unit": "physical specimen correctness",
        "Number_of_Specimens": len(correctness), "Number_of_Models": len(MODEL_ORDER),
        "Statistic": float(result.statistic), "p_value": float(result.pvalue),
        "alpha": alpha, "Significant": significant,
    }])
    records: list[dict[str, object]] = []
    if significant:
        best = correctness[best_model].to_numpy()
        for other_model in MODEL_ORDER:
            if other_model == best_model:
                continue
            other = correctness[other_model].to_numpy()
            b = int(np.sum((best == 0) & (other == 1)))
            c = int(np.sum((best == 1) & (other == 0)))
            discordant = b + c
            exact = discordant < 25
            table = np.asarray([
                [np.sum((best == 0) & (other == 0)), b],
                [c, np.sum((best == 1) & (other == 1))],
            ], dtype=int)
            test = mcnemar(table, exact=exact, correction=not exact)
            records.append({
                "Best_Model": best_model, "Other_Model": other_model,
                "Best_Wrong_Other_Correct": b, "Best_Correct_Other_Wrong": c,
                "Discordant_Pairs": discordant, "Exact_McNemar": exact,
                "Statistic": float(test.statistic), "Raw_p_value": float(test.pvalue),
            })
    pairwise_columns = [
        "Best_Model", "Other_Model", "Best_Wrong_Other_Correct",
        "Best_Correct_Other_Wrong", "Discordant_Pairs", "Exact_McNemar",
        "Statistic", "Raw_p_value",
    ]
    pairwise = pd.DataFrame(records, columns=pairwise_columns)
    if not pairwise.empty:
        reject, adjusted, _, _ = multipletests(
            pairwise["Raw_p_value"].to_numpy(), alpha=alpha, method="holm"
        )
        pairwise["Holm_Adjusted_p_value"] = adjusted
        pairwise["Significant_After_Holm"] = reject
    else:
        pairwise["Holm_Adjusted_p_value"] = pd.Series(dtype=float)
        pairwise["Significant_After_Holm"] = pd.Series(dtype=bool)
    summary = pd.DataFrame([{
        "Highest_Accuracy_Model": best_model,
        "Mean_CV_Accuracy": float(mean_accuracy.iloc[0]),
        "Cochran_Q_Significant": significant,
        "Post_Hoc_Performed": significant,
    }])
    return omnibus, pairwise, summary


def run(train_csv: Path, output_dir: Path, strict_dimensions: bool = True) -> None:
    frame = pd.read_csv(train_csv).reset_index(drop=True)
    groups = specimen_ids_from_dataframe(frame, dataset_name="training/testing")
    validate_block_dataset(
        frame, groups, dataset_name="training/testing",
        expected=EXPECTED_DATASETS["training/testing"] if strict_dimensions else None,
    )
    features = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    labels = frame[LABEL_COLUMN].astype(str).to_numpy()
    encoder = LabelEncoder().fit(list(USDA_CLASS_ORDER))
    encoded = encoder.transform(labels)
    all_encoded = encoder.transform(list(USDA_CLASS_ORDER))
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(splitter.split(features, encoded, groups=groups))
    assert_complete_oof([groups[test] for _, test in splits], groups)

    metric_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []
    lda_records: list[dict[str, object]] = []
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        train_groups, test_groups = groups[train_index], groups[test_index]
        assert_no_group_leakage(train_groups, test_groups, fold)
        preprocessor = FoldPreprocessor("USDA texture class")
        train_features = preprocessor.fit_transform(features[train_index], encoded[train_index])
        test_features = preprocessor.transform(features[test_index])
        balanced_features, balanced_labels = apply_training_smote(train_features, encoded[train_index])
        lda_records.append({
            "Fold": fold, "Supervision": "USDA texture class",
            "Power_Threshold": preprocessor.power_to_keep,
            "Components_Retained": preprocessor.n_components_,
        })
        true_by_specimen = specimen_first(encoded[test_index], test_groups)
        for model_name, model in build_models(len(USDA_CLASS_ORDER)).items():
            model.fit(balanced_features, balanced_labels)
            probabilities = model.predict_proba(test_features)
            specimen_probabilities = aggregate_probabilities_to_specimen(
                probabilities, test_groups, model.classes_, all_encoded
            )
            specimen_ids = specimen_probabilities.index.to_numpy()
            predicted_encoded = specimen_probabilities.idxmax(axis=1).to_numpy(dtype=int)
            true_encoded = true_by_specimen.loc[specimen_ids].to_numpy(dtype=int)
            true_names = encoder.inverse_transform(true_encoded)
            predicted_names = encoder.inverse_transform(predicted_encoded)
            record = classification_metrics(true_names, predicted_names)
            record.update({
                "Model": model_name, "Fold": fold,
                "Train_Specimens": len(np.unique(train_groups)),
                "Test_Specimens": len(specimen_ids),
                "LDA_Components": preprocessor.n_components_,
            })
            metric_records.append(record)
            prediction_records.extend({
                "Model": model_name, "Fold": fold, SPECIMEN_COLUMN: specimen_id,
                "True_Texture": true_name, "Predicted_Texture": predicted_name,
                "Correct": int(true_name == predicted_name),
            } for specimen_id, true_name, predicted_name in zip(
                specimen_ids, true_names, predicted_names
            ))

    metrics = pd.DataFrame(metric_records)
    predictions = pd.DataFrame(prediction_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary = metrics.groupby("Model")[["Accuracy", "Macro_F1", "Macro_Recall"]].agg(["mean", "std"])
    summary.to_csv(output_dir / "summary_metrics.csv")
    predictions.to_csv(output_dir / "out_of_fold_specimen_predictions.csv", index=False)
    pd.DataFrame(lda_records).to_csv(output_dir / "lda_diagnostics.csv", index=False)
    omnibus, pairwise, selection = run_classification_statistics(predictions, metrics)
    omnibus.to_csv(output_dir / "cochran_q_test.csv", index=False)
    pairwise.to_csv(output_dir / "conditional_mcnemar_holm_tests.csv", index=False)
    selection.to_csv(output_dir / "statistical_summary.csv", index=False)

    knn = predictions[predictions["Model"] == "KNN"].sort_values(SPECIMEN_COLUMN)
    if strict_dimensions and knn[SPECIMEN_COLUMN].nunique() != 440:
        raise AssertionError("Figure 12 must contain all 440 held-out specimens once.")
    save_normalized_confusion_matrix(
        knn["True_Texture"].to_numpy(), knn["Predicted_Texture"].to_numpy(),
        output_dir / "Figure_12_KNN_Direct_Classification_Confusion_Matrix.png",
    )
    print(summary)
    print(f"Saved direct-classification outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-nonofficial-dimensions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.train_csv, arguments.output_dir, not arguments.allow_nonofficial_dimensions)
