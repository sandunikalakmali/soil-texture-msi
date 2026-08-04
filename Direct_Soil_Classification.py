"""
Leakage-free grouped stratified cross-validation for soil texture classification.

What this script does when you run it:
1. Loads the training dataset from TRAIN_DATA_PATH.
2. Uses StratifiedGroupKFold so all 100 blocks from one physical specimen stay together.
3. Fits imputer, scaler, and LDA inside each training fold only.
4. Applies SMOTE only to the training fold.
5. Trains several classification models.
6. Aggregates block-level predictions to specimen-level predictions using mean class probability.
7. Saves a confusion matrix for every fold and every model.
8. Saves an overall confusion matrix and a summary containing accuracy, macro F1, and macro recall.
9. Performs statistical testing:
   - Cochran's Q test across all classifiers
   - McNemar tests comparing the best-accuracy model against all other models
   - Holm correction for multiple comparisons
10. Saves statistical test p-values to CSV.

Run:
    python Direct_Soil_Classification.py

Outputs:
    outputs/01_Direct_Soil_Classification/

Required extra package:
    pip install statsmodels
"""

from pathlib import Path

# =========================================================
# CONFIG: EDIT THESE ONLY IF NEEDED
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data" / "Data_files"
TRAIN_DATA_PATH = DATA_DIR / "dataAll_Soil_Composition_Training_Histogram_No_Augmentation_Adjusted_Files_1crop_0_03_sigma_subOperator_SuperPixel.csv"
# Keep generated paths short enough for Windows' legacy MAX_PATH limit. The
# full preprocessing description is already encoded in the input CSV name.
FILE_NAME = "grouped_cv"

LABEL_COL = None
BLOCKS_PER_SPECIMEN = 100
FEATURE_COLS = None

N_SPLITS = 5
RANDOM_STATE = 42
USE_SMOTE = True
USE_LDA = True
LDA_VARIANCE_TO_KEEP = 0.99

OUTPUT_BASE = SCRIPT_DIR / "outputs" / "01_Direct_Soil_Classification"
OUTPUT_ROOT = OUTPUT_BASE / "confusion_matrices"
METRICS_OUTPUT_ROOT = OUTPUT_BASE / "metrics_and_statistical_tests"

CLASS_ORDER = None

# =========================================================
# IMPORTS
# =========================================================

import os
import copy
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def load_table(path: str) -> pd.DataFrame:
    """Load CSV or Excel file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find TRAIN_DATA_PATH: {path}\n"
            f"Edit TRAIN_DATA_PATH at the top of this script."
        )

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError("TRAIN_DATA_PATH must be .csv, .xlsx, or .xls")


def choose_label_column(df: pd.DataFrame, label_col: Optional[str]) -> str:
    """Return label column name."""
    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"LABEL_COL '{label_col}' not found in dataset.")
        return label_col

    return df.columns[-1]


def choose_feature_columns(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]],
) -> List[str]:
    """Return feature column names."""
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"These FEATURE_COLS are missing from dataset: {missing}")
        return feature_cols

    selected = df.columns[:13].tolist()

    if len(selected) < 13:
        raise ValueError(
            "Dataset must have at least 13 columns to select the first 13 feature columns."
        )

    return selected


def make_groups(df: pd.DataFrame, blocks_per_specimen: int) -> np.ndarray:
    """Create specimen groups from consecutive row blocks."""
    if len(df) % blocks_per_specimen != 0:
        raise ValueError(
            f"Dataset has {len(df)} rows, which is not divisible by "
            f"BLOCKS_PER_SPECIMEN={blocks_per_specimen}.\n"
            f"Fix BLOCKS_PER_SPECIMEN so it matches the number of rows per specimen."
        )

    n_specimens = len(df) // blocks_per_specimen
    return np.repeat(np.arange(n_specimens), blocks_per_specimen)


def build_models(num_classes: int) -> Dict[str, object]:
    """Create classification models."""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            random_seed=RANDOM_STATE,
            verbose=False,
        )
    else:
        print("CatBoost is not installed. Skipping CatBoost.")

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        print("XGBoost is not installed. Skipping XGBoost.")

    return models


def preprocess_train_test_with_lda(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_train_encoded: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Fit imputer, scaler, and LDA only on training data.
    Then transform training and test data.
    """
    info = {}

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    info["imputer"] = imputer
    info["scaler"] = scaler

    if not USE_LDA:
        info["lda"] = None
        info["n_lda_components"] = None
        return X_train_scaled, X_test_scaled, info

    n_classes = len(np.unique(y_train_encoded))
    max_components = min(X_train_scaled.shape[1], n_classes - 1)

    if max_components < 1:
        raise ValueError("LDA cannot be fitted because training fold has fewer than 2 classes.")

    lda = LinearDiscriminantAnalysis(n_components=max_components)
    X_train_lda_all = lda.fit_transform(X_train_scaled, y_train_encoded)
    X_test_lda_all = lda.transform(X_test_scaled)

    if hasattr(lda, "explained_variance_ratio_") and lda.explained_variance_ratio_ is not None:
        cum_var = np.cumsum(lda.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cum_var, LDA_VARIANCE_TO_KEEP) + 1)
        n_keep = max(1, min(n_keep, X_train_lda_all.shape[1]))
    else:
        n_keep = X_train_lda_all.shape[1]

    info["lda"] = lda
    info["n_lda_components"] = n_keep
    info["lda_explained_variance_ratio"] = getattr(lda, "explained_variance_ratio_", None)

    return X_train_lda_all[:, :n_keep], X_test_lda_all[:, :n_keep], info


def apply_smote_if_possible(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE only on training fold, if possible."""
    if not USE_SMOTE:
        return X_train, y_train

    if not IMBLEARN_AVAILABLE:
        print("imblearn is not installed. SMOTE skipped.")
        return X_train, y_train

    unique, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()

    if min_count < 2:
        print("SMOTE skipped because at least one class has fewer than 2 samples in this fold.")
        return X_train, y_train

    k_neighbors = min(5, min_count - 1)

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    return smote.fit_resample(X_train, y_train)


def aggregate_probabilities_to_specimen(
    proba_blocks: np.ndarray,
    block_groups: np.ndarray,
    model_classes_encoded: np.ndarray,
    all_classes_encoded: np.ndarray,
) -> pd.DataFrame:
    """
    Aggregate block-level probabilities to specimen-level probabilities.
    Handles cases where a model saw only a subset of classes.
    """
    full_proba = np.zeros((proba_blocks.shape[0], len(all_classes_encoded)), dtype=float)

    class_to_col = {cls: i for i, cls in enumerate(all_classes_encoded)}

    for local_col, cls in enumerate(model_classes_encoded):
        full_col = class_to_col[cls]
        full_proba[:, full_col] = proba_blocks[:, local_col]

    proba_df = pd.DataFrame(full_proba, columns=all_classes_encoded)
    proba_df["Specimen_ID"] = block_groups

    return proba_df.groupby("Specimen_ID")[list(all_classes_encoded)].mean()


def majority_vote_to_specimen(
    pred_blocks: np.ndarray,
    block_groups: np.ndarray,
) -> pd.Series:
    """Fallback aggregation when predict_proba is unavailable."""
    pred_df = pd.DataFrame({"Specimen_ID": block_groups, "Pred": pred_blocks})
    return pred_df.groupby("Specimen_ID")["Pred"].agg(lambda x: x.value_counts().idxmax())


def true_labels_to_specimen(y_true_blocks: np.ndarray, block_groups: np.ndarray) -> pd.Series:
    """Convert block-level true labels to specimen-level true labels."""
    true_df = pd.DataFrame({"Specimen_ID": block_groups, "True": y_true_blocks})
    return true_df.groupby("Specimen_ID")["True"].first()


def features_to_specimen(
    X_blocks: np.ndarray,
    block_groups: np.ndarray,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Aggregate raw block-level features to specimen-level features."""
    feature_df = pd.DataFrame(X_blocks, columns=feature_cols)
    feature_df["Specimen_ID"] = block_groups
    return feature_df.groupby("Specimen_ID")[feature_cols].mean()


def fresh_model(model_template: object) -> object:
    """Return an unfitted model instance from a template."""
    try:
        return clone(model_template)
    except Exception:
        return copy.deepcopy(model_template)


def save_confusion_matrix(
    y_true_names: np.ndarray,
    y_pred_names: np.ndarray,
    class_names: List[str],
    out_path: str,
    title: str,
):
    """Save normalized confusion matrix as PNG."""
    cm = confusion_matrix(
        y_true_names,
        y_pred_names,
        labels=class_names,
        normalize="true",
    )

    plt.figure(figsize=(10, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        annot_kws={"size": 10},
    )
    plt.title(title)
    plt.xlabel("Predicted Soil Texture")
    plt.ylabel("Actual Soil Texture")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """Calculate specimen-level classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "Macro_Recall": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
    }


def run_best_model_mcnemar_statistics(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: str,
    alpha: float = 0.05,
):
    """
    Statistical testing for classification models.

    Overall test:
        Cochran's Q test across all models.

    Pairwise test:
        McNemar test comparing the best-accuracy model against each other model.

    Multiple-comparison correction:
        Holm correction.

    The tests use specimen-level correctness:
        Correct = 1
        Wrong   = 0
    """

    os.makedirs(output_dir, exist_ok=True)

    required_prediction_cols = ["Model", "Specimen_ID", "Correct"]
    for col in required_prediction_cols:
        if col not in predictions_df.columns:
            raise ValueError(f"Missing required column in predictions_df: {col}")

    required_metric_cols = ["Model", "Accuracy"]
    for col in required_metric_cols:
        if col not in metrics_df.columns:
            raise ValueError(f"Missing required column in metrics_df: {col}")

    # ---------------------------------------------------------
    # Select practical best model by highest mean CV accuracy
    # ---------------------------------------------------------
    mean_accuracy = (
        metrics_df.groupby("Model")["Accuracy"]
        .mean()
        .sort_values(ascending=False)
    )

    best_model = mean_accuracy.index[0]

    # ---------------------------------------------------------
    # Build specimen-level correctness matrix
    # Rows = specimens, columns = models, values = 1 correct or 0 wrong
    # ---------------------------------------------------------
    correct_matrix = predictions_df.pivot_table(
        index="Specimen_ID",
        columns="Model",
        values="Correct",
        aggfunc="mean",
    )

    available_models = [m for m in mean_accuracy.index if m in correct_matrix.columns]
    correct_matrix = correct_matrix[available_models].dropna()

    correct_matrix = correct_matrix.round().astype(int)

    if best_model not in correct_matrix.columns:
        raise ValueError(f"Best model {best_model} is not available in correctness matrix.")

    if len(available_models) < 2:
        raise ValueError("Need at least two models for statistical testing.")

    if len(correct_matrix) < 2:
        raise ValueError("Need at least two specimens for statistical testing.")

    # ---------------------------------------------------------
    # Cochran's Q test
    # ---------------------------------------------------------
    try:
        q_result = cochrans_q(correct_matrix.to_numpy())
        cochran_stat = q_result.statistic
        cochran_p = q_result.pvalue
    except Exception as e:
        cochran_stat = np.nan
        cochran_p = np.nan
        print(f"Cochran's Q test failed: {e}")

    cochran_df = pd.DataFrame({
        "Test": ["Cochran_Q"],
        "Metric": ["Specimen-level correctness"],
        "Best_Model_By_Mean_Accuracy": [best_model],
        "Number_of_Models": [len(available_models)],
        "Number_of_Specimens": [len(correct_matrix)],
        "Statistic": [cochran_stat],
        "p_value": [cochran_p],
        "alpha": [alpha],
        "Significant": [bool(cochran_p < alpha) if not np.isnan(cochran_p) else False],
    })

    # ---------------------------------------------------------
    # McNemar tests: best model vs each other model
    # ---------------------------------------------------------
    pairwise_records = []
    raw_p_values = []

    best_correct = correct_matrix[best_model].to_numpy()

    for other_model in available_models:
        if other_model == best_model:
            continue

        other_correct = correct_matrix[other_model].to_numpy()

        both_wrong = int(np.sum((best_correct == 0) & (other_correct == 0)))
        best_wrong_other_correct = int(np.sum((best_correct == 0) & (other_correct == 1)))
        best_correct_other_wrong = int(np.sum((best_correct == 1) & (other_correct == 0)))
        both_correct = int(np.sum((best_correct == 1) & (other_correct == 1)))

        table = np.array([
            [both_wrong, best_wrong_other_correct],
            [best_correct_other_wrong, both_correct],
        ])

        discordant_pairs = best_wrong_other_correct + best_correct_other_wrong
        use_exact = discordant_pairs < 25

        try:
            result = mcnemar(
                table,
                exact=use_exact,
                correction=not use_exact,
            )
            stat = result.statistic
            p_value = result.pvalue
        except Exception as e:
            stat = np.nan
            p_value = np.nan
            print(f"McNemar test failed for {best_model} vs {other_model}: {e}")

        raw_p_values.append(p_value)

        pairwise_records.append({
            "Best_Model": best_model,
            "Other_Model": other_model,
            "Comparison": f"{best_model} vs {other_model}",
            "Mean_Accuracy_Best": correct_matrix[best_model].mean(),
            "Mean_Accuracy_Other": correct_matrix[other_model].mean(),
            "Accuracy_Difference_Best_Minus_Other": (
                correct_matrix[best_model].mean()
                - correct_matrix[other_model].mean()
            ),
            "Both_Wrong": both_wrong,
            "Best_Wrong_Other_Correct": best_wrong_other_correct,
            "Best_Correct_Other_Wrong": best_correct_other_wrong,
            "Both_Correct": both_correct,
            "Discordant_Pairs": discordant_pairs,
            "Exact_McNemar_Used": use_exact,
            "McNemar_Statistic": stat,
            "Raw_p_value": p_value,
        })

    pairwise_df = pd.DataFrame(pairwise_records)

    # ---------------------------------------------------------
    # Holm correction
    # ---------------------------------------------------------
    pairwise_df["Holm_Adjusted_p_value"] = np.nan
    pairwise_df["Significant_After_Holm"] = False

    if len(pairwise_df) > 0:
        valid_p_mask = pairwise_df["Raw_p_value"].notna().to_numpy()

        if valid_p_mask.sum() > 0:
            reject, adjusted_p, _, _ = multipletests(
                pairwise_df.loc[valid_p_mask, "Raw_p_value"].to_numpy(),
                alpha=alpha,
                method="holm",
            )

            pairwise_df.loc[valid_p_mask, "Holm_Adjusted_p_value"] = adjusted_p
            pairwise_df.loc[valid_p_mask, "Significant_After_Holm"] = reject

    # ---------------------------------------------------------
    # Model selection summary
    # ---------------------------------------------------------
    if len(pairwise_df) > 0 and pairwise_df["Significant_After_Holm"].all():
        decision = f"{best_model} is statistically superior to all compared models at alpha={alpha}."
    elif len(pairwise_df) > 0:
        not_sig_models = pairwise_df.loc[
            ~pairwise_df["Significant_After_Holm"],
            "Other_Model"
        ].tolist()
        decision = (
            f"{best_model} has the highest mean accuracy, but it is not statistically "
            f"separated from: {', '.join(not_sig_models)} at alpha={alpha}."
        )
    else:
        decision = "Pairwise testing was not performed."

    selection_df = pd.DataFrame({
        "Best_Model_By_Mean_Accuracy": [best_model],
        "Best_Model_Mean_Accuracy": [mean_accuracy.loc[best_model]],
        "Cochran_Q_p_value": [cochran_p],
        "All_Best_vs_Others_Significant_After_Holm": [
            bool(len(pairwise_df) > 0 and pairwise_df["Significant_After_Holm"].all())
        ],
        "Decision": [decision],
    })

    # ---------------------------------------------------------
    # Save CSV files
    # ---------------------------------------------------------
    pairwise_csv_path = os.path.join(
        output_dir,
        "best_model_vs_others_mcnemar_p_values.csv",
    )

    cochran_csv_path = os.path.join(
        output_dir,
        "cochran_q_test.csv",
    )

    selection_csv_path = os.path.join(
        output_dir,
        "statistical_model_selection_summary.csv",
    )

    pairwise_df.to_csv(pairwise_csv_path, index=False)
    cochran_df.to_csv(cochran_csv_path, index=False)
    selection_df.to_csv(selection_csv_path, index=False)

    print("\nClassification statistical testing completed.")
    print(f"Best model by mean accuracy: {best_model}")
    print(f"Cochran's Q p-value: {cochran_p}")
    print(f"Best-vs-others McNemar p-values saved to: {pairwise_csv_path}")
    print(f"Cochran's Q result saved to: {cochran_csv_path}")
    print(f"Model selection summary saved to: {selection_csv_path}")

    return pairwise_df, cochran_df, correct_matrix, selection_df


# =========================================================
# MAIN SCRIPT
# =========================================================

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(METRICS_OUTPUT_ROOT, exist_ok=True)

    print("Loading dataset...")
    df = load_table(TRAIN_DATA_PATH)

    label_col = choose_label_column(df, LABEL_COL)
    feature_cols = choose_feature_columns(df, FEATURE_COLS)
    groups = make_groups(df, BLOCKS_PER_SPECIMEN)

    X_raw = df[feature_cols].to_numpy(dtype=float)
    y_raw_names = df[label_col].astype(str).to_numpy()

    print(f"Dataset rows: {len(df)}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Label column: {label_col}")
    print(f"Number of unique specimens/groups: {len(np.unique(groups))}")

    if len(groups) != len(X_raw):
        raise ValueError("groups length does not match X rows.")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw_names)

    if CLASS_ORDER is None:
        class_names = label_encoder.classes_.tolist()
    else:
        class_names = list(CLASS_ORDER)
        missing_from_encoder = [c for c in class_names if c not in label_encoder.classes_]
        if missing_from_encoder:
            raise ValueError(f"CLASS_ORDER contains labels not found in dataset: {missing_from_encoder}")

    all_classes_encoded = label_encoder.transform(label_encoder.classes_)
    num_classes = len(label_encoder.classes_)

    model_templates = build_models(num_classes=num_classes)

    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_metrics_records = []
    all_prediction_records = []

    model_all_true = {model_name: [] for model_name in model_templates}
    model_all_pred = {model_name: [] for model_name in model_templates}

    print("\nStarting grouped stratified cross-validation...")

    splits = list(sgkf.split(X_raw, y_encoded, groups))

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]
        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]
        train_groups = groups[train_idx]
        test_groups = groups[test_idx]

        # Leakage check
        overlap = set(train_groups).intersection(set(test_groups))
        if len(overlap) > 0:
            raise RuntimeError(f"Data leakage detected in fold {fold}.")

        print(
            f"\nFold {fold}: "
            f"train specimens={len(np.unique(train_groups))}, "
            f"test specimens={len(np.unique(test_groups))}"
        )

        # Fit preprocessing only on training data
        X_train, X_test, pre_info = preprocess_train_test_with_lda(
            X_train_raw,
            X_test_raw,
            y_train,
        )

        # Apply SMOTE only to training data
        X_train_bal, y_train_bal = apply_smote_if_possible(X_train, y_train)

        y_true_specimen_series = true_labels_to_specimen(y_test, test_groups)
        specimen_features = features_to_specimen(X_test_raw, test_groups, feature_cols)

        for model_name, model_template in model_templates.items():
            current_model = fresh_model(model_template)
            current_model.fit(X_train_bal, y_train_bal)

            # Block-level prediction, then specimen-level aggregation
            if hasattr(current_model, "predict_proba"):
                proba_blocks = current_model.predict_proba(X_test)
                model_classes_encoded = current_model.classes_

                proba_specimen = aggregate_probabilities_to_specimen(
                    proba_blocks=proba_blocks,
                    block_groups=test_groups,
                    model_classes_encoded=model_classes_encoded,
                    all_classes_encoded=all_classes_encoded,
                )

                y_pred_specimen_encoded = proba_specimen.idxmax(axis=1).to_numpy(dtype=int)
                specimen_index = proba_specimen.index

            else:
                pred_blocks = current_model.predict(X_test)
                pred_specimen_series = majority_vote_to_specimen(pred_blocks, test_groups)
                y_pred_specimen_encoded = pred_specimen_series.to_numpy(dtype=int)
                specimen_index = pred_specimen_series.index

            y_true_specimen_encoded = y_true_specimen_series.loc[specimen_index].to_numpy(dtype=int)

            y_true_specimen_names = label_encoder.inverse_transform(y_true_specimen_encoded)
            y_pred_specimen_names = label_encoder.inverse_transform(y_pred_specimen_encoded)

            # Metrics
            fold_metrics = metric_dict(
                y_true_specimen_names,
                y_pred_specimen_names,
                class_names,
            )

            fold_metrics.update({
                "Model": model_name,
                "Fold": fold,
                "Train_Specimens": len(np.unique(train_groups)),
                "Test_Specimens": len(np.unique(test_groups)),
                "LDA_Components": pre_info.get("n_lda_components"),
            })

            all_metrics_records.append(fold_metrics)

            print(
                f"  {model_name}: "
                f"Accuracy={fold_metrics['Accuracy']:.4f}, "
                f"Macro_F1={fold_metrics['Macro_F1']:.4f}, "
                f"Macro_Recall={fold_metrics['Macro_Recall']:.4f}"
            )

            # Save fold confusion matrix
            fold_cm_path = os.path.join(
                OUTPUT_ROOT,
                f"{model_name}_confusion_matrix_fold_{fold}.png",
            )
            save_confusion_matrix(
                y_true_names=y_true_specimen_names,
                y_pred_names=y_pred_specimen_names,
                class_names=class_names,
                out_path=fold_cm_path,
                title=f"{model_name} - Fold {fold} - Specimen-level Confusion Matrix",
            )

            model_all_true[model_name].extend(y_true_specimen_names.tolist())
            model_all_pred[model_name].extend(y_pred_specimen_names.tolist())

            # Store predictions
            for sid, yt, yp in zip(specimen_index, y_true_specimen_names, y_pred_specimen_names):
                prediction_record = {
                    "Model": model_name,
                    "Fold": fold,
                    "Specimen_ID": sid,
                    "True_Soil_Type": yt,
                    "Predicted_Soil_Type": yp,
                    "Correct": int(yt == yp),
                }

                for feature_col in feature_cols:
                    prediction_record[feature_col] = specimen_features.loc[sid, feature_col]

                all_prediction_records.append(prediction_record)

    # Save overall confusion matrix for each model
    for model_name in model_templates:
        overall_cm_path = os.path.join(
            OUTPUT_ROOT,
            f"{model_name}_confusion_matrix_overall_grouped_cv.png",
        )

        save_confusion_matrix(
            y_true_names=np.asarray(model_all_true[model_name]),
            y_pred_names=np.asarray(model_all_pred[model_name]),
            class_names=class_names,
            out_path=overall_cm_path,
            title=f"{model_name} - Overall Grouped CV Specimen-level Confusion Matrix",
        )

    # Build metrics and predictions tables for summary/statistical calculations.
    # Only the requested summary/statistical CSV files are written to disk.
    metrics_df = pd.DataFrame(all_metrics_records)
    predictions_df = pd.DataFrame(all_prediction_records)

    # Save summary metrics
    summary_cols = [
        "Accuracy",
        "Macro_F1",
        "Macro_Recall",
    ]

    summary_df = metrics_df.groupby("Model")[summary_cols].agg(["mean", "std"])
    summary_csv_path = os.path.join(METRICS_OUTPUT_ROOT, "metrics_summaery.csv")
    summary_df.to_csv(summary_csv_path)

    # Statistical testing
    pairwise_stats_df, cochran_stats_df, correctness_matrix_df, selection_df = (
        run_best_model_mcnemar_statistics(
            predictions_df=predictions_df,
            metrics_df=metrics_df,
            output_dir=METRICS_OUTPUT_ROOT,
            alpha=0.05,
        )
    )

    print("\nDone.")
    print(f"Confusion matrices saved under: {OUTPUT_ROOT}")
    print(f"Summary metrics saved to: {summary_csv_path}")
    print(f"Statistical test outputs saved under: {METRICS_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
