"""
Grouped CV regression for soil MSI paper.

Outputs are separated directly under:
    ./outputs/02_Soil_Composition_Prediction/
    ./outputs/03_Indirect Soil_Classfication/

Created outputs:
1) regression/cross_validation_metrics/<model>_cross_validation_metrics.csv
   - one file per model containing five-fold metrics plus mean and standard deviation

2) regression/testing_best_fold_<model>.csv
   - one file per model
   - best fold selected by lowest specimen-level Overall_RMSE
   - columns: 13 wavelength features + true Clay/Sand/Silt + pred Clay/Sand/Silt

3) regression/validation_results_<model>.csv
   - one file per model
   - columns: 13 wavelength features + true Clay/Sand/Silt + pred Clay/Sand/Silt

4) regression and indirect_classification statistical CSV files
5) indirect_classification/summary_metrics.csv and normalized confusion matrices

IMPORTANT:
- Use RAW 13 wavelength features as input. Do NOT use features where scaler/LDA was fitted before CV.
- This code assumes every 100 consecutive rows belong to one specimen.
- The final CSV column is treated as the original soil texture label for indirect classification.
"""

import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import friedmanchisquare, wilcoxon

from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multioutput import MultiOutputRegressor

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

from statsmodels.stats.contingency_tables import cochrans_q, mcnemar

try:
    from catboost import CatBoostRegressor
except ImportError as e:
    raise ImportError("CatBoost is not installed. Install it with: pip install catboost") from e

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError("XGBoost is not installed. Install it with: pip install xgboost") from e


# =========================================================
# USER CONFIGURATION
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data" / "Data_files"
TRAIN_CSV_PATH = DATA_DIR / "dataAll_Soil_Composition_Training_Histogram_No_Augmentation_Adjusted_Files_1crop_0_03_sigma_subOperator_SuperPixel.csv"
VAL_CSV_PATH = DATA_DIR / "dataAll_Soil_Composition_Validation_Histogram_No_Augmentation_Adjusted_Files_1crop_0_03_sigma_subOperator_SuperPixel.csv"

OUTPUT_ROOT = SCRIPT_DIR / "outputs"
REGRESSION_OUTPUT_DIR = OUTPUT_ROOT / "02_Soil_Composition_Prediction"
CV_METRICS_OUTPUT_DIR = REGRESSION_OUTPUT_DIR / "cross_validation_metrics"
REGRESSION_PLOTS_OUTPUT_DIR = REGRESSION_OUTPUT_DIR / "composition_prediction_plots"
INDIRECT_OUTPUT_DIR = OUTPUT_ROOT / "03_Indirect Soil_Classfication"
INDIRECT_CM_OUTPUT_DIR = INDIRECT_OUTPUT_DIR / "confusion_matrices"

for output_path in (
    REGRESSION_OUTPUT_DIR,
    CV_METRICS_OUTPUT_DIR,
    REGRESSION_PLOTS_OUTPUT_DIR,
    INDIRECT_OUTPUT_DIR,
    INDIRECT_CM_OUTPUT_DIR,
):
    os.makedirs(output_path, exist_ok=True)

BLOCKS_PER_SPECIMEN = 100
N_SPLITS = 5
RANDOM_STATE = 42

USE_LDA = True
LDA_VARIANCE_TO_KEEP = 0.99

# Target column order is important: Clay, Sand, Silt.
TARGET_COLS = ["Clay", "Sand", "Silt"]

# If None, the first 13 columns are used as wavelength features.
WAVELENGTH_FEATURE_COLS = None

MODEL_ORDER = [
    "LinearRegression",
    "KNN",
    "DecisionTree",
    "RandomForest",
    "CatBoost",
    "XGBoost",
]


# =========================================================
# USDA TEXTURE TRIANGLE CLASSIFIER
# =========================================================

if "classify" not in globals():
    def classify(clay, sand, silt):
        """Return USDA soil texture class from Clay, Sand, Silt percentages."""
        clay = float(clay)
        sand = float(sand)
        silt = float(silt)

        total = clay + sand + silt
        if total <= 0:
            return "Unknown"

        clay = 100.0 * clay / total
        sand = 100.0 * sand / total
        silt = 100.0 * silt / total

        if (silt + 1.5 * clay) < 15:
            return "Sand"
        elif (silt + 1.5 * clay) >= 15 and (silt + 2 * clay) < 30:
            return "Loamy Sand"
        elif ((clay >= 7 and clay < 20 and sand > 52 and (silt + 2 * clay) >= 30) or
              (clay < 7 and silt < 50 and (silt + 2 * clay) >= 30)):
            return "Sandy Loam"
        elif clay >= 7 and clay < 27 and silt >= 28 and silt < 50 and sand <= 52:
            return "Loam"
        elif ((silt >= 50 and clay >= 12 and clay < 27) or
              (silt >= 50 and silt < 80 and clay < 12)):
            return "Silt Loam"
        elif silt >= 80 and clay < 12:
            return "Silt"
        elif clay >= 20 and clay < 35 and silt < 28 and sand > 45:
            return "Sandy Clay Loam"
        elif clay >= 27 and clay < 40 and sand > 20 and sand <= 45:
            return "Clay Loam"
        elif clay >= 27 and clay < 40 and sand <= 20:
            return "Silty Clay Loam"
        elif clay >= 35 and sand > 45:
            return "Sandy Clay"
        elif clay >= 40 and silt >= 40:
            return "Silty Clay"
        elif clay >= 40 and sand <= 45 and silt < 40:
            return "Clay"
        else:
            return "Unknown"


def classify_many(Y):
    """Classify many Clay/Sand/Silt rows into USDA texture classes."""
    return np.array([classify(row[0], row[1], row[2]) for row in np.asarray(Y)])


def normalize_components(Y):
    """
    Enforce non-negative Clay/Sand/Silt and sum = 100.
    Y column order: Clay, Sand, Silt.
    """
    Y = np.asarray(Y, dtype=float)
    Y = np.clip(Y, 0, None)
    row_sum = Y.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return 100.0 * Y / row_sum


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def holm_correction(p_values, alpha=0.05):
    """
    Manual Holm-Bonferroni correction.

    Returns:
        reject: boolean array
        adjusted: Holm-adjusted p-values
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    reject = np.zeros(m, dtype=bool)

    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        adj = min(1.0, factor * p_values[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max

    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            reject[idx] = True
        else:
            break

    return reject, adjusted


def run_indirect_classification_statistics(
    indirect_cls_df,
    cv_fold_metrics_df,
    model_order,
    alpha=0.05,
):
    """
    Statistical tests for indirect soil texture classification.

    Uses specimen-level correctness after:
        regression prediction -> simplex normalization -> USDA texture triangle mapping.

    Tests:
        1. Cochran's Q test across all indirect classification models.
        2. Pairwise McNemar tests comparing the best-accuracy model with all other models.
        3. Holm correction for multiple pairwise comparisons.
    """

    required_cols = ["Model", "Specimen_ID", "Correct"]
    for col in required_cols:
        if col not in indirect_cls_df.columns:
            raise ValueError(f"Missing required column in indirect_cls_df: {col}")

    mean_acc = (
        cv_fold_metrics_df.groupby("Model")["Indirect_Accuracy"]
        .mean()
        .sort_values(ascending=False)
    )

    best_model = mean_acc.index[0]

    correctness_matrix = indirect_cls_df.pivot_table(
        index="Specimen_ID",
        columns="Model",
        values="Correct",
        aggfunc="mean",
    )

    available_models = [m for m in model_order if m in correctness_matrix.columns]
    correctness_matrix = correctness_matrix[available_models].dropna()
    correctness_matrix = correctness_matrix.round().astype(int)

    if best_model not in correctness_matrix.columns:
        raise ValueError(f"Best model {best_model} not found in indirect correctness matrix.")

    if len(available_models) < 2:
        raise ValueError("Need at least two models for indirect classification statistical testing.")

    if len(correctness_matrix) < 2:
        raise ValueError("Need at least two specimens for indirect classification statistical testing.")

    # Cochran's Q test
    try:
        q_result = cochrans_q(correctness_matrix.to_numpy())
        cochran_stat = q_result.statistic
        cochran_p = q_result.pvalue
    except Exception as e:
        print("Indirect Cochran's Q test failed:", e)
        cochran_stat = np.nan
        cochran_p = np.nan

    cochran_df = pd.DataFrame({
        "Test": ["Cochran_Q"],
        "Metric": ["Indirect classification specimen-level correctness"],
        "Best_Model_By_Mean_Indirect_Accuracy": [best_model],
        "Number_of_Models": [len(available_models)],
        "Number_of_Specimens": [len(correctness_matrix)],
        "Statistic": [cochran_stat],
        "p_value": [cochran_p],
        "alpha": [alpha],
        "Significant": [bool(cochran_p < alpha) if not np.isnan(cochran_p) else False],
    })

    # Pairwise McNemar tests
    pairwise_records = []
    raw_p_values = []

    best_correct = correctness_matrix[best_model].to_numpy()

    for other_model in available_models:
        if other_model == best_model:
            continue

        other_correct = correctness_matrix[other_model].to_numpy()

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
            print(f"Indirect McNemar test failed for {best_model} vs {other_model}:", e)
            stat = np.nan
            p_value = np.nan

        raw_p_values.append(p_value)

        pairwise_records.append({
            "Best_Model": best_model,
            "Other_Model": other_model,
            "Comparison": f"{best_model} vs {other_model}",
            "Mean_Indirect_Accuracy_Best": correctness_matrix[best_model].mean(),
            "Mean_Indirect_Accuracy_Other": correctness_matrix[other_model].mean(),
            "Accuracy_Difference_Best_Minus_Other": (
                correctness_matrix[best_model].mean()
                - correctness_matrix[other_model].mean()
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

    pairwise_df["Holm_Adjusted_p_value"] = np.nan
    pairwise_df["Significant_After_Holm"] = False

    if len(pairwise_df) > 0:
        valid_mask = pairwise_df["Raw_p_value"].notna().to_numpy()

        if valid_mask.sum() > 0:
            reject, adjusted_p = holm_correction(
                pairwise_df.loc[valid_mask, "Raw_p_value"].to_numpy(),
                alpha=alpha,
            )

            pairwise_df.loc[valid_mask, "Holm_Adjusted_p_value"] = adjusted_p
            pairwise_df.loc[valid_mask, "Significant_After_Holm"] = reject

    mean_acc_df = mean_acc.reset_index()
    mean_acc_df.columns = ["Model", "Mean_Indirect_CV_Accuracy"]

    if np.isnan(cochran_p) or cochran_p >= alpha:
        decision = (
            f"Cochran's Q test was not significant. {best_model} has the highest "
            f"mean indirect accuracy, but the overall difference among indirect classifiers "
            f"is not statistically significant."
        )
        statistical_best = "No statistically unique best model"
    else:
        if len(pairwise_df) > 0 and pairwise_df["Significant_After_Holm"].all():
            decision = (
                f"{best_model} has the highest mean indirect accuracy and is significantly "
                f"better than all other indirect classification models after Holm correction."
            )
            statistical_best = best_model
        else:
            not_sig = pairwise_df.loc[
                ~pairwise_df["Significant_After_Holm"],
                "Other_Model",
            ].tolist()

            decision = (
                f"{best_model} has the highest mean indirect accuracy, but it is not "
                f"statistically separated from: {', '.join(not_sig)} after Holm correction."
            )
            statistical_best = "Statistical tie group"

    selection_df = pd.DataFrame({
        "Practical_Best_By_Mean_Indirect_Accuracy": [best_model],
        "Statistical_Best_Decision": [statistical_best],
        "Cochran_Q_p_value": [cochran_p],
        "All_Best_vs_Others_Significant_After_Holm": [
            bool(len(pairwise_df) > 0 and pairwise_df["Significant_After_Holm"].all())
        ],
        "Decision": [decision],
    })

    return selection_df, cochran_df, pairwise_df, mean_acc_df, correctness_matrix


# =========================================================
# DATA PREPARATION
# =========================================================

def load_dataframe(csv_path, label):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{label} CSV file was not found: {csv_path}")

    print(f"Loaded {label} CSV: {csv_path}")
    return pd.read_csv(csv_path).reset_index(drop=True)


def detect_feature_columns(df):
    if WAVELENGTH_FEATURE_COLS is not None:
        missing = [c for c in WAVELENGTH_FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"These WAVELENGTH_FEATURE_COLS are missing from dataset: {missing}")

        if len(WAVELENGTH_FEATURE_COLS) != 13:
            raise ValueError("WAVELENGTH_FEATURE_COLS must contain exactly 13 wavelength feature columns.")

        return list(WAVELENGTH_FEATURE_COLS)

    if df.shape[1] < 13:
        raise ValueError(
            f"Could not select 13 wavelength columns. Dataset has only {df.shape[1]} columns."
        )

    feature_cols = df.columns[:13].tolist()
    print("Selected first 13 wavelength feature columns:")
    print(feature_cols)
    return feature_cols


def make_groups(df, prefix="train"):
    if len(df) % BLOCKS_PER_SPECIMEN != 0:
        raise ValueError(
            f"{prefix}: number of rows ({len(df)}) is not divisible by "
            f"BLOCKS_PER_SPECIMEN={BLOCKS_PER_SPECIMEN}."
        )

    n_specimens = len(df) // BLOCKS_PER_SPECIMEN
    return np.repeat(np.arange(n_specimens), BLOCKS_PER_SPECIMEN)


def make_texture_labels(df):
    return df.iloc[:, -1].astype(str).to_numpy()


def make_composition_labels(df):
    compositions = df[TARGET_COLS].astype(str)
    return (
        compositions["Clay"] + "_"
        + compositions["Sand"] + "_"
        + compositions["Silt"]
    ).to_numpy()


def aggregate_truth_and_features(df, groups, feature_cols):
    temp = df[feature_cols + TARGET_COLS].copy()
    temp["Specimen_ID"] = groups

    feature_specimen = temp.groupby("Specimen_ID")[feature_cols].mean()
    target_specimen = temp.groupby("Specimen_ID")[TARGET_COLS].first()

    labels_block = make_texture_labels(df)
    label_df = pd.DataFrame({"Specimen_ID": groups, "True_Texture": labels_block})
    labels_specimen = label_df.groupby("Specimen_ID")["True_Texture"].first()

    out = pd.concat([feature_specimen, target_specimen, labels_specimen], axis=1).reset_index()
    return out


train_df = load_dataframe(TRAIN_CSV_PATH, "training")
val_df = load_dataframe(VAL_CSV_PATH, "validation")

for col in TARGET_COLS:
    if col not in train_df.columns:
        raise ValueError(f"Target column '{col}' not found in training dataset.")
    if col not in val_df.columns:
        raise ValueError(f"Target column '{col}' not found in validation dataset.")

FEATURE_COLS = detect_feature_columns(train_df)

train_groups = make_groups(train_df, prefix="train")
val_groups = make_groups(val_df, prefix="validation")

X_raw = train_df[FEATURE_COLS].to_numpy()
y_raw = train_df[TARGET_COLS].copy().reset_index(drop=True)
labels_block = make_texture_labels(train_df)
composition_labels_block = make_composition_labels(train_df)

X_val_raw = val_df[FEATURE_COLS].to_numpy()
y_val_raw = val_df[TARGET_COLS].copy().reset_index(drop=True)
labels_val_block = make_texture_labels(val_df)
composition_labels_val_block = make_composition_labels(val_df)

assert len(X_raw) == len(y_raw) == len(labels_block) == len(composition_labels_block) == len(train_groups)
assert len(X_val_raw) == len(y_val_raw) == len(labels_val_block) == len(composition_labels_val_block) == len(val_groups)

train_specimen_truth = aggregate_truth_and_features(train_df, train_groups, FEATURE_COLS)
val_specimen_truth = aggregate_truth_and_features(val_df, val_groups, FEATURE_COLS)

LABELS_ALL = sorted(pd.unique(np.concatenate([labels_block, labels_val_block])))

print("\nData check")
print("Train rows:", len(train_df), "Train specimens:", len(np.unique(train_groups)))
print("Validation rows:", len(val_df), "Validation specimens:", len(np.unique(val_groups)))
print("Feature columns:", FEATURE_COLS)
print("Texture labels:", LABELS_ALL)
print("LDA composition labels:", len(np.unique(composition_labels_block)))


# =========================================================
# PREPROCESSING INSIDE FOLD ONLY
# =========================================================

def fit_transform_preprocess(X_train, X_test, lda_labels_train=None):
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    preprocess_info = {
        "Use_LDA": USE_LDA,
        "Pre_LDA_Features": X_train_scaled.shape[1],
        "Post_LDA_Features": X_train_scaled.shape[1],
    }

    if not USE_LDA:
        return X_train_scaled, X_test_scaled, preprocess_info

    if lda_labels_train is None:
        raise ValueError("lda_labels_train is required when USE_LDA=True.")

    y_train_encoded, _ = pd.factorize(lda_labels_train)
    n_classes = len(np.unique(y_train_encoded))
    max_components = min(X_train_scaled.shape[1], n_classes - 1)

    if max_components < 1:
        print("Warning: LDA skipped because training fold has fewer than 2 classes.")
        return X_train_scaled, X_test_scaled, preprocess_info

    lda = LinearDiscriminantAnalysis(n_components=max_components)

    X_train_lda_all = lda.fit_transform(X_train_scaled, y_train_encoded)
    X_test_lda_all = lda.transform(X_test_scaled)

    if hasattr(lda, "explained_variance_ratio_") and lda.explained_variance_ratio_ is not None:
        cum_var = np.cumsum(lda.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cum_var, LDA_VARIANCE_TO_KEEP) + 1)
        n_keep = max(1, min(n_keep, X_train_lda_all.shape[1]))
    else:
        n_keep = X_train_lda_all.shape[1]

    preprocess_info["Post_LDA_Features"] = n_keep
    preprocess_info["LDA_Max_Components"] = max_components
    preprocess_info["LDA_Components_Kept"] = n_keep
    preprocess_info["LDA_Variance_To_Keep"] = LDA_VARIANCE_TO_KEEP

    return X_train_lda_all[:, :n_keep], X_test_lda_all[:, :n_keep], preprocess_info


# =========================================================
# MODELS
# =========================================================

def make_models():
    return {
        "LinearRegression": LinearRegression(),

        "KNN": KNeighborsRegressor(),

        "DecisionTree": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "CatBoost": MultiOutputRegressor(
            CatBoostRegressor(
                loss_function="RMSE",
                random_seed=RANDOM_STATE,
                verbose=False,
            )
        ),

        "XGBoost": MultiOutputRegressor(
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


def compute_regression_metrics(y_true, y_pred):
    row = {}
    row["Overall_R2"] = r2_score(y_true, y_pred)
    row["Overall_RMSE"] = rmse(y_true, y_pred)
    row["Overall_MAE"] = mean_absolute_error(y_true, y_pred)

    for i, comp in enumerate(TARGET_COLS):
        row[f"{comp}_R2"] = r2_score(y_true[:, i], y_pred[:, i])
        row[f"{comp}_RMSE"] = rmse(y_true[:, i], y_pred[:, i])
        row[f"{comp}_MAE"] = mean_absolute_error(y_true[:, i], y_pred[:, i])

    return row


def compute_classification_metrics(y_true_textures, y_pred_textures):
    return {
        "Indirect_Accuracy": accuracy_score(y_true_textures, y_pred_textures),
        "Indirect_F1_Macro": f1_score(
            y_true_textures,
            y_pred_textures,
            average="macro",
            labels=LABELS_ALL,
            zero_division=0,
        ),
        "Indirect_Recall_Macro": recall_score(
            y_true_textures,
            y_pred_textures,
            average="macro",
            labels=LABELS_ALL,
            zero_division=0,
        ),
    }


def save_indirect_confusion_matrix(y_true, y_pred, model_name, out_path):
    """Save a normalized specimen-level indirect-classification confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ALL, normalize="true")
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=LABELS_ALL,
        yticklabels=LABELS_ALL,
        square=True,
    )
    plt.title(f"{model_name} - Indirect Classification - Grouped CV")
    plt.xlabel("Predicted Soil Texture")
    plt.ylabel("Actual Soil Texture")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def aggregate_prediction_to_specimen(Y_pred_blocks, test_groups_for_blocks):
    pred_df = pd.DataFrame(Y_pred_blocks, columns=["Pred_Clay", "Pred_Sand", "Pred_Silt"])
    pred_df["Specimen_ID"] = test_groups_for_blocks

    pred_specimen = pred_df.groupby("Specimen_ID")[["Pred_Clay", "Pred_Sand", "Pred_Silt"]].mean()

    Y_pred_specimen = normalize_components(
        pred_specimen[["Pred_Clay", "Pred_Sand", "Pred_Silt"]].to_numpy()
    )

    pred_specimen[["Pred_Clay", "Pred_Sand", "Pred_Silt"]] = Y_pred_specimen

    return pred_specimen


def make_output_prediction_table(specimen_ids, pred_specimen_df, truth_specimen_df):
    truth = truth_specimen_df.set_index("Specimen_ID").loc[specimen_ids]
    pred = pred_specimen_df.loc[specimen_ids]

    out = pd.DataFrame(index=specimen_ids)

    for c in FEATURE_COLS:
        out[c] = truth[c].values

    out["True_Clay"] = truth["Clay"].values
    out["True_Sand"] = truth["Sand"].values
    out["True_Silt"] = truth["Silt"].values

    out["Pred_Clay"] = pred["Pred_Clay"].values
    out["Pred_Sand"] = pred["Pred_Sand"].values
    out["Pred_Silt"] = pred["Pred_Silt"].values

    out.insert(0, "Specimen_ID", out.index)

    return out.reset_index(drop=True)


def save_composition_prediction_plot(prediction_df, model_name, dataset_name, out_path):
    """Save notebook-style true-vs-predicted panels for each composition."""
    components = ["Clay", "Silt", "Sand"]
    panel_labels = ["(a)", "(b)", "(c)"]
    fig = plt.figure(figsize=(18, 5))

    for index, component in enumerate(components):
        axis = fig.add_subplot(1, 3, index + 1)
        y_true = prediction_df[f"True_{component}"].to_numpy(dtype=float)
        y_pred = prediction_df[f"Pred_{component}"].to_numpy(dtype=float)
        component_r2 = r2_score(y_true, y_pred)
        component_rmse = rmse(y_true, y_pred)

        print(
            f"{model_name} {dataset_name} {component}: "
            f"R2={component_r2:.4f}, RMSE={component_rmse:.4f}"
        )

        axis.scatter(y_true, y_pred, alpha=0.5)
        minimum = min(y_true.min(), y_pred.min())
        maximum = max(y_true.max(), y_pred.max())
        axis.plot([minimum, maximum], [minimum, maximum], "r--")
        axis.set_xlabel(f"True {component} (%)", fontsize=15)
        axis.set_ylabel(f"Predicted {component} (%)", fontsize=15)
        axis.set_title(panel_labels[index], y=-0.20, fontsize=15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# CREATE GROUPED CV SPLITS ONCE
# =========================================================

try:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = list(splitter.split(X_raw, labels_block, groups=train_groups))
except ValueError as e:
    print("StratifiedGroupKFold failed. Falling back to GroupKFold.")
    print("Reason:", e)
    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(X_raw, labels_block, groups=train_groups))


# =========================================================
# CROSS-VALIDATION
# =========================================================

cv_fold_metrics_records = []
cv_prediction_records = []
cv_specimen_error_records = []
cv_indirect_classification_records = []

models = make_models()

for fold, (train_idx, test_idx) in enumerate(splits, start=1):
    fold_train_groups = train_groups[train_idx]
    fold_test_groups = train_groups[test_idx]

    overlap = set(fold_train_groups).intersection(set(fold_test_groups))
    assert len(overlap) == 0, f"Data leakage detected in fold {fold}: same specimen in train and test."

    X_train_raw = X_raw[train_idx]
    X_test_raw = X_raw[test_idx]
    y_train = y_raw.iloc[train_idx]
    composition_labels_train = composition_labels_block[train_idx]

    X_train_proc, X_test_proc, pre_info = fit_transform_preprocess(
        X_train_raw,
        X_test_raw,
        lda_labels_train=composition_labels_train,
    )

    test_specimen_truth = aggregate_truth_and_features(
        train_df.iloc[test_idx].reset_index(drop=True),
        fold_test_groups,
        FEATURE_COLS,
    )

    specimen_ids = sorted(np.unique(fold_test_groups))

    y_true_specimen = test_specimen_truth.set_index("Specimen_ID").loc[
        specimen_ids,
        TARGET_COLS,
    ].to_numpy()

    true_textures = test_specimen_truth.set_index("Specimen_ID").loc[
        specimen_ids,
        "True_Texture",
    ].to_numpy()

    print(
        f"\nFold {fold}: "
        f"train specimens={len(np.unique(fold_train_groups))}, "
        f"test specimens={len(np.unique(fold_test_groups))}"
    )

    for model_name in MODEL_ORDER:
        model = make_models()[model_name]
        model.fit(X_train_proc, y_train[TARGET_COLS])

        Y_pred_blocks = model.predict(X_test_proc)
        pred_specimen_df = aggregate_prediction_to_specimen(Y_pred_blocks, fold_test_groups)

        Y_pred_specimen = pred_specimen_df.loc[
            specimen_ids,
            ["Pred_Clay", "Pred_Sand", "Pred_Silt"],
        ].to_numpy()

        reg_metrics = compute_regression_metrics(y_true_specimen, Y_pred_specimen)

        pred_textures = classify_many(Y_pred_specimen)
        cls_metrics = compute_classification_metrics(true_textures, pred_textures)

        # Store specimen-level indirect classification outputs for statistical testing
        for sid, true_tex, pred_tex in zip(specimen_ids, true_textures, pred_textures):
            cv_indirect_classification_records.append({
                "Model": model_name,
                "Fold": fold,
                "Specimen_ID": sid,
                "True_Texture": true_tex,
                "Predicted_Texture": pred_tex,
                "Correct": int(true_tex == pred_tex),
            })

        metric_row = {
            "Model": model_name,
            "Fold": fold,
            "Train_Specimens": len(np.unique(fold_train_groups)),
            "Test_Specimens": len(np.unique(fold_test_groups)),
        }

        metric_row.update(pre_info)
        metric_row.update(reg_metrics)
        metric_row.update(cls_metrics)
        cv_fold_metrics_records.append(metric_row)

        out_table = make_output_prediction_table(
            specimen_ids,
            pred_specimen_df,
            test_specimen_truth,
        )

        out_table.insert(0, "Fold", fold)
        out_table.insert(0, "Model", model_name)
        cv_prediction_records.append(out_table)

        abs_err = np.abs(y_true_specimen - Y_pred_specimen)
        sq_err = (y_true_specimen - Y_pred_specimen) ** 2

        for j, sid in enumerate(specimen_ids):
            cv_specimen_error_records.append({
                "Model": model_name,
                "Fold": fold,
                "Specimen_ID": sid,
                "AbsErr_Clay": abs_err[j, 0],
                "AbsErr_Sand": abs_err[j, 1],
                "AbsErr_Silt": abs_err[j, 2],
                "MAE_AllComponents": abs_err[j].mean(),
                "RMSE_AllComponents": np.sqrt(sq_err[j].mean()),
            })

        print(
            f"  {model_name}: "
            f"RMSE={reg_metrics['Overall_RMSE']:.4f}, "
            f"MAE={reg_metrics['Overall_MAE']:.4f}, "
            f"Indirect Acc={cls_metrics['Indirect_Accuracy']:.4f}"
        )

cv_fold_metrics_df = pd.DataFrame(cv_fold_metrics_records)
cv_predictions_df = pd.concat(cv_prediction_records, ignore_index=True)
cv_errors_df = pd.DataFrame(cv_specimen_error_records)
cv_indirect_classification_df = pd.DataFrame(cv_indirect_classification_records)


# =========================================================
# EXTERNAL VALIDATION
# =========================================================

validation_metric_records = []
validation_prediction_tables = {}

X_all_proc, X_val_proc, pre_info_val = fit_transform_preprocess(
    X_raw,
    X_val_raw,
    lda_labels_train=composition_labels_block,
)

val_specimen_ids = sorted(np.unique(val_groups))

y_val_specimen = val_specimen_truth.set_index("Specimen_ID").loc[
    val_specimen_ids,
    TARGET_COLS,
].to_numpy()

val_true_textures = val_specimen_truth.set_index("Specimen_ID").loc[
    val_specimen_ids,
    "True_Texture",
].to_numpy()

for model_name in MODEL_ORDER:
    model = make_models()[model_name]
    model.fit(X_all_proc, y_raw[TARGET_COLS])

    Y_val_pred_blocks = model.predict(X_val_proc)
    pred_val_specimen_df = aggregate_prediction_to_specimen(Y_val_pred_blocks, val_groups)

    Y_val_pred_specimen = pred_val_specimen_df.loc[
        val_specimen_ids,
        ["Pred_Clay", "Pred_Sand", "Pred_Silt"],
    ].to_numpy()

    reg_metrics = compute_regression_metrics(y_val_specimen, Y_val_pred_specimen)

    pred_val_textures = classify_many(Y_val_pred_specimen)
    cls_metrics = compute_classification_metrics(val_true_textures, pred_val_textures)

    metric_row = {
        "Model": model_name,
        "Validation_Specimens": len(val_specimen_ids),
    }

    metric_row.update(pre_info_val)
    metric_row.update(reg_metrics)
    metric_row.update(cls_metrics)
    validation_metric_records.append(metric_row)

    val_out_table = make_output_prediction_table(
        val_specimen_ids,
        pred_val_specimen_df,
        val_specimen_truth,
    )

    validation_prediction_tables[model_name] = val_out_table

    print(
        f"Validation {model_name}: "
        f"RMSE={reg_metrics['Overall_RMSE']:.4f}, "
        f"MAE={reg_metrics['Overall_MAE']:.4f}, "
        f"Indirect Acc={cls_metrics['Indirect_Accuracy']:.4f}"
    )

validation_metrics_df = pd.DataFrame(validation_metric_records)


# =========================================================
# REGRESSION STATISTICAL TESTS
# Friedman + Wilcoxon + Holm correction
# =========================================================

error_matrix = cv_errors_df.pivot_table(
    index="Specimen_ID",
    columns="Model",
    values="RMSE_AllComponents",
    aggfunc="mean",
)

error_matrix = error_matrix[MODEL_ORDER].dropna()

mean_errors = error_matrix.mean().sort_values()
best_candidate = mean_errors.index[0]

friedman_stat, friedman_p = friedmanchisquare(
    *[error_matrix[m].to_numpy() for m in MODEL_ORDER]
)

friedman_df = pd.DataFrame({
    "Test": ["Friedman test"],
    "Metric": ["Specimen-level RMSE across Clay, Sand, Silt"],
    "Statistic": [friedman_stat],
    "p_value": [friedman_p],
    "alpha": [0.05],
    "Significant": [friedman_p < 0.05],
})

pairwise_records = []
raw_p_values = []

other_models = [m for m in MODEL_ORDER if m != best_candidate]

for other in other_models:
    try:
        stat, p = wilcoxon(
            error_matrix[best_candidate].to_numpy(),
            error_matrix[other].to_numpy(),
            alternative="less",
            zero_method="wilcox",
        )
    except ValueError:
        stat, p = np.nan, 1.0

    raw_p_values.append(p)

    pairwise_records.append({
        "Comparison": f"{best_candidate} < {other}",
        "Best_Candidate": best_candidate,
        "Other_Model": other,
        "Wilcoxon_Statistic": stat,
        "Raw_p_value": p,
        "Mean_Error_Best_Candidate": error_matrix[best_candidate].mean(),
        "Mean_Error_Other_Model": error_matrix[other].mean(),
    })

reject, adjusted_p = holm_correction(raw_p_values, alpha=0.05)

for i, rec in enumerate(pairwise_records):
    rec["Holm_Adjusted_p_value"] = adjusted_p[i]
    rec["Significant_After_Holm"] = bool(reject[i])

pairwise_df = pd.DataFrame(pairwise_records)

mean_errors_df = mean_errors.reset_index()
mean_errors_df.columns = ["Model", "Mean_Specimen_RMSE_AllComponents"]

if friedman_p >= 0.05:
    best_model_statistical = "No statistically unique best model"
    selection_reason = (
        f"Friedman test was not significant (p={friedman_p:.6f}). "
        f"{best_candidate} has the lowest mean specimen-level RMSE, but the overall model difference is not statistically significant."
    )
    tied_models = ", ".join(MODEL_ORDER)
else:
    if len(pairwise_df) > 0 and pairwise_df["Significant_After_Holm"].all():
        best_model_statistical = best_candidate
        selection_reason = (
            f"{best_candidate} has the lowest mean specimen-level RMSE and is significantly better than all other models "
            f"using Wilcoxon signed-rank tests with Holm correction."
        )
        tied_models = best_candidate
    else:
        non_significant_others = pairwise_df.loc[
            ~pairwise_df["Significant_After_Holm"],
            "Other_Model",
        ].tolist()

        tied_group = [best_candidate] + non_significant_others

        best_model_statistical = "Statistical tie group"
        selection_reason = (
            f"Friedman test was significant (p={friedman_p:.6f}), but {best_candidate} was not significantly better "
            f"than all models after Holm correction. Practical best by mean RMSE is {best_candidate}."
        )
        tied_models = ", ".join(tied_group)

selection_df = pd.DataFrame({
    "Practical_Best_By_Lowest_Mean_RMSE": [best_candidate],
    "Statistical_Best_Decision": [best_model_statistical],
    "Tied_or_Not_Separated_Models": [tied_models],
    "Friedman_p_value": [friedman_p],
    "Selection_Reason": [selection_reason],
})


# =========================================================
# INDIRECT CLASSIFICATION STATISTICAL TESTS
# Cochran's Q + McNemar + Holm correction
# =========================================================

(
    indirect_selection_df,
    indirect_cochran_df,
    indirect_pairwise_df,
    indirect_mean_accuracy_df,
    indirect_correctness_matrix_df,
) = run_indirect_classification_statistics(
    indirect_cls_df=cv_indirect_classification_df,
    cv_fold_metrics_df=cv_fold_metrics_df,
    model_order=MODEL_ORDER,
    alpha=0.05,
)


# =========================================================
# SAVE OUTPUTS
# =========================================================

# One grouped-CV metrics CSV per regression model, including mean and std rows.
for model_name in MODEL_ORDER:
    model_fold_df = cv_fold_metrics_df[cv_fold_metrics_df["Model"] == model_name].copy()
    numeric_cols = model_fold_df.select_dtypes(include=[np.number]).columns.tolist()

    mean_row = model_fold_df[numeric_cols].mean().to_dict()
    mean_row.update({"Model": model_name, "Fold": "Mean"})
    std_row = model_fold_df[numeric_cols].std().to_dict()
    std_row.update({"Model": model_name, "Fold": "Std"})

    model_metrics_output_df = pd.concat(
        [model_fold_df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True,
    )
    metrics_path = CV_METRICS_OUTPUT_DIR / f"{model_name}_cross_validation_metrics.csv"
    model_metrics_output_df.to_csv(metrics_path, index=False)
    print("Saved:", metrics_path)


# Regression best-fold specimen predictions.
for model_name in MODEL_ORDER:
    model_fold_df = cv_fold_metrics_df[cv_fold_metrics_df["Model"] == model_name].copy()

    best_fold = int(
        model_fold_df.sort_values("Overall_RMSE", ascending=True).iloc[0]["Fold"]
    )

    best_test_df = cv_predictions_df[
        (cv_predictions_df["Model"] == model_name)
        & (cv_predictions_df["Fold"] == best_fold)
    ].copy()

    required_cols = ["Specimen_ID"] + FEATURE_COLS + [
        "True_Clay", "True_Sand", "True_Silt",
        "Pred_Clay", "Pred_Sand", "Pred_Silt",
    ]

    best_test_df = best_test_df[required_cols]

    out_path = REGRESSION_OUTPUT_DIR / f"testing_best_fold_{model_name}.csv"
    best_test_df.to_csv(out_path, index=False)

    print("Saved:", out_path)

    testing_plot_path = (
        REGRESSION_PLOTS_OUTPUT_DIR
        / f"{model_name}_testing_true_vs_predicted_composition.png"
    )
    save_composition_prediction_plot(
        prediction_df=best_test_df,
        model_name=model_name,
        dataset_name=f"best grouped-CV test fold {best_fold}",
        out_path=testing_plot_path,
    )
    print("Saved:", testing_plot_path)


# Regression external-validation specimen predictions.
for model_name in MODEL_ORDER:
    val_out_df = validation_prediction_tables[model_name].copy()

    required_cols = ["Specimen_ID"] + FEATURE_COLS + [
        "True_Clay", "True_Sand", "True_Silt",
        "Pred_Clay", "Pred_Sand", "Pred_Silt",
    ]

    val_out_df = val_out_df[required_cols]

    out_path = REGRESSION_OUTPUT_DIR / f"validation_results_{model_name}.csv"
    val_out_df.to_csv(out_path, index=False)

    print("Saved:", out_path)

    validation_plot_path = (
        REGRESSION_PLOTS_OUTPUT_DIR
        / f"{model_name}_validation_true_vs_predicted_composition.png"
    )
    save_composition_prediction_plot(
        prediction_df=val_out_df,
        model_name=model_name,
        dataset_name="external validation",
        out_path=validation_plot_path,
    )
    print("Saved:", validation_plot_path)


# Regression validation metrics and statistical results.
validation_metrics_df.to_csv(REGRESSION_OUTPUT_DIR / "validation_metrics.csv", index=False)
selection_df.to_csv(REGRESSION_OUTPUT_DIR / "statistical_model_selection.csv", index=False)
friedman_df.to_csv(REGRESSION_OUTPUT_DIR / "friedman_test.csv", index=False)
pairwise_df.to_csv(
    REGRESSION_OUTPUT_DIR / "best_model_vs_other_models_p_values.csv",
    index=False,
)
mean_errors_df.to_csv(REGRESSION_OUTPUT_DIR / "mean_specimen_errors.csv", index=False)


# Indirect-classification CV summary: accuracy, macro F1, and macro recall only.
indirect_metric_cols = [
    "Indirect_Accuracy",
    "Indirect_F1_Macro",
    "Indirect_Recall_Macro",
]
indirect_summary_df = (
    cv_fold_metrics_df.groupby("Model")[indirect_metric_cols]
    .agg(["mean", "std"])
)
indirect_summary_df.to_csv(INDIRECT_OUTPUT_DIR / "summary_metrics.csv")

indirect_selection_df.to_csv(
    INDIRECT_OUTPUT_DIR / "statistical_model_selection.csv",
    index=False,
)
indirect_cochran_df.to_csv(INDIRECT_OUTPUT_DIR / "cochran_q_test.csv", index=False)
indirect_pairwise_df.to_csv(
    INDIRECT_OUTPUT_DIR / "best_model_vs_other_models_mcnemar_p_values.csv",
    index=False,
)


# One normalized grouped-CV confusion matrix per indirect-classification model.
for model_name in MODEL_ORDER:
    model_indirect_df = cv_indirect_classification_df[
        cv_indirect_classification_df["Model"] == model_name
    ]
    cm_path = INDIRECT_CM_OUTPUT_DIR / f"{model_name}_confusion_matrix.png"
    save_indirect_confusion_matrix(
        model_indirect_df["True_Texture"].to_numpy(),
        model_indirect_df["Predicted_Texture"].to_numpy(),
        model_name,
        cm_path,
    )
    print("Saved:", cm_path)


# =========================================================
# FINAL PRINT SUMMARY
# =========================================================

print("\nDONE. Regression outputs are inside:")
print(REGRESSION_OUTPUT_DIR)
print("\nIndirect classification outputs are inside:")
print(INDIRECT_OUTPUT_DIR)

print("\nRegression practical/statistical model selection:")
print(selection_df.to_string(index=False))

print("\nRegression best performing model compared with other models:")
print(pairwise_df.to_string(index=False))

print("\nIndirect classification practical/statistical model selection:")
print(indirect_selection_df.to_string(index=False))

print("\nIndirect classification best performing model compared with other models:")
print(indirect_pairwise_df.to_string(index=False))
