"""
Permutation spectral-band importance for soil MSI using KNN only.

This script evaluates wavelength importance using leakage-free grouped CV.
It supports:
  1) KNN Regressor for Clay/Sand/Silt prediction
  2) KNN Classifier for direct USDA texture classification

Outputs are saved inside:
    ./outputs/KNN_spectral_band_importance/

Created files:
  - regression_permutation_importance.csv
  - classification_permutation_importance.csv
  - KNN_spectral_band_importance.png

Method:
  - Fit imputer, MinMaxScaler, and LDA only on training specimens within each fold.
  - Train KNN on the processed training fold.
  - Compute baseline specimen-level performance on the held-out fold.
  - For each wavelength, permute that wavelength in the held-out fold only.
  - Transform the permuted test fold using the already-fitted preprocessing.
  - Recompute specimen-level performance.
  - Importance is measured by performance degradation:
        Regression:   RMSE_Increase = RMSE_permuted - RMSE_baseline
        Classification: Accuracy_Drop = Accuracy_baseline - Accuracy_permuted

IMPORTANT:
  - Use raw 13 wavelength features as input, not precomputed LDA features.
  - By default, every 100 consecutive rows are treated as one specimen.
  - If the first 13 columns are not wavelength features, set WAVELENGTH_FEATURE_COLS manually.
"""

import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
)

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False


# =========================================================
# USER CONFIGURATION
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data" / "Data_files"
TRAIN_CSV_PATH = DATA_DIR / "dataAll_Soil_Composition_Training_Histogram_No_Augmentation_Adjusted_Files_1crop_0_03_sigma_subOperator_SuperPixel.csv"

OUTPUT_DIR = SCRIPT_DIR / "outputs" / "KNN_spectral_band_importance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BLOCKS_PER_SPECIMEN = 100
N_SPLITS = 5
RANDOM_STATE = 42
N_PERMUTATION_REPEATS = 10

# If True, permutation is done at specimen level: the whole band-vector of 100 blocks is swapped between specimens.
# This is more appropriate for grouped/specimen data.
SPECIMEN_LEVEL_PERMUTATION = True

USE_LDA = True
LDA_VARIANCE_TO_KEEP = 0.99

# Use SMOTE only for KNN classifier, only inside training fold.
USE_SMOTE_FOR_CLASSIFICATION = True

TARGET_COLS = ["Clay", "Sand", "Silt"]

# If None, first 13 columns are used as wavelength features.
# Example:
# WAVELENGTH_FEATURE_COLS = ["365", "405", "473", "530", "575", "621", "660", "735", "770", "830", "850", "890", "940"]
WAVELENGTH_FEATURE_COLS = None

# The script uses the final column as soil texture label unless LABEL_COL is specified.
LABEL_COL = None

# KNN hyperparameters
KNN_REGRESSOR_PARAMS = dict(n_neighbors=5)
KNN_CLASSIFIER_PARAMS = dict(n_neighbors=5)


# =========================================================
# USDA TEXTURE TRIANGLE CLASSIFIER FOR INDIRECT CLASSIFICATION
# =========================================================
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
    return np.array([classify(row[0], row[1], row[2]) for row in np.asarray(Y)])


def normalize_components(Y):
    """Clip negative predictions and normalize Clay+Sand+Silt to 100."""
    Y = np.asarray(Y, dtype=float)
    Y = np.clip(Y, 0, None)
    row_sum = Y.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return 100.0 * Y / row_sum


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_sheet_name(name):
    return str(name).replace("/", "_").replace("\\", "_")[:31]


# =========================================================
# DATA HELPERS
# =========================================================
def detect_feature_columns(df):
    if WAVELENGTH_FEATURE_COLS is not None:
        missing = [c for c in WAVELENGTH_FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing wavelength columns: {missing}")
        if len(WAVELENGTH_FEATURE_COLS) != 13:
            raise ValueError("WAVELENGTH_FEATURE_COLS must contain exactly 13 columns.")
        return list(WAVELENGTH_FEATURE_COLS)

    if df.shape[1] < 13:
        raise ValueError("Dataset must have at least 13 columns for wavelength features.")

    cols = df.columns[:13].tolist()
    print("Using first 13 columns as wavelength features:")
    print(cols)
    return cols


def make_groups(df):
    if len(df) % BLOCKS_PER_SPECIMEN != 0:
        raise ValueError(
            f"Rows={len(df)} is not divisible by BLOCKS_PER_SPECIMEN={BLOCKS_PER_SPECIMEN}. "
            "Check row ordering or BLOCKS_PER_SPECIMEN."
        )
    n_specimens = len(df) // BLOCKS_PER_SPECIMEN
    return np.repeat(np.arange(n_specimens), BLOCKS_PER_SPECIMEN)


def make_composition_labels(df):
    comps = df[TARGET_COLS].astype(str)
    return comps["Clay"] + "_" + comps["Sand"] + "_" + comps["Silt"]


def aggregate_truth_and_features(df, groups, feature_cols, label_col):
    temp = df[feature_cols + TARGET_COLS].copy()
    temp["Specimen_ID"] = groups
    features = temp.groupby("Specimen_ID")[feature_cols].mean()
    targets = temp.groupby("Specimen_ID")[TARGET_COLS].first()

    labels = df[label_col].astype(str).to_numpy()
    label_df = pd.DataFrame({"Specimen_ID": groups, "True_Texture": labels})
    labels_specimen = label_df.groupby("Specimen_ID")["True_Texture"].first()

    return pd.concat([features, targets, labels_specimen], axis=1).reset_index()


# =========================================================
# PREPROCESSING OBJECT
# =========================================================
class FoldPreprocessor:
    def __init__(self, use_lda=True, lda_variance_to_keep=0.99):
        self.use_lda = use_lda
        self.lda_variance_to_keep = lda_variance_to_keep
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = MinMaxScaler()
        self.lda = None
        self.n_keep = None
        self.info = {}

    def fit(self, X_train_raw, lda_labels_train=None):
        X_train_imp = self.imputer.fit_transform(X_train_raw)
        X_train_scaled = self.scaler.fit_transform(X_train_imp)

        self.info = {
            "Use_LDA": self.use_lda,
            "Pre_LDA_Features": X_train_scaled.shape[1],
            "Post_LDA_Features": X_train_scaled.shape[1],
        }

        if not self.use_lda:
            self.n_keep = X_train_scaled.shape[1]
            return self

        if lda_labels_train is None:
            raise ValueError("lda_labels_train is required when USE_LDA=True.")

        y_enc, _ = pd.factorize(lda_labels_train)
        n_classes = len(np.unique(y_enc))
        max_components = min(X_train_scaled.shape[1], n_classes - 1)

        if max_components < 1:
            self.lda = None
            self.n_keep = X_train_scaled.shape[1]
            print("Warning: LDA skipped because training fold has fewer than 2 labels.")
            return self

        self.lda = LinearDiscriminantAnalysis(n_components=max_components)
        X_train_lda_all = self.lda.fit_transform(X_train_scaled, y_enc)

        if hasattr(self.lda, "explained_variance_ratio_") and self.lda.explained_variance_ratio_ is not None:
            cum_var = np.cumsum(self.lda.explained_variance_ratio_)
            self.n_keep = int(np.searchsorted(cum_var, self.lda_variance_to_keep) + 1)
            self.n_keep = max(1, min(self.n_keep, X_train_lda_all.shape[1]))
        else:
            self.n_keep = X_train_lda_all.shape[1]

        self.info["Post_LDA_Features"] = self.n_keep
        self.info["LDA_Max_Components"] = max_components
        self.info["LDA_Components_Kept"] = self.n_keep
        self.info["LDA_Variance_To_Keep"] = self.lda_variance_to_keep
        return self

    def transform(self, X_raw):
        X_imp = self.imputer.transform(X_raw)
        X_scaled = self.scaler.transform(X_imp)
        if self.use_lda and self.lda is not None:
            X_lda_all = self.lda.transform(X_scaled)
            return X_lda_all[:, :self.n_keep]
        return X_scaled

    def fit_transform(self, X_train_raw, lda_labels_train=None):
        self.fit(X_train_raw, lda_labels_train=lda_labels_train)
        return self.transform(X_train_raw)


# =========================================================
# METRIC HELPERS
# =========================================================
def aggregate_regression_prediction_to_specimen(Y_pred_blocks, groups_for_blocks):
    pred_df = pd.DataFrame(Y_pred_blocks, columns=["Pred_Clay", "Pred_Sand", "Pred_Silt"])
    pred_df["Specimen_ID"] = groups_for_blocks
    pred_specimen = pred_df.groupby("Specimen_ID")[["Pred_Clay", "Pred_Sand", "Pred_Silt"]].mean()
    pred_norm = normalize_components(pred_specimen[["Pred_Clay", "Pred_Sand", "Pred_Silt"]].to_numpy())
    pred_specimen[["Pred_Clay", "Pred_Sand", "Pred_Silt"]] = pred_norm
    return pred_specimen


def regression_metrics(y_true_specimen, y_pred_specimen):
    out = {
        "Overall_R2": r2_score(y_true_specimen, y_pred_specimen),
        "Overall_RMSE": rmse(y_true_specimen, y_pred_specimen),
        "Overall_MAE": mean_absolute_error(y_true_specimen, y_pred_specimen),
    }
    for i, comp in enumerate(TARGET_COLS):
        out[f"{comp}_R2"] = r2_score(y_true_specimen[:, i], y_pred_specimen[:, i])
        out[f"{comp}_RMSE"] = rmse(y_true_specimen[:, i], y_pred_specimen[:, i])
        out[f"{comp}_MAE"] = mean_absolute_error(y_true_specimen[:, i], y_pred_specimen[:, i])
    return out


def classification_metrics(y_true, y_pred, class_names):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", labels=class_names, zero_division=0),
        "Macro_Recall": recall_score(y_true, y_pred, average="macro", labels=class_names, zero_division=0),
        "Macro_Precision": precision_score(y_true, y_pred, average="macro", labels=class_names, zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def aggregate_classifier_proba_to_specimen(proba_blocks, block_groups, model_classes_encoded, all_classes_encoded):
    full_proba = np.zeros((proba_blocks.shape[0], len(all_classes_encoded)), dtype=float)
    class_to_col = {cls: i for i, cls in enumerate(all_classes_encoded)}
    for local_col, cls in enumerate(model_classes_encoded):
        full_proba[:, class_to_col[cls]] = proba_blocks[:, local_col]
    proba_df = pd.DataFrame(full_proba, columns=all_classes_encoded)
    proba_df["Specimen_ID"] = block_groups
    return proba_df.groupby("Specimen_ID")[list(all_classes_encoded)].mean()


def apply_smote_if_possible(X_train, y_train):
    if not USE_SMOTE_FOR_CLASSIFICATION:
        return X_train, y_train
    if not IMBLEARN_AVAILABLE:
        print("imblearn not installed; SMOTE skipped.")
        return X_train, y_train

    unique, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()
    if min_count < 2:
        print("SMOTE skipped because at least one class has fewer than 2 samples.")
        return X_train, y_train

    k_neighbors = min(5, min_count - 1)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    return smote.fit_resample(X_train, y_train)


# =========================================================
# PERMUTATION HELPERS
# =========================================================
def permute_column_block_level(X_test_raw, col_idx, rng):
    X_perm = X_test_raw.copy()
    permuted_values = X_perm[:, col_idx].copy()
    rng.shuffle(permuted_values)
    X_perm[:, col_idx] = permuted_values
    return X_perm


def permute_column_specimen_level(X_test_raw, test_groups, col_idx, rng):
    """
    Permute one wavelength at specimen level.
    The 100-block band vector of one specimen is assigned to another specimen.
    This keeps within-specimen block structure while breaking band-target association.
    """
    X_perm = X_test_raw.copy()
    specimen_ids = np.array(sorted(np.unique(test_groups)))
    shuffled_specimen_ids = specimen_ids.copy()
    rng.shuffle(shuffled_specimen_ids)

    for dest_sid, src_sid in zip(specimen_ids, shuffled_specimen_ids):
        dest_idx = np.where(test_groups == dest_sid)[0]
        src_idx = np.where(test_groups == src_sid)[0]

        if len(dest_idx) == len(src_idx):
            X_perm[dest_idx, col_idx] = X_test_raw[src_idx, col_idx]
        else:
            # Fallback if group sizes somehow differ
            vals = X_test_raw[src_idx, col_idx].copy()
            if len(vals) == 0:
                continue
            X_perm[dest_idx, col_idx] = rng.choice(vals, size=len(dest_idx), replace=True)

    return X_perm


def permute_one_band(X_test_raw, test_groups, col_idx, rng):
    if SPECIMEN_LEVEL_PERMUTATION:
        return permute_column_specimen_level(X_test_raw, test_groups, col_idx, rng)
    return permute_column_block_level(X_test_raw, col_idx, rng)


# =========================================================
# MAIN ANALYSIS FUNCTIONS
# =========================================================
def run_knn_regressor_permutation(df, feature_cols, groups, labels_block, comp_labels_block, specimen_truth):
    print("\nRunning KNN regressor permutation importance...")
    X_raw = df[feature_cols].to_numpy(dtype=float)
    y_raw = df[TARGET_COLS].copy().reset_index(drop=True)

    try:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(splitter.split(X_raw, labels_block, groups=groups))
    except ValueError as e:
        print("StratifiedGroupKFold failed for regression. Falling back to GroupKFold.")
        print("Reason:", e)
        splitter = GroupKFold(n_splits=N_SPLITS)
        splits = list(splitter.split(X_raw, labels_block, groups=groups))

    baseline_records = []
    perm_records = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_groups = groups[train_idx]
        test_groups = groups[test_idx]
        assert len(set(train_groups).intersection(set(test_groups))) == 0, f"Leakage in fold {fold}"

        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]
        y_train = y_raw.iloc[train_idx]
        lda_labels_train = comp_labels_block[train_idx]

        pre = FoldPreprocessor(USE_LDA, LDA_VARIANCE_TO_KEEP)
        X_train_proc = pre.fit_transform(X_train_raw, lda_labels_train=lda_labels_train)
        X_test_proc = pre.transform(X_test_raw)

        model = KNeighborsRegressor(**KNN_REGRESSOR_PARAMS)
        model.fit(X_train_proc, y_train[TARGET_COLS])

        specimen_ids = sorted(np.unique(test_groups))
        truth = specimen_truth.set_index("Specimen_ID").loc[specimen_ids]
        y_true_spec = truth[TARGET_COLS].to_numpy()
        true_textures = truth["True_Texture"].to_numpy()

        # Baseline
        Y_pred_blocks = model.predict(X_test_proc)
        pred_spec_df = aggregate_regression_prediction_to_specimen(Y_pred_blocks, test_groups)
        y_pred_spec = pred_spec_df.loc[specimen_ids, ["Pred_Clay", "Pred_Sand", "Pred_Silt"]].to_numpy()
        base_metrics = regression_metrics(y_true_spec, y_pred_spec)
        base_textures = classify_many(y_pred_spec)
        base_acc = accuracy_score(true_textures, base_textures)

        baseline_record = {"Fold": fold, "Model": "KNN_Regressor", "Test_Specimens": len(specimen_ids)}
        baseline_record.update(pre.info)
        baseline_record.update(base_metrics)
        baseline_record["Indirect_Accuracy"] = base_acc
        baseline_records.append(baseline_record)

        print(f"  Fold {fold}: baseline RMSE={base_metrics['Overall_RMSE']:.4f}, indirect acc={base_acc:.4f}")

        # Permutations
        for band_idx, band in enumerate(feature_cols):
            for repeat in range(1, N_PERMUTATION_REPEATS + 1):
                rng = np.random.default_rng(RANDOM_STATE + 10000 * fold + 100 * band_idx + repeat)
                X_perm_raw = permute_one_band(X_test_raw, test_groups, band_idx, rng)
                X_perm_proc = pre.transform(X_perm_raw)

                Y_perm_blocks = model.predict(X_perm_proc)
                pred_perm_df = aggregate_regression_prediction_to_specimen(Y_perm_blocks, test_groups)
                y_perm_spec = pred_perm_df.loc[specimen_ids, ["Pred_Clay", "Pred_Sand", "Pred_Silt"]].to_numpy()

                perm_metrics = regression_metrics(y_true_spec, y_perm_spec)
                perm_textures = classify_many(y_perm_spec)
                perm_acc = accuracy_score(true_textures, perm_textures)

                rec = {
                    "Fold": fold,
                    "Repeat": repeat,
                    "Model": "KNN_Regressor",
                    "Permuted_Wavelength": band,
                    "Baseline_Overall_RMSE": base_metrics["Overall_RMSE"],
                    "Permuted_Overall_RMSE": perm_metrics["Overall_RMSE"],
                    "Overall_RMSE_Increase": perm_metrics["Overall_RMSE"] - base_metrics["Overall_RMSE"],
                    "Baseline_Indirect_Accuracy": base_acc,
                    "Permuted_Indirect_Accuracy": perm_acc,
                    "Indirect_Accuracy_Drop": base_acc - perm_acc,
                }

                for comp in TARGET_COLS:
                    rec[f"Baseline_{comp}_RMSE"] = base_metrics[f"{comp}_RMSE"]
                    rec[f"Permuted_{comp}_RMSE"] = perm_metrics[f"{comp}_RMSE"]
                    rec[f"{comp}_RMSE_Increase"] = perm_metrics[f"{comp}_RMSE"] - base_metrics[f"{comp}_RMSE"]

                perm_records.append(rec)

    baseline_df = pd.DataFrame(baseline_records)
    perm_df = pd.DataFrame(perm_records)

    summary_df = (
        perm_df.groupby("Permuted_Wavelength", as_index=False)
        .agg(
            Mean_Overall_RMSE_Increase=("Overall_RMSE_Increase", "mean"),
            Std_Overall_RMSE_Increase=("Overall_RMSE_Increase", "std"),
            Mean_Clay_RMSE_Increase=("Clay_RMSE_Increase", "mean"),
            Mean_Sand_RMSE_Increase=("Sand_RMSE_Increase", "mean"),
            Mean_Silt_RMSE_Increase=("Silt_RMSE_Increase", "mean"),
            Mean_Indirect_Accuracy_Drop=("Indirect_Accuracy_Drop", "mean"),
            Std_Indirect_Accuracy_Drop=("Indirect_Accuracy_Drop", "std"),
        )
        .sort_values("Mean_Overall_RMSE_Increase", ascending=False)
    )

    return baseline_df, perm_df, summary_df


def save_figure_16_band_importance(reg_summary_df, cls_summary_df, output_path):
    """Save the notebook-style two-panel spectral-band-importance figure."""
    reg_plot = reg_summary_df.copy()
    cls_plot = cls_summary_df.copy()
    reg_plot["Wavelength_nm"] = pd.to_numeric(reg_plot["Permuted_Wavelength"])
    cls_plot["Wavelength_nm"] = pd.to_numeric(cls_plot["Permuted_Wavelength"])
    reg_plot = reg_plot.sort_values("Wavelength_nm")
    cls_plot = cls_plot.sort_values("Wavelength_nm")

    wavelengths = cls_plot["Wavelength_nm"].to_numpy()
    wavelength_steps = np.diff(np.sort(wavelengths))
    bar_width = np.median(wavelength_steps) * 0.05
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    axes[0].bar(
        wavelengths,
        cls_plot["Mean_Accuracy_Drop"],
        width=bar_width,
        color="blue",
    )
    axes[0].set_xlabel("Wavelength (nm)", fontsize=15)
    axes[0].set_ylabel("Mean Accuracy Drop", fontsize=15)
    axes[0].set_xticks(wavelengths)
    axes[0].tick_params(axis="x", labelrotation=90, labelsize=14)
    axes[0].text(
        0.5, -0.32, "(a)",
        transform=axes[0].transAxes,
        ha="center", va="top", fontsize=15,
    )

    axes[1].bar(
        reg_plot["Wavelength_nm"].to_numpy(),
        reg_plot["Mean_Overall_RMSE_Increase"],
        width=bar_width,
        color="red",
    )
    axes[1].set_xlabel("Wavelength (nm)", fontsize=15)
    axes[1].set_ylabel("Mean RMSE Increase", fontsize=15)
    axes[1].set_xticks(wavelengths)
    axes[1].tick_params(axis="x", labelrotation=90, labelsize=14)
    axes[1].text(
        0.5, -0.32, "(b)",
        transform=axes[1].transAxes,
        ha="center", va="top", fontsize=15,
    )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.55, bottom=0.16)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_knn_classifier_permutation(df, feature_cols, groups, labels_block, class_names):
    print("\nRunning KNN classifier permutation importance...")
    X_raw = df[feature_cols].to_numpy(dtype=float)

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels_block)
    all_classes_encoded = le.transform(le.classes_)

    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(X_raw, y_encoded, groups=groups))

    baseline_records = []
    perm_records = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_groups = groups[train_idx]
        test_groups = groups[test_idx]
        assert len(set(train_groups).intersection(set(test_groups))) == 0, f"Leakage in fold {fold}"

        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]
        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]

        pre = FoldPreprocessor(USE_LDA, LDA_VARIANCE_TO_KEEP)
        X_train_proc = pre.fit_transform(X_train_raw, lda_labels_train=y_train)
        X_test_proc = pre.transform(X_test_raw)

        X_train_bal, y_train_bal = apply_smote_if_possible(X_train_proc, y_train)

        model = KNeighborsClassifier(**KNN_CLASSIFIER_PARAMS)
        model.fit(X_train_bal, y_train_bal)

        specimen_ids = sorted(np.unique(test_groups))
        true_df = pd.DataFrame({"Specimen_ID": test_groups, "True": y_test})
        y_true_spec_encoded = true_df.groupby("Specimen_ID")["True"].first().loc[specimen_ids].to_numpy(dtype=int)
        y_true_spec_names = le.inverse_transform(y_true_spec_encoded)

        # Baseline
        proba_blocks = model.predict_proba(X_test_proc)
        proba_spec = aggregate_classifier_proba_to_specimen(
            proba_blocks,
            test_groups,
            model.classes_,
            all_classes_encoded,
        )
        y_pred_spec_encoded = proba_spec.loc[specimen_ids].idxmax(axis=1).to_numpy(dtype=int)
        y_pred_spec_names = le.inverse_transform(y_pred_spec_encoded)
        base_metrics = classification_metrics(y_true_spec_names, y_pred_spec_names, class_names)

        base_record = {"Fold": fold, "Model": "KNN_Classifier", "Test_Specimens": len(specimen_ids)}
        base_record.update(pre.info)
        base_record.update(base_metrics)
        baseline_records.append(base_record)

        print(f"  Fold {fold}: baseline accuracy={base_metrics['Accuracy']:.4f}, macroF1={base_metrics['Macro_F1']:.4f}")

        # Permutations
        for band_idx, band in enumerate(feature_cols):
            for repeat in range(1, N_PERMUTATION_REPEATS + 1):
                rng = np.random.default_rng(RANDOM_STATE + 20000 * fold + 100 * band_idx + repeat)
                X_perm_raw = permute_one_band(X_test_raw, test_groups, band_idx, rng)
                X_perm_proc = pre.transform(X_perm_raw)

                proba_perm_blocks = model.predict_proba(X_perm_proc)
                proba_perm_spec = aggregate_classifier_proba_to_specimen(
                    proba_perm_blocks,
                    test_groups,
                    model.classes_,
                    all_classes_encoded,
                )
                y_perm_spec_encoded = proba_perm_spec.loc[specimen_ids].idxmax(axis=1).to_numpy(dtype=int)
                y_perm_spec_names = le.inverse_transform(y_perm_spec_encoded)
                perm_metrics = classification_metrics(y_true_spec_names, y_perm_spec_names, class_names)

                rec = {
                    "Fold": fold,
                    "Repeat": repeat,
                    "Model": "KNN_Classifier",
                    "Permuted_Wavelength": band,
                    "Baseline_Accuracy": base_metrics["Accuracy"],
                    "Permuted_Accuracy": perm_metrics["Accuracy"],
                    "Accuracy_Drop": base_metrics["Accuracy"] - perm_metrics["Accuracy"],
                    "Baseline_Macro_F1": base_metrics["Macro_F1"],
                    "Permuted_Macro_F1": perm_metrics["Macro_F1"],
                    "Macro_F1_Drop": base_metrics["Macro_F1"] - perm_metrics["Macro_F1"],
                    "Baseline_Balanced_Accuracy": base_metrics["Balanced_Accuracy"],
                    "Permuted_Balanced_Accuracy": perm_metrics["Balanced_Accuracy"],
                    "Balanced_Accuracy_Drop": base_metrics["Balanced_Accuracy"] - perm_metrics["Balanced_Accuracy"],
                }
                perm_records.append(rec)

    baseline_df = pd.DataFrame(baseline_records)
    perm_df = pd.DataFrame(perm_records)

    summary_df = (
        perm_df.groupby("Permuted_Wavelength", as_index=False)
        .agg(
            Mean_Accuracy_Drop=("Accuracy_Drop", "mean"),
            Std_Accuracy_Drop=("Accuracy_Drop", "std"),
            Mean_Macro_F1_Drop=("Macro_F1_Drop", "mean"),
            Std_Macro_F1_Drop=("Macro_F1_Drop", "std"),
            Mean_Balanced_Accuracy_Drop=("Balanced_Accuracy_Drop", "mean"),
            Std_Balanced_Accuracy_Drop=("Balanced_Accuracy_Drop", "std"),
        )
        .sort_values("Mean_Accuracy_Drop", ascending=False)
    )

    return baseline_df, perm_df, summary_df


# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading training dataset...")
    if not os.path.exists(TRAIN_CSV_PATH):
        raise FileNotFoundError(f"TRAIN_CSV_PATH not found: {TRAIN_CSV_PATH}")

    df = pd.read_csv(TRAIN_CSV_PATH).reset_index(drop=True)
    feature_cols = detect_feature_columns(df)

    for col in TARGET_COLS:
        if col not in df.columns:
            raise ValueError(f"Target column {col} was not found in the dataset.")

    label_col = LABEL_COL if LABEL_COL is not None else df.columns[-1]
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col} was not found in dataset.")

    groups = make_groups(df)
    labels_block = df[label_col].astype(str).to_numpy()
    comp_labels_block = make_composition_labels(df).to_numpy()
    class_names = sorted(pd.unique(labels_block).tolist())

    specimen_truth = aggregate_truth_and_features(df, groups, feature_cols, label_col)

    print("Rows:", len(df))
    print("Specimens:", len(np.unique(groups)))
    print("Features:", feature_cols)
    print("Label column:", label_col)
    print("Classes:", class_names)
    print("Permutation repeats:", N_PERMUTATION_REPEATS)
    print("Specimen-level permutation:", SPECIMEN_LEVEL_PERMUTATION)

    # Run KNN regressor importance
    reg_baseline_df, reg_perm_df, reg_summary_df = run_knn_regressor_permutation(
        df=df,
        feature_cols=feature_cols,
        groups=groups,
        labels_block=labels_block,
        comp_labels_block=comp_labels_block,
        specimen_truth=specimen_truth,
    )

    # Run KNN classifier importance
    cls_baseline_df, cls_perm_df, cls_summary_df = run_knn_classifier_permutation(
        df=df,
        feature_cols=feature_cols,
        groups=groups,
        labels_block=labels_block,
        class_names=class_names,
    )

    # Save only the two requested summary CSV files.
    regression_csv_path = OUTPUT_DIR / "regression_permutation_importance.csv"
    classification_csv_path = OUTPUT_DIR / "classification_permutation_importance.csv"
    reg_summary_df.to_csv(regression_csv_path, index=False)
    cls_summary_df.to_csv(classification_csv_path, index=False)
    print("Saved:", regression_csv_path)
    print("Saved:", classification_csv_path)

    figure_path = OUTPUT_DIR / "KNN_spectral_band_importance.png"
    save_figure_16_band_importance(reg_summary_df, cls_summary_df, figure_path)
    print("Saved:", figure_path)

    print("\nTop wavelengths for KNN regressor by RMSE increase:")
    print(reg_summary_df.head(13).to_string(index=False))

    print("\nTop wavelengths for KNN classifier by accuracy drop:")
    print(cls_summary_df.head(13).to_string(index=False))

    print("\nDONE. Outputs saved inside:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
