"""Generate the spectral-signature and LDA visualizations (Figures 8--11).

The input is the Training CSV produced by Make_Datafiles.py. Figures are saved
under outputs/04_Spectral_Signatures_and_LDA.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_CSV_PATH = (
    SCRIPT_DIR
    / "Data"
    / "Data_files"
    / "dataAll_Soil_Composition_Training_Histogram_No_Augmentation_"
    "Adjusted_Files_1crop_0_03_sigma_subOperator_SuperPixel.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "04_Spectral_Signatures_and_LDA"

BAND_COLS = [
    "365",
    "405",
    "473",
    "530",
    "575",
    "621",
    "660",
    "735",
    "770",
    "830",
    "850",
    "890",
    "940",
]
BLOCKS_PER_SPECIMEN = 100

SOILTYPE_GROUPS = {
    "Sand group": ["Sand"],
    "Mixed sand-loam-clay group": [
        "Loamy Sand",
        "Sandy Loam",
        "Loam",
        "Clay Loam",
        "Sandy Clay Loam",
        "Sandy Clay",
        "Clay",
    ],
    "Silt group": ["Silt", "Silt Loam", "Silty Clay Loam", "Silty Clay"],
}

SOILTYPE_COLOR_MAP = {
    "Sand": "#D62728",
    "Loamy Sand": "#9ECAE1",
    "Sandy Loam": "#6BAED6",
    "Loam": "#2171B5",
    "Clay Loam": "#08519C",
    "Sandy Clay Loam": "#08306B",
    "Sandy Clay": "#54278F",
    "Clay": "#0B0024",
    "Silt": "#74C476",
    "Silt Loam": "#41AB5D",
    "Silty Clay Loam": "#238B45",
    "Silty Clay": "#006D2C",
}

COMPOSITION_GROUPS = {
    "Sand group": ["Cl0-M0-S100"],
    "Mixed sand-loam-clay group": [
        "Cl2-M12-S86",
        "Cl4-M19-S78",
        "Cl5-M28-S67",
        "Cl15-M4-S81",
        "Cl21-M39-S40",
        "Cl9-M46-S44",
        "Cl29-M8-S63",
        "Cl32-M13-S55",
        "Cl34-M31-S35",
        "Cl35-M10-S55",
        "Cl39-M11-S50",
        "Cl44-M12-S44",
        "Cl46-M24-S30",
        "Cl58-M16-S26",
        "Cl79-M21-S0",
    ],
    "Silt group": [
        "Cl4-M60-S36",
        "Cl19-M51-S30",
        "Cl5-M87-S8",
        "Cl6-M94-S0",
        "Cl31-M59-S10",
        "Cl42-M53-S5",
    ],
}

COMPOSITION_COLOR_MAP = {
    "Cl0-M0-S100": "#D62728",
    "Cl2-M12-S86": "#9ECAE1",
    "Cl4-M19-S78": "#6BAED6",
    "Cl5-M28-S67": "#4292C6",
    "Cl15-M4-S81": "#2171B5",
    "Cl21-M39-S40": "#08519C",
    "Cl9-M46-S44": "#08306B",
    "Cl29-M8-S63": "#2C7FB8",
    "Cl32-M13-S55": "#253494",
    "Cl34-M31-S35": "#756BB1",
    "Cl35-M10-S55": "#6A51A3",
    "Cl39-M11-S50": "#54278F",
    "Cl44-M12-S44": "#3F007D",
    "Cl46-M24-S30": "#2B005A",
    "Cl58-M16-S26": "#1D003F",
    "Cl79-M21-S0": "#0B0024",
    "Cl4-M60-S36": "#A1D99B",
    "Cl19-M51-S30": "#74C476",
    "Cl5-M87-S8": "#41AB5D",
    "Cl6-M94-S0": "#238B45",
    "Cl31-M59-S10": "#006D2C",
    "Cl42-M53-S5": "#00441B",
}


def ordered_labels(groups):
    """Flatten grouped labels while retaining the requested legend order."""
    return [label for labels in groups.values() for label in labels]


def composition_labels(df):
    """Return labels in the paper's Cl{clay}-M{silt}-S{sand} format."""
    clay = df["Clay"].round().astype(int).astype(str)
    silt = df["Silt"].round().astype(int).astype(str)
    sand = df["Sand"].round().astype(int).astype(str)
    return "Cl" + clay + "-M" + silt + "-S" + sand


def validate_input(df):
    required = BAND_COLS + ["Clay", "Sand", "Silt", "Soil_Type"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {missing}")
    if len(df) % BLOCKS_PER_SPECIMEN:
        raise ValueError(
            f"CSV has {len(df)} rows, which is not divisible by "
            f"BLOCKS_PER_SPECIMEN={BLOCKS_PER_SPECIMEN}."
        )


def validate_color_coverage(labels, color_map, label_kind):
    missing = sorted(set(labels) - set(color_map))
    if missing:
        raise ValueError(
            f"No requested color is defined for these {label_kind} labels: {missing}"
        )


def make_specimen_table(df):
    """Average each contiguous 100-row block into one physical specimen."""
    work = df.copy()
    work["Specimen_ID"] = np.arange(len(work)) // BLOCKS_PER_SPECIMEN
    aggregations = {band: "mean" for band in BAND_COLS}
    aggregations.update(
        {
            "Clay": "first",
            "Sand": "first",
            "Silt": "first",
            "Soil_Type": "first",
        }
    )
    specimen_df = work.groupby("Specimen_ID", sort=True).agg(aggregations)
    specimen_df["Composition"] = composition_labels(specimen_df)
    return specimen_df


def plot_signatures(data, label_col, groups, color_map, legend_title, output_path):
    """Plot group-wise mean curves after per-band min-max normalization."""
    labels = data[label_col].astype(str)
    validate_color_coverage(labels.unique(), color_map, label_col)
    normalized = MinMaxScaler().fit_transform(data[BAND_COLS].to_numpy(float))
    normalized_df = pd.DataFrame(normalized, columns=BAND_COLS, index=data.index)
    normalized_df[label_col] = labels.to_numpy()
    means = normalized_df.groupby(label_col)[BAND_COLS].mean()

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    wavelengths = np.asarray(BAND_COLS, dtype=int)
    for label in ordered_labels(groups):
        if label not in means.index:
            continue
        ax.plot(
            wavelengths,
            means.loc[label].to_numpy(float),
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            color=color_map[label],
            label=label,
        )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized mean intensity")
    ax.set_xticks(wavelengths)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linewidth=0.35, alpha=0.4)
    ax.legend(
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lda(df, labels, groups, color_map, legend_title, output_path):
    """Fit two-component LDA to scaled block features and save its projection."""
    labels = pd.Series(labels, index=df.index).astype(str).to_numpy()
    validate_color_coverage(np.unique(labels), color_map, legend_title)

    imputed = SimpleImputer(strategy="median").fit_transform(df[BAND_COLS])
    X_scaled = MinMaxScaler().fit_transform(imputed)
    lda = LinearDiscriminantAnalysis(n_components=2)
    projected = lda.fit_transform(X_scaled, labels)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    present = set(labels)
    for label in ordered_labels(groups):
        if label not in present:
            continue
        selected = labels == label
        ax.scatter(
            projected[selected, 0],
            projected[selected, 1],
            s=10,
            alpha=0.80,
            color=color_map[label],
            edgecolors="none",
        )

    ax.set_xlabel("LDA1", fontsize=12)
    ax.set_ylabel("LDA2", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="none",
            markerfacecolor=color_map[label],
            markeredgecolor="none",
            markersize=6,
            label=label,
        )
        for label in ordered_labels(groups)
        if label in present
    ]
    ax.legend(
        handles=handles,
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        labelspacing=0.35,
        handletextpad=0.4,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    if not TRAIN_CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Training CSV not found: {TRAIN_CSV_PATH}\n"
            "Run Make_Datafiles.py first to create it."
        )

    df = pd.read_csv(TRAIN_CSV_PATH)
    validate_input(df)
    df["Composition"] = composition_labels(df)
    specimen_df = make_specimen_table(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "Figure_08_Spectral_Signatures_USDA_Classes.png": lambda path: plot_signatures(
            specimen_df,
            "Soil_Type",
            SOILTYPE_GROUPS,
            SOILTYPE_COLOR_MAP,
            "USDA soil texture class",
            path,
        ),
        "Figure_09_Spectral_Signatures_Composition_Groups.png": lambda path: plot_signatures(
            specimen_df,
            "Composition",
            COMPOSITION_GROUPS,
            COMPOSITION_COLOR_MAP,
            "Composition (Clay/Silt/Sand %)",
            path,
        ),
        "Figure_10_LDA_USDA_Classes.png": lambda path: plot_lda(
            df,
            df["Soil_Type"],
            SOILTYPE_GROUPS,
            SOILTYPE_COLOR_MAP,
            "USDA soil texture class",
            path,
        ),
        "Figure_11_LDA_Composition_Groups.png": lambda path: plot_lda(
            df,
            df["Composition"],
            COMPOSITION_GROUPS,
            COMPOSITION_COLOR_MAP,
            "Composition (Clay/Silt/Sand %)",
            path,
        ),
    }

    for filename, create_figure in outputs.items():
        output_path = OUTPUT_DIR / filename
        create_figure(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
