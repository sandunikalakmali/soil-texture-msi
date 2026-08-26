"""Create the paper's block-feature CSVs from raw multispectral images.

The uint8 conversion before block averaging is intentionally preserved from
the result-generating repository implementation. It is a quantization step in
the published-result reproduction path, not part of the mathematical tanh
mapping itself; floating-point arrays should be used in future experiments
only as an explicitly versioned methodological variant.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scientific_utils import (
    BLOCKS_PER_SPECIMEN,
    EXPECTED_DATASETS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    SPECIMEN_COLUMN,
    TARGET_COLUMNS,
    WAVELENGTHS_NM,
)
from usda_texture import classify_usda_texture


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data"
IMAGES_DIR = DATA_DIR / "Images"
DATA_FILES_DIR = DATA_DIR / "Data_files"

TANH_KAPPA = 0.03
WAVELENGTHS = list(FEATURE_COLUMNS)
REGION = (775, 875, 350, 450)  # y1, y2, x1, x2
BLOCK_SIZE = 10

CSV_SUFFIX = (
    "Histogram_No_Augmentation_Adjusted_Files_1crop_0_03_sigma_"
    "subOperator_SuperPixel"
)

# Laboratory-measured composition of the three source soils.
CLAY_SOURCE = {"Clay": 78.634, "Sand": 0.000, "Silt": 21.366}
SAND_SOURCE = {"Clay": 0.000, "Sand": 100.000, "Silt": 0.000}
SILT_SOURCE = {"Clay": 5.750, "Sand": 0.000, "Silt": 94.250}


def mixture_composition(folder_name):
    """Calculate final Clay/Sand/Silt percentages from a mixture folder name."""
    parts = folder_name.split("_")
    if len(parts) < 7:
        raise ValueError(
            f"Composition folder '{folder_name}' does not contain expected "
            "mixture values at positions 2, 4, and 6."
        )

    try:
        clay_fraction = float(parts[2])
        sand_fraction = float(parts[4])
        silt_fraction = float(parts[6])
    except ValueError as exc:
        raise ValueError(f"Cannot parse mixture values from '{folder_name}'.") from exc

    clay = round(
        CLAY_SOURCE["Clay"] * clay_fraction
        + SAND_SOURCE["Clay"] * sand_fraction
        + SILT_SOURCE["Clay"] * silt_fraction,
        2,
    )
    sand = round(
        CLAY_SOURCE["Sand"] * clay_fraction
        + SAND_SOURCE["Sand"] * sand_fraction
        + SILT_SOURCE["Sand"] * silt_fraction,
        2,
    )
    silt = round(
        CLAY_SOURCE["Silt"] * clay_fraction
        + SAND_SOURCE["Silt"] * sand_fraction
        + SILT_SOURCE["Silt"] * silt_fraction,
        2,
    )
    return clay, sand, silt


def specimen_directories(composition_dir):
    """Yield third-level specimen directories in deterministic order."""
    for level_two in sorted(path for path in composition_dir.iterdir() if path.is_dir()):
        for specimen_dir in sorted(path for path in level_two.iterdir() if path.is_dir()):
            yield specimen_dir


def read_grayscale(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read required grayscale image: {path}")
    return image


def bounded_tanh_normalize(roi):
    """Apply the bounded per-band mapping in Section III-E, Eqs. (3)--(5)."""

    roi_float = np.asarray(roi, dtype=np.float64)
    mean = float(np.mean(roi_float))
    std = float(np.std(roi_float))
    return (mean - std) + 2.0 * std * (
        (np.tanh(TANH_KAPPA * (roi_float - mean)) + 1.0) / 2.0
    )


def process_specimen(specimen_dir):
    """Return all thirteen processed 100x100 bands in memory."""
    y1, y2, x1, x2 = REGION
    base_image = read_grayscale(specimen_dir / "000nm.png")

    if base_image.shape[0] < y2 or base_image.shape[1] < x2:
        raise ValueError(
            f"Image {specimen_dir / '000nm.png'} has shape {base_image.shape}; "
            f"the configured ROI requires at least ({y2}, {x2})."
        )

    processed_bands = {}
    for wavelength in WAVELENGTHS:
        band_image = read_grayscale(specimen_dir / f"{wavelength}nm.png")
        if band_image.shape != base_image.shape:
            raise ValueError(
                f"Shape mismatch in {specimen_dir}: base={base_image.shape}, "
                f"{wavelength}nm={band_image.shape}."
            )

        # Paper Eq. (1): Y(lambda) = max(X(lambda) - D, 0). OpenCV's
        # subtraction is saturated for uint8 inputs, so it implements the
        # stated correction; an absolute difference would be incorrect.
        subtracted = cv2.subtract(band_image, base_image)
        crop = subtracted[y1:y2, x1:x2]
        if crop.shape != (100, 100):
            raise AssertionError(f"Configured ROI produced {crop.shape}, expected (100, 100).")
        corrected = bounded_tanh_normalize(crop)

        # Reproducibility note: the original implementation extracted features
        # from this clipped/quantized uint8 array. Preserve that numerical path.
        processed_bands[wavelength] = np.clip(corrected, 0, 255).astype(np.uint8)

    return processed_bands


def save_processed_images(processed_bands, output_dir, specimen_name):
    specimen_output_dir = output_dir / specimen_name
    specimen_output_dir.mkdir(parents=True, exist_ok=True)

    for wavelength, image in processed_bands.items():
        output_path = specimen_output_dir / f"{wavelength}nm.png"
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Failed to save processed image: {output_path}")


def append_feature_rows(
    rows, processed_bands, specimen_id, clay, silt, sand, soil_type
):
    """Create 100 block-level feature rows directly from in-memory images."""
    image_height, image_width = next(iter(processed_bands.values())).shape
    if image_height % BLOCK_SIZE or image_width % BLOCK_SIZE:
        raise ValueError(
            f"Processed image shape {(image_height, image_width)} is not divisible "
            f"by block size {BLOCK_SIZE}."
        )

    before = len(rows)
    for block_row, y in enumerate(range(0, image_height, BLOCK_SIZE)):
        for block_column, x in enumerate(range(0, image_width, BLOCK_SIZE)):
            row = {
                wavelength: float(
                    np.mean(processed_bands[wavelength][y:y + BLOCK_SIZE, x:x + BLOCK_SIZE])
                )
                for wavelength in WAVELENGTHS
            }
            row.update(
                {
                    SPECIMEN_COLUMN: specimen_id,
                    "Block_Row": block_row,
                    "Block_Column": block_column,
                    "Clay": clay,
                    "Silt": silt,
                    "Sand": sand,
                    LABEL_COLUMN: soil_type,
                }
            )
            rows.append(row)
    if len(rows) - before != BLOCKS_PER_SPECIMEN:
        raise AssertionError("Each physical specimen must produce exactly 100 block rows.")


def process_split(split_name):
    """Process one image split, save its images, and create its CSV."""
    input_dir = IMAGES_DIR / split_name
    output_dir = DATA_DIR / f"cropped_images_{split_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    DATA_FILES_DIR.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input image directory was not found: {input_dir}")

    rows = []
    specimen_count = 0
    composition_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    expected_key = "training/testing" if split_name == "Training" else "external validation"
    expected = EXPECTED_DATASETS[expected_key]
    if len(composition_dirs) != expected["composition_groups"]:
        raise ValueError(
            f"{split_name}: found {len(composition_dirs)} mixture directories; "
            f"expected {expected['composition_groups']}."
        )

    for composition_dir in composition_dirs:
        clay, sand, silt = mixture_composition(composition_dir.name)
        soil_type = classify_usda_texture(clay, silt, sand)

        for specimen_index, specimen_dir in enumerate(specimen_directories(composition_dir)):
            print(f"[{split_name}] Processing {specimen_dir}")
            processed_bands = process_specimen(specimen_dir)
            specimen_name = (
                f"c_{clay}_sa_{sand}_si_{silt}_{specimen_index}_0_0"
            )
            specimen_id = (
                f"{split_name}:{composition_dir.name}:"
                f"{specimen_dir.relative_to(composition_dir).as_posix()}"
            )
            save_processed_images(processed_bands, output_dir, specimen_name)
            append_feature_rows(
                rows, processed_bands, specimen_id, clay, silt, sand, soil_type
            )
            specimen_count += 1

    columns = WAVELENGTHS + [
        SPECIMEN_COLUMN,
        "Block_Row",
        "Block_Column",
        *TARGET_COLUMNS,
        LABEL_COLUMN,
    ]
    dataframe = pd.DataFrame(rows, columns=columns)
    actual_groups = dataframe[list(TARGET_COLUMNS)].drop_duplicates()
    if specimen_count != expected["specimens"] or len(dataframe) != expected["rows"]:
        raise ValueError(
            f"{split_name} is incomplete: found {specimen_count} specimens and "
            f"{len(dataframe)} rows; expected {expected['specimens']} and {expected['rows']}."
        )
    if len(actual_groups) != expected["composition_groups"]:
        raise ValueError(
            f"{split_name}: found {len(actual_groups)} composition groups; "
            f"expected {expected['composition_groups']}."
        )
    csv_path = DATA_FILES_DIR / f"dataAll_Soil_Composition_{split_name}_{CSV_SUFFIX}.csv"
    dataframe.to_csv(csv_path, index=False)

    print(
        f"[{split_name}] Saved {specimen_count} specimens, {len(dataframe)} rows, "
        f"and {len(WAVELENGTHS) * specimen_count} processed images."
    )
    print(f"[{split_name}] CSV saved to: {csv_path}")
    return dataframe


def main():
    datasets = {}
    for split_name in ("Training", "Validation"):
        datasets[split_name] = process_split(split_name)
    return datasets


if __name__ == "__main__":
    main()
