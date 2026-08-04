"""Create processed Training and Validation images and their feature CSV files."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data"
IMAGES_DIR = DATA_DIR / "Images"
DATA_FILES_DIR = DATA_DIR / "Data_files"

SLOPE = 0.03
WAVELENGTHS = [
    "365", "405", "473", "530", "575", "621", "660",
    "735", "770", "830", "850", "890", "940",
]
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


def classify_usda_soil_texture(clay, sand, silt):
    """Return the USDA texture class after normalizing composition to 100%."""
    total = clay + sand + silt
    if total == 0:
        return "Unknown"

    clay = clay * 100 / total
    sand = sand * 100 / total
    silt = silt * 100 / total

    if clay >= 40:
        if silt >= 40:
            return "Silty Clay"
        if sand >= 45:
            return "Sandy Clay"
        return "Clay"
    if 35 <= clay < 40:
        if sand >= 45:
            return "Sandy Clay"
        if sand <= 20:
            return "Silty Clay Loam"
        return "Clay Loam"
    if 27 <= clay < 35:
        if sand >= 45:
            return "Sandy Clay Loam"
        if sand <= 20:
            return "Silty Clay Loam"
        return "Clay Loam"
    if 20 <= clay < 27 and sand >= 45:
        return "Sandy Clay Loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand < 52:
        return "Loam"
    if silt >= 50 and clay < 27:
        if silt >= 80 and clay < 12:
            return "Silt"
        return "Silt Loam"
    if clay < 20 and sand >= 52 and (silt + 2 * clay) > 30:
        return "Sandy Loam"
    if clay < 7 and 43 <= sand < 52 and silt < 50:
        return "Sandy Loam"
    if sand >= 85 and (silt + 1.5 * clay) < 15:
        return "Sand"
    if 70 <= sand < 90 and clay < 15 and (silt + 1.5 * clay) >= 15:
        return "Loamy Sand"
    return "Unclassified"


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

        subtracted = cv2.subtract(band_image, base_image)
        crop = subtracted[y1:y2, x1:x2]
        mean = float(np.mean(crop))
        std = float(np.std(crop))
        corrected = (mean - std) + std * (
            np.tanh(SLOPE * (crop.astype(np.float64) - mean)) + 1.0
        )

        # Use the same uint8 array for image saving and CSV feature extraction.
        processed_bands[wavelength] = np.clip(corrected, 0, 255).astype(np.uint8)

    return processed_bands


def save_processed_images(processed_bands, output_dir, specimen_name):
    specimen_output_dir = output_dir / specimen_name
    specimen_output_dir.mkdir(parents=True, exist_ok=True)

    for wavelength, image in processed_bands.items():
        output_path = specimen_output_dir / f"{wavelength}nm.png"
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Failed to save processed image: {output_path}")


def append_feature_rows(rows, processed_bands, clay, sand, silt, soil_type):
    """Create 100 block-level feature rows directly from in-memory images."""
    image_height, image_width = next(iter(processed_bands.values())).shape
    if image_height % BLOCK_SIZE or image_width % BLOCK_SIZE:
        raise ValueError(
            f"Processed image shape {(image_height, image_width)} is not divisible "
            f"by block size {BLOCK_SIZE}."
        )

    for y in range(0, image_height, BLOCK_SIZE):
        for x in range(0, image_width, BLOCK_SIZE):
            row = [
                float(np.mean(processed_bands[wavelength][y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]))
                for wavelength in WAVELENGTHS
            ]
            row.extend([clay, sand, silt, soil_type])
            rows.append(row)


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

    for composition_dir in composition_dirs:
        clay, sand, silt = mixture_composition(composition_dir.name)
        soil_type = classify_usda_soil_texture(clay, sand, silt)

        for specimen_index, specimen_dir in enumerate(specimen_directories(composition_dir)):
            print(f"[{split_name}] Processing {specimen_dir}")
            processed_bands = process_specimen(specimen_dir)
            specimen_name = (
                f"c_{clay}_sa_{sand}_si_{silt}_{specimen_index}_0_0"
            )
            save_processed_images(processed_bands, output_dir, specimen_name)
            append_feature_rows(
                rows, processed_bands, clay, sand, silt, soil_type
            )
            specimen_count += 1

    columns = WAVELENGTHS + ["Clay", "Sand", "Silt", "Soil_Type"]
    dataframe = pd.DataFrame(rows, columns=columns)
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
