# Reflectance Multispectral Imaging for Soil Grain-Size Composition Estimation and USDA Texture Classification

This repository is the official implementation of the 2026 IEEE Access article **“Reflectance Multispectral Imaging for Soil Grain-Size Composition Estimation and USDA Texture Classification.”** It implements a custom 13-band reflectance multispectral imaging (MSI) workflow spanning 365–940 nm for grain-size composition regression, direct USDA texture classification, and indirect USDA classification through the texture triangle.

## Paper information

G. A. S. L. Ranasinghe, J. A. S. T. Jayakody, M. C. L. De Silva, G. Thilakarathne, G. M. R. I. Godaliyadda, H. M. V. R. Herath, M. P. B. Ekanayake, and S. K. Navaratnarajah, “Reflectance Multispectral Imaging for Soil Grain-Size Composition Estimation and USDA Texture Classification,” *IEEE Access*, vol. 14, pp. 128744–128763, 2026, doi: [10.1109/ACCESS.2026.3721276](https://doi.org/10.1109/ACCESS.2026.3721276). [IEEE Xplore record](https://ieeexplore.ieee.org/document/11643937/)

## Method overview

```text
Raw multispectral images
→ dark-current correction: max(X(λ) − D, 0)
→ identical fixed 100×100 ROI in all 13 bands
→ per-band bounded tanh normalization (κ = 0.03)
→ 10×10 non-overlapping blocks (100 blocks/specimen)
→ 13 mean spectral features/block
→ fold-specific training-only min-max scaling
→ fold-specific training-only LDA (99% cumulative discriminative power; 5 components)
→ direct USDA classification (training-only SMOTE)
or
→ clay/silt/sand regression
→ raw specimen-level regression evaluation
→ Eq. (14) projection and USDA triangle mapping for indirect classification only
```

The physical specimen—not an individual image block—is the experimental unit. Every evaluation path keeps all 100 blocks from a specimen in one fold and computes final metrics at specimen level.

## Dataset design

| Dataset | Mixture ratios | Specimens per ratio | Physical specimens | Block observations |
|---|---:|---:|---:|---:|
| Training/testing | 22 | 20 | 440 | 44,000 |
| External validation | 7 | 12 | 84 | 8,400 |

The external-validation ratios are unseen/intermediate ratios prepared from the same three source soils as the training/testing set. They therefore test interpolation across mixture ratios, not generalization to geographically or mineralogically independent soils.

The wavelengths are **365, 405, 473, 530, 575, 621, 660, 735, 770, 830, 850, 890, and 940 nm**.

### Data availability

The data supporting the study are available from the corresponding author upon reasonable request. The private image and feature datasets are not included in this repository.

## Repository structure

- `Make_Datafiles.py` — dark-current correction, fixed ROI, tanh mapping, result-compatible uint8 quantization, processed-image export, and 100×13 block-feature generation with explicit specimen IDs.
- `scientific_utils.py` — canonical constants, dataset validation, legacy specimen-ID compatibility, leakage checks, training-only scaler/LDA, and raw specimen aggregation.
- `usda_texture.py` — the single USDA triangle implementation and the separate Eq. (14) compositional projection.
- `Direct_Soil_Classification.py` — six direct classifiers, grouped CV, training-only SMOTE, Figure 12, and conditional Cochran Q/McNemar/Holm tests.
- `Soil_Composition_Prediction_and_Indirect_Classification.py` — six regressors, raw regression evaluation, indirect classification, external validation, Figures 13–15, and conditional statistical tests.
- `KNN_permutation_importance.py` — grouped specimen-pattern wavelength permutation and Figure 16.
- `Plot_Spectral_Signatures_and_LDA.py` — qualitative whole-dataset Figures 8–11. These LDA fits are visualizations, not held-out evaluations.
- `benchmark_inference.py` — hardware-dependent KNN model-level timing with preprocessing, training, disk I/O, and plotting excluded.
- `tests/` — private-data-independent tests for USDA regions/boundaries, Eq. (14), grouping, leakage, wavelength dimensions, LDA selection, and preprocessing equations.

Output directories now use the consistent names shown above. The earlier untracked names `02_Soil_Composition_Prediction`, `03_Indirect Soil_Classfication`, `KNN_spectral_band_importance`, and `04_Spectral_Signatures_and_LDA` are no longer written; existing local outputs under those paths are not deleted.

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/sandunikalakmali/soil-texture-msi.git
cd soil-texture-msi
python -m pip install -r requirements.txt
```

The historical package versions used to produce the paper could not be recovered from the repository. `requirements.txt` consequently lists the required packages without fabricated pins. Record `python -m pip freeze` when reproducing results; XGBoost/CatBoost defaults and timing can vary across releases.

## Expected data layout

```text
Data/
├── Images/
│   ├── Training/
│   │   └── <mixture-folder>/<level>/<physical-specimen>/
│   │       ├── 000nm.png
│   │       ├── 365nm.png
│   │       ├── ...
│   │       └── 940nm.png
│   └── Validation/
│       └── <mixture-folder>/<level>/<physical-specimen>/
│           └── <same 14 image names>
└── Data_files/                         # created by Make_Datafiles.py
```

Mixture folder names must retain the acquisition convention used by the study: underscore-delimited clay, sand, and silt source fractions at token positions 2, 4, and 6 (zero-based), as validated by `mixture_composition()`. Each physical-specimen directory must contain a dark frame (`000nm.png`) and all 13 wavelength images.

New CSVs store `Specimen_ID`, `Block_Row`, and `Block_Column`. Legacy paper CSVs without `Specimen_ID` remain supported only through a warning-producing, validated fallback of 100 consecutive rows per specimen. The scripts reject incomplete official datasets, nonfinite values, inconsistent within-specimen targets, noncanonical USDA labels, and train/test specimen overlap.

## Reproducing the experiments

Run from the repository root:

```bash
# Preprocessing and feature generation (Training and Validation)
python Make_Datafiles.py

# Direct classification, Figure 12, and direct statistical tests
python Direct_Soil_Classification.py

# Regression, indirect classification, external validation, Figures 13–15,
# Friedman/Wilcoxon/Holm and Cochran Q/McNemar/Holm tests
python Soil_Composition_Prediction_and_Indirect_Classification.py

# Figure 16 permutation band importance
python KNN_permutation_importance.py

# Figures 8–11 spectral signatures and qualitative LDA projections
python Plot_Spectral_Signatures_and_LDA.py

# Model-level inference benchmark
python benchmark_inference.py

# Unit tests
python -m pytest -q
```

Each analysis command accepts `--help`, explicit input/output paths, and—only for exploratory use—`--allow-nonofficial-dimensions`. Official reproduction should not use that override.

## Evaluation protocol

- Five-fold `StratifiedGroupKFold`, shuffled deterministically with seed **42**, groups by physical specimen.
- All 100 block observations from a specimen remain together; every one of the 440 specimens appears exactly once in pooled out-of-fold predictions.
- `MinMaxScaler` is fitted only on training specimens in a fold. External validation uses a scaler fitted on the complete 440-specimen training/testing dataset.
- Direct LDA is supervised by USDA texture class. Regression/indirect LDA is separately supervised by the 22 unique ground-truth composition groups. The 99% cumulative-power rule is retained and the official path asserts five components.
- Every regression fold asserts that all 22 composition groups remain represented among its training specimens.
- SMOTE is used only for direct classification, after the fold split and fold-only scaler/LDA, and never touches held-out or external-validation data.
- Direct specimen predictions average 100 class-probability vectors before argmax. Regression predictions average 100 raw block predictions per component.
- R², RMSE, and regression statistical errors use raw averaged clay/silt/sand predictions. Eq. (14)—negative clipping followed by rescaling to 100%—exists only in the indirect-classification branch.
- Figure 13 uses pooled KNN out-of-fold predictions for all 440 specimens. Figure 14 uses raw KNN predictions for all 84 external specimens after complete-development-set fitting. Figures 12 and 15 are normalized, pooled, specimen-level KNN confusion matrices in one canonical 12-class order.

### Statistical sequence

- Direct and indirect classification: specimen correctness → Cochran Q → only if `p < 0.05`, highest-accuracy model versus each remaining model by paired McNemar tests → Holm correction.
- Regression: per-specimen `sqrt(mean((true − predicted)² across clay, silt, sand))` → Friedman test → only if `p < 0.05`, lowest-error model versus each remaining model by paired one-sided Wilcoxon signed-rank tests → Holm correction.

### Result-generating implementation details

Two details are retained because changing them can alter numerical reproduction:

1. After the floating-point tanh mapping, the original feature-generation code clipped to `[0, 255]`, converted to `uint8`, and calculated block means from that quantized array. The paper gives the mathematical mapping but does not describe this quantization. The repository now documents and tests the behavior instead of silently changing it.
2. The original Figure 16 source used 10 deterministic held-out permutations per wavelength in each fold. The paper describes averaging across five folds but does not state the within-fold repeat count. The code retains 10 and saves both repeat-level and summarized outputs.

The published Figures 8 and 9 visibly use shaded variability regions, but neither their captions nor the available repository history identifies those regions as standard deviation, standard error, confidence intervals, or another statistic. `Plot_Spectral_Signatures_and_LDA.py` therefore reproduces the documented normalized mean curves without assigning an invented meaning to the shading.

## Published-result sanity checks

These values are checks, never constants used by the implementation:

| Evaluation | KNN reference |
|---|---:|
| Direct grouped-CV specimen accuracy | ≈ 0.9286 |
| Clay grouped-CV R² | ≈ 0.9869 |
| Silt grouped-CV R² | ≈ 0.9892 |
| Sand grouped-CV R² | ≈ 0.9860 |
| Indirect grouped-CV specimen accuracy | ≈ 0.8941 |
| External clay R² / RMSE | ≈ 0.9865 / 2.3616 |
| External silt R² / RMSE | ≈ 0.9665 / 4.4470 |
| External sand R² / RMSE | ≈ 0.9618 / 4.4958 |

Exact grouped-CV RMSE values should be compared directly with Table 3 of the version of record. They are not fabricated here because the final PDF and private data are absent from this checkout. Minor numerical differences can also arise from unrecovered historical package versions.

The paper reports model-level inference on an AMD Ryzen 7 6800HS with 16 GB RAM: direct KNN 1.33 ms/specimen; indirect KNN regression 1.15 ms/specimen; USDA mapping 0.009 ms/specimen; total 1.159 ms/specimen. `benchmark_inference.py` defines the timed boundaries precisely, but other hardware will not reproduce the same wall-clock values.

## Model configuration

The scripts preserve the checked-in result-generating constructors: KNN uses five uniform Euclidean neighbors; direct RF uses 500 trees; direct CatBoost uses 500 iterations, learning rate 0.05, and depth 6; direct XGBoost uses 500 estimators, learning rate 0.05, depth 6, 0.9 row/column subsampling; regression RF uses 500 trees, depth 15, leaf size 5, split size 10, and square-root features. Stochastic estimators, splitting, SMOTE, and permutations use seed 42. Regression CatBoost/XGBoost parameters that were implicit in the original scripts remain library defaults because the historical versions are unknown; guessing values would undermine published-result reproduction.

## Scope and limitation

This study is a proof of concept based on controlled mixtures made from three source soils. It demonstrates discrimination and interpolation within that mixture system; it does not establish broad field generalization across sites, parent materials, mineralogy, organic matter, moisture states, or acquisition conditions. Independent, geographically diverse field validation remains necessary.

## License and citation

The repository already carried an MIT software license from its initial commit; see `LICENSE`. The article's CC BY 4.0 publication license is separate and was not used to infer the software license.

```bibtex
@article{ranasinghe2026reflectance,
  author  = {G. A. S. L. Ranasinghe and J. A. S. T. Jayakody and M. C. L. De Silva and G. Thilakarathne and G. M. R. I. Godaliyadda and H. M. V. R. Herath and M. P. B. Ekanayake and S. K. Navaratnarajah},
  title   = {Reflectance Multispectral Imaging for Soil Grain-Size Composition Estimation and USDA Texture Classification},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {128744--128763},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3721276}
}
```

Machine-readable citation metadata are in `CITATION.cff`.
