# UrbanSound8K Exploratory Data Analysis (EDA) Report

- **Total Audio Records**: 8,732
- **Feature Dimension**: 62 audio descriptors
- **Generated Visualizations**:
  - `eda/plots/class_distribution.png`: Sound category counts
  - `eda/plots/fold_class_distribution.png`: Fold vs class distribution
  - `eda/plots/feature_correlation_heatmap.png`: 62×62 correlation heatmap
  - `eda/plots/spectral_features_distribution.png`: Spectral descriptors distribution
  - `eda/plots/mfcc_class_separability.png`: MFCC boxplots across classes

## 1. Class Balance Assessment

The dataset contains 10 environmental sound categories:
| Class Name | Samples | Percentage |
| --- | --- | --- |
| `dog_bark` | 1,000 | 11.45% |
| `children_playing` | 1,000 | 11.45% |
| `air_conditioner` | 1,000 | 11.45% |
| `street_music` | 1,000 | 11.45% |
| `engine_idling` | 1,000 | 11.45% |
| `jackhammer` | 1,000 | 11.45% |
| `drilling` | 1,000 | 11.45% |
| `siren` | 929 | 10.64% |
| `car_horn` | 429 | 4.91% |
| `gun_shot` | 374 | 4.28% |

*Note: 8 classes contain exactly 1,000 samples, car_horn contains 429 samples, and gun_shot contains 374 samples, reflecting standard UrbanSound8K real-world urban acoustic conditions.*

## 2. High Feature Collinearity (|r| >= 0.85)

Detected **6** strongly correlated feature pairs. Notable pairs:

| Feature 1 | Feature 2 | Correlation (r) |
| --- | --- | --- |
| `mean_spectral_rolloff` | `mean_spectral_zcr` | 0.9638 |
| `mean_spectral_centroid` | `mean_spectral_zcr` | 0.9271 |
| `mean_spectral_centroid` | `mean_spectral_rolloff` | 0.9184 |
| `std_spectral_spread` | `std_mfcc_0` | 0.8841 |
| `std_spectral_rolloff` | `std_spectral_zcr` | 0.8624 |
| `std_spectral_centroid` | `std_spectral_spread` | 0.8579 |

## 3. Outlier Analysis Findings

Outliers detected via standard 1.5×IQR and 3-sigma thresholds are intrinsic acoustic variations (e.g., intense impulses in gun_shot, sustained high-frequency whine in drilling, low hums in engine_idling). **No records are removed**, in accordance with project constraints and scientific rigor.

## 4. Feature Summary Table (Top 15 Features by Variance)

| Feature | Mean | Std | Min | Median | Max | Skewness |
| --- | --- | --- | --- | --- | --- | --- |
| `mean_spectral_rolloff` | 6854.5153 | 4051.1122 | 103.66 | 6126.95 | 21654.9 | 0.6606 |
| `mean_spectral_centroid` | 6061.411 | 2249.8835 | 354.584 | 5861.585 | 16197.3 | 0.4754 |
| `std_spectral_rolloff` | 1780.0402 | 1479.1896 | 0.0068 | 1381.035 | 9429.31 | 1.6765 |
| `mean_spectral_spread` | 4447.9972 | 1133.842 | 262.04 | 4532.735 | 8166.59 | -0.2665 |
| `std_spectral_centroid` | 831.5682 | 752.3015 | 87.9675 | 593.976 | 6065.77 | 2.5444 |
| `std_spectral_flux` | 121.94 | 578.0032 | 0.0 | 5.1328 | 24224.3 | 17.0688 |
| `std_spectral_spread` | 451.3447 | 467.9618 | 22.9856 | 313.3705 | 3287.3 | 3.0065 |
| `mean_spectral_flux` | 101.3537 | 458.4816 | 0.0 | 7.0008 | 14062.1 | 15.4277 |
| `mean_mfcc_0` | -158.2424 | 150.3331 | -1042.73 | -133.5675 | 147.63 | -1.9005 |
| `std_mfcc_0` | 48.4913 | 92.8249 | 2.3326 | 17.9139 | 627.082 | 3.6186 |
| `mean_mfcc_1` | 2.7597 | 34.7521 | -131.696 | 5.0821 | 152.878 | -0.3205 |
| `mean_mfcc_2` | -29.3833 | 25.8405 | -145.712 | -26.4725 | 50.6245 | -0.8519 |
| `mean_mfcc_3` | 1.748 | 17.0298 | -104.489 | 2.097 | 66.5068 | -0.4761 |
| `mean_mfcc_4` | -12.8173 | 13.421 | -71.2775 | -11.0562 | 47.4082 | -0.5806 |
| `mean_mfcc_5` | 3.0721 | 10.8885 | -55.7596 | 2.9801 | 52.6198 | -0.1406 |
