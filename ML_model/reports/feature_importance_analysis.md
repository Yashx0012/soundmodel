# UrbanSound8K Feature Importance and Acoustic Influence Analysis

- Analyzed Model: `LGBMClassifier`
- Total Evaluated Features: 62 audio descriptors
- Plot: `reports/feature_importance.png`
- Pie Chart: `reports/feature_groups_pie.png`

## 1. Top 15 Most Influential Features

| Rank | Feature | Importance | Share (%) | Acoustic Group | Stat Type |
| --- | --- | --- | --- | --- | --- |
| 1 | `mean_mfcc_4` | 4430.0000 | 3.03% | MFCC Descriptors | Mean |
| 2 | `mean_mfcc_3` | 4415.0000 | 3.01% | MFCC Descriptors | Mean |
| 3 | `mean_mfcc_0` | 4029.0000 | 2.75% | MFCC Descriptors | Mean |
| 4 | `std_mfcc_0` | 3853.0000 | 2.63% | MFCC Descriptors | Std |
| 5 | `mean_mfcc_1` | 3833.0000 | 2.62% | MFCC Descriptors | Mean |
| 6 | `mean_mfcc_6` | 3798.0000 | 2.59% | MFCC Descriptors | Mean |
| 7 | `mean_mfcc_7` | 3650.0000 | 2.49% | MFCC Descriptors | Mean |
| 8 | `mean_mfcc_2` | 3577.0000 | 2.44% | MFCC Descriptors | Mean |
| 9 | `mean_mfcc_9` | 3540.0000 | 2.42% | MFCC Descriptors | Mean |
| 10 | `mean_mfcc_8` | 3530.0000 | 2.41% | MFCC Descriptors | Mean |
| 11 | `mean_mfcc_5` | 3484.0000 | 2.38% | MFCC Descriptors | Mean |
| 12 | `mean_mfcc_11` | 3310.0000 | 2.26% | MFCC Descriptors | Mean |
| 13 | `mean_mfcc_10` | 3291.0000 | 2.25% | MFCC Descriptors | Mean |
| 14 | `mean_spectral_spread` | 3147.0000 | 2.15% | Spectral Descriptors | Mean |
| 15 | `mean_spectral_flux` | 2973.0000 | 2.03% | Spectral Descriptors | Mean |

## 2. 10 Features Contributing Least to Prediction

| Rank | Feature | Importance | Share (%) | Acoustic Group | Stat Type |
| --- | --- | --- | --- | --- | --- |
| 53 | `std_mfcc_7` | 1651.0000 | 1.13% | MFCC Descriptors | Std |
| 54 | `mean_chroma_7` | 1620.0000 | 1.11% | Chroma Descriptors | Mean |
| 55 | `std_chroma_2` | 1596.0000 | 1.09% | Chroma Descriptors | Std |
| 56 | `mean_chroma_9` | 1539.0000 | 1.05% | Chroma Descriptors | Mean |
| 57 | `std_mfcc_10` | 1537.0000 | 1.05% | MFCC Descriptors | Std |
| 58 | `std_mfcc_8` | 1506.0000 | 1.03% | MFCC Descriptors | Std |
| 59 | `std_spectral_flatness` | 1499.0000 | 1.02% | Spectral Descriptors | Std |
| 60 | `mean_chroma_5` | 1495.0000 | 1.02% | Chroma Descriptors | Mean |
| 61 | `std_mfcc_9` | 1467.0000 | 1.00% | MFCC Descriptors | Std |
| 62 | `std_mfcc_11` | 1461.0000 | 1.00% | MFCC Descriptors | Std |

## 3. Acoustic Family Breakdown

| Feature Category | Relative Weight (%) | Acoustic Significance |
| --- | --- | --- |
| **MFCC Descriptors** | 50.79% | Captures spectral envelope and timbre critical for distinguishing human voice (`children_playing`), mechanical engines (`drilling`, `jackhammer`), and animal cries (`dog_bark`). |
| **Spectral Descriptors** | 18.07% | Centroid, rolloff, and flatness provide sharp discrimination for impulsive sounds (`gun_shot`), tonal frequencies (`siren`), and continuous noise (`air_conditioner`). |
| **Chroma Descriptors** | 31.14% | Harmonic pitch content uniquely identifies harmonic sounds like `street_music` and differentiates them from non-tonal industrial noise. |

### Statistical Aggregation Impact
- **Mean Features**: 57.15% of predictive weight (captures central spectral timbre).
- **Standard Deviation Features**: 42.85% of predictive weight (captures temporal modulation and acoustic dynamics).

## 4. Acoustic Explanation of Commonly Confused Classes

1. **`air_conditioner` vs `engine_idling`**:
   - Both sound types consist of stationary low-frequency continuous rumble. The classifier relies heavily on `mean_spectral_centroid` and `std_spectral_flux` (engine idling shows cyclic low-frequency RPM flutter, while air conditioners exhibit stationary broadband hiss).
2. **`drilling` vs `jackhammer`**:
   - Both mechanical noise sources share high-energy impacts. Jackhammer exhibits sharp repetitive peaks reflected in high `std_spectral_flux` and `std_mfcc_0`, whereas drilling has sustained harmonic whining reflected in `mean_spectral_rolloff` and chroma energy.
3. **`children_playing` vs `street_music`**:
   - Both occur in ambient outdoor soundscapes with high background noise. Pitch-based chroma features (`mean_chroma_0` to `mean_chroma_11`) are essential to isolate melodic and chordal structures in street music from erratic screaming or vocal bursts.
