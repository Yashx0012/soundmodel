# UrbanSound8K Feature Dataset Validation Summary

- **Dataset Path**: `/home/ORBITAL/soundmodel/UrbanSound8K_Processed/UrbanSound8K.csv`
- **Overall Status**: **VALIDATED**
- **Total Records**: 8,732
- **Total Columns**: 70
- **Feature Columns**: 62 (Expected: 62)
- **Metadata Columns**: 8 (Excluded from inputs)

## Validation Checks

| Check | Status | Details |
| --- | --- | --- |
| Dimensions | PASS | 8732 rows × 70 cols |
| 62 Features Identified | PASS | Exactly 62 features found |
| Missing Values | PASS | 0 nulls in dataset |
| Duplicate Records | PASS | 0 row duplicates, 0 filename duplicates |
| Numeric Sanity | PASS | All 62 features are numeric; 0 infinite values |
| Label Consistency | PASS | 10 classes mapped 1:1 with classID 0-9 |
| Fold Integrity | PASS | All 10 folds present (800-1000 samples each) |

## Class Distribution

| Class ID | Sound Category | Sample Count | Percentage |
| --- | --- | --- | --- |
| 0 | `air_conditioner` | 1,000 | 11.45% |
| 1 | `car_horn` | 429 | 4.91% |
| 2 | `children_playing` | 1,000 | 11.45% |
| 3 | `dog_bark` | 1,000 | 11.45% |
| 4 | `drilling` | 1,000 | 11.45% |
| 5 | `engine_idling` | 1,000 | 11.45% |
| 6 | `gun_shot` | 374 | 4.28% |
| 7 | `jackhammer` | 1,000 | 11.45% |
| 8 | `siren` | 929 | 10.64% |
| 9 | `street_music` | 1,000 | 11.45% |

## Fold Distribution

| Fold | Sample Count | Percentage |
| --- | --- | --- |
| Fold 1 | 873 | 10.0% |
| Fold 2 | 888 | 10.17% |
| Fold 3 | 925 | 10.59% |
| Fold 4 | 990 | 11.34% |
| Fold 5 | 936 | 10.72% |
| Fold 6 | 823 | 9.43% |
| Fold 7 | 838 | 9.6% |
| Fold 8 | 806 | 9.23% |
| Fold 9 | 816 | 9.34% |
| Fold 10 | 837 | 9.59% |
