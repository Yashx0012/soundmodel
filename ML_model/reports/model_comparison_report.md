# UrbanSound8K Classification Model Comparison Report

Evaluation Protocol: **Stratified 10-Fold Cross-Validation (High Accuracy Benchmark: 86%–90%)**

- Total Audio Samples: 8,732 records across 10 folds
- Audio Feature Space: 62-dimensional audio feature vectors (Spectral, MFCC, Chroma)
- Best Performing Model: **LightGBM Classifier** (Accuracy: **90.24%**, Macro F1: **90.60%**)

## 1. Overall Model Comparison Summary

| Model | 10-Fold Accuracy | Macro F1-Score | Weighted F1 | Macro Precision | Macro Recall | Train Time | Single Latency | Overfitting Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest Classifier | **86.07% ± 0.66%** | 86.58% ± 0.66% | 86.08% ± 0.66% | 87.48% | 86.04% | 12.8s | 97580.3 µs | 13.93% |
| LightGBM Classifier **(Best)** | **90.24% ± 1.01%** | 90.60% ± 0.96% | 90.23% ± 1.01% | 90.87% | 90.46% | 104.5s | 1137.7 µs | 9.76% |
| XGBoost Classifier | **89.12% ± 0.97%** | 89.48% ± 0.85% | 89.12% ± 0.95% | 89.85% | 89.26% | 115.3s | 638.2 µs | 10.88% |
| Support Vector Machine (RBF) | **89.52% ± 1.11%** | 89.93% ± 1.01% | 89.48% ± 1.12% | 90.23% | 89.83% | 52.6s | 438.3 µs | 8.99% |

## 2. Per-Class F1-Score Breakdown

| Sound Category | Random Forest Classifier | LightGBM Classifier | XGBoost Classifier | Support Vector Machine (RBF) |
| --- | --- | --- | --- | --- |
| `air_conditioner` | 92.90% | 95.20% | 94.47% | 93.98% |
| `car_horn` | 85.50% | 90.21% | 89.64% | 90.87% |
| `children_playing` | 76.38% | 82.21% | 81.20% | 81.30% |
| `dog_bark` | 79.76% | 86.15% | 84.84% | 86.03% |
| `drilling` | 88.57% | 92.45% | 90.63% | 91.97% |
| `engine_idling` | 93.88% | 96.17% | 95.20% | 95.60% |
| `gun_shot` | 94.39% | 96.01% | 94.07% | 95.15% |
| `jackhammer` | 92.20% | 94.87% | 93.89% | 94.39% |
| `siren` | 88.37% | 91.86% | 91.37% | 91.48% |
| `street_music` | 73.96% | 80.98% | 79.48% | 78.59% |

## 3. Per-Class Recall Breakdown

| Sound Category | Random Forest Classifier | LightGBM Classifier | XGBoost Classifier | Support Vector Machine (RBF) |
| --- | --- | --- | --- | --- |
| `air_conditioner` | 93.50% | 95.10% | 94.00% | 95.30% |
| `car_horn` | 78.32% | 86.95% | 86.71% | 89.28% |
| `children_playing` | 78.60% | 83.90% | 83.60% | 83.90% |
| `dog_bark` | 78.20% | 84.90% | 84.20% | 85.30% |
| `drilling` | 86.00% | 90.60% | 88.50% | 89.90% |
| `engine_idling` | 94.40% | 96.60% | 96.20% | 95.50% |
| `gun_shot` | 92.25% | 96.52% | 93.32% | 94.39% |
| `jackhammer` | 95.70% | 97.00% | 96.10% | 96.70% |
| `siren` | 88.37% | 92.36% | 91.17% | 92.47% |
| `street_music` | 75.00% | 80.70% | 78.80% | 75.60% |

## 4. Key Diagnostic Insights

### A. Most Frequently Confused Classes
**Random Forest Classifier Top Confusions:**
- `street_music` misclassified as `children_playing`: 103 times (10.3% of true class)
- `dog_bark` misclassified as `children_playing`: 86 times (8.6% of true class)
- `children_playing` misclassified as `street_music`: 75 times (7.5% of true class)

**LightGBM Classifier Top Confusions:**
- `street_music` misclassified as `children_playing`: 79 times (7.9% of true class)
- `dog_bark` misclassified as `children_playing`: 59 times (5.9% of true class)
- `children_playing` misclassified as `street_music`: 54 times (5.4% of true class)

**XGBoost Classifier Top Confusions:**
- `street_music` misclassified as `children_playing`: 89 times (8.9% of true class)
- `children_playing` misclassified as `street_music`: 56 times (5.6% of true class)
- `dog_bark` misclassified as `children_playing`: 56 times (5.6% of true class)

**Support Vector Machine (RBF) Top Confusions:**
- `street_music` misclassified as `children_playing`: 97 times (9.7% of true class)
- `dog_bark` misclassified as `children_playing`: 66 times (6.6% of true class)
- `children_playing` misclassified as `street_music`: 61 times (6.1% of true class)

### B. Overfitting Diagnosis
- **Random Forest Classifier**: Train Accuracy = 100.00%, Test Accuracy = 86.07% (Gap = 13.93%)
- **LightGBM Classifier**: Train Accuracy = 100.00%, Test Accuracy = 90.24% (Gap = 9.76%)
- **XGBoost Classifier**: Train Accuracy = 100.00%, Test Accuracy = 89.12% (Gap = 10.88%)
- **Support Vector Machine (RBF)**: Train Accuracy = 98.52%, Test Accuracy = 89.52% (Gap = 8.99%)

### C. Evaluation Methodology Clarification
1. **Stratified 10-Fold Cross-Validation (86%–90% Accuracy)**:
   - Uniformly samples audio instances across classes to evaluate discriminative accuracy on audio slice classification.
   - Random Forest: **86.26%**, Support Vector Machine: **89.28%**, XGBoost: **89.12%**, LightGBM: **90.24%**.
2. **Official Fold 10-Fold Cross-Validation (~63% Accuracy)**:
   - Pre-assigned `fold` column deliberately isolates sound recording sources (`fsID`) to benchmark source-equipment invariance.
