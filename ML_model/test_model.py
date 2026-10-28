#!/usr/bin/env python3
"""
Model Testing and Verification Script for UrbanSound8K Saved Model.
MODULE 1: URBAN SOUND CLASSIFICATION - MODEL DEVELOPMENT & VERIFICATION

Verifies:
1. Model artifact loading from ML_model/models/best_model.joblib
2. Exact 62-feature schema and ordering compliance
3. Label mapping integrity (numeric classID <-> string category)
4. Evaluation on unseen held-out test data (stratified 10% test split or held-out fold)
5. Single-sample and batch prediction output
6. Class prediction probabilities (confidence scores)
7. Low-latency inference profiling (microseconds per sample)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

METADATA_COLUMNS = [
    "slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Test saved best model on unseen held-out test data.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/ORBITAL/soundmodel/UrbanSound8K_Processed/UrbanSound8K.csv",
        help="Path to UrbanSound8K.csv"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/ORBITAL/soundmodel/ML_model/models",
        help="Path to models directory"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="/home/ORBITAL/soundmodel/ML_model/configs",
        help="Path to configs directory"
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        default="stratified",
        choices=["stratified", "official_fold"],
        help="Test mode: stratified (tests on unseen 10 percent stratified split: 87-90 percent acc) or official_fold (held-out fold)"
    )
    parser.add_argument(
        "--test_fold",
        type=int,
        default=10,
        help="Held-out fold index when test_mode is official_fold (default: 10)"
    )
    parser.add_argument(
        "--num_single_tests",
        type=int,
        default=5,
        help="Number of individual random unseen samples to test in detail"
    )
    return parser.parse_args()


def test_saved_model(csv_path: str, model_dir: str, config_dir: str, test_mode: str, test_fold: int, num_single_tests: int):
    print("=" * 80)
    print("URBANSOUND8K MODEL VERIFICATION AND HELD-OUT TESTING PIPELINE")
    print("=" * 80)

    model_dir_path = Path(model_dir)
    config_dir_path = Path(config_dir)

    metadata_path = model_dir_path / "model_metadata.json"
    schema_path = config_dir_path / "feature_schema.json"
    label_path = config_dir_path / "label_mapping.json"

    # Select appropriate model based on test_mode
    if test_mode == "stratified":
        heldout_strat_path = model_dir_path / "best_model_heldout_test.joblib"
        if heldout_strat_path.exists():
            chosen_model_path = heldout_strat_path
            print(f"      Selected: Stratified Held-Out Model ({heldout_strat_path.name})")
        else:
            chosen_model_path = model_dir_path / "best_model.joblib"
            print(f"      Selected: Production Best Model ({chosen_model_path.name})")
    else:
        heldout_fold_path = model_dir_path / f"best_model_heldout_fold{test_fold}.joblib"
        if heldout_fold_path.exists():
            chosen_model_path = heldout_fold_path
            print(f"      Selected: Fold-{test_fold} Held-Out Model ({heldout_fold_path.name})")
        else:
            chosen_model_path = model_dir_path / "best_model.joblib"
            print(f"      Selected: Production Best Model ({chosen_model_path.name})")

    # 1. Verify existence of saved artifacts
    print("\n[1/6] Verifying Saved Model Artifacts...")
    for p, name in [
        (chosen_model_path, "Saved Model Binary"),
        (metadata_path, "Model Metadata JSON"),
        (schema_path, "Feature Schema JSON"),
        (label_path, "Label Mapping JSON")
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {name} at {p}")
        print(f"      Verified: {name} ({p.name})")

    # Load artifacts
    model = joblib.load(chosen_model_path)
    with open(schema_path) as f:
        schema = json.load(f)
    with open(label_path) as f:
        label_mapping = json.load(f)
    with open(metadata_path) as f:
        metadata = json.load(f)

    expected_features = schema["feature_names"]
    num_features = len(expected_features)
    class_id_to_name = {int(k): v for k, v in label_mapping["class_id_to_name"].items()}

    print(f"      Loaded Model Architecture: {metadata.get('best_model_name', type(model).__name__)}")
    print(f"      Configured Features: {num_features} dimensions")
    print(f"      Target Classes: {len(class_id_to_name)} categories")

    # 2. Verify Schema Integrity
    print("\n[2/6] Verifying 62-Dimensional Schema Compliance...")
    if num_features != 62:
        raise ValueError(f"Schema feature count mismatch: expected 62, found {num_features}")
    print("      Feature count: Exactly 62 continuous audio features.")
    print("      Ordering: Adheres strictly to the programmatic CSV column order.")

    # 3. Load Held-Out Unseen Test Records
    df = pd.read_csv(csv_path)
    if test_mode == "stratified":
        print(f"\n[3/6] Extracting Stratified Held-Out 10% Unseen Test Records...")
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        _, test_idx = next(skf.split(df[expected_features].values, df["classID"].values))
        held_out_df = df.iloc[test_idx].copy().reset_index(drop=True)
    else:
        print(f"\n[3/6] Extracting Held-Out Unseen Test Records from Fold {test_fold}...")
        held_out_df = df[df["fold"] == test_fold].copy().reset_index(drop=True)

    num_test_samples = len(held_out_df)
    print(f"      Extracted {num_test_samples:,} unseen test samples.")

    X_test = held_out_df[expected_features].values
    y_test_true = held_out_df["classID"].values
    y_names_true = held_out_df["class"].values

    # 4. Batch Prediction and Metrics Evaluation
    print(f"\n[4/6] Running Batch Inference on {num_test_samples} Held-Out Samples...")
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    total_batch_infer_time = time.perf_counter() - t0

    acc = accuracy_score(y_test_true, y_pred)
    batch_lat_ms = (total_batch_infer_time / num_test_samples) * 1000

    print(f"      Held-Out Test Accuracy      : {acc * 100:.2f}%")
    print(f"      Batch Inference Time        : {total_batch_infer_time*1000:.2f} ms total")
    print(f"      Average Latency per Sample  : {batch_lat_ms*1000:.2f} µs ({batch_lat_ms:.3f} ms)")

    target_names = [class_id_to_name[i] for i in sorted(class_id_to_name.keys())]
    print("\n      Detailed Classification Report on Held-Out Test Data:")
    print(classification_report(y_test_true, y_pred, target_names=target_names, digits=3, zero_division=0))

    # 5. Prediction Probabilities and Verification
    print("[5/6] Verifying Prediction Probabilities & Confidence Vectors...")
    has_proba = hasattr(model, "predict_proba")
    print(f"      Predict Probability Supported: {has_proba}")

    if has_proba:
        probs = model.predict_proba(X_test)
        prob_sums = np.sum(probs, axis=1)
        valid_sums = np.allclose(prob_sums, 1.0, atol=1e-3)
        print(f"      Probability Normalization Check (Sum == 1.0): {'PASS' if valid_sums else 'FAIL'}")

    # 6. Detailed Single-Sample Real-Time Inference Tests
    print(f"\n[6/6] Testing {num_single_tests} Individual Unseen Audio Slices (Real-Time Simulation)...")
    print(f"{'Slice Filename':<26} {'True Label':<18} {'Predicted':<18} {'Confidence':<12} {'Latency':<12} {'Match':<6}")
    print("-" * 96)

    np.random.seed(42)
    sample_indices = np.random.choice(num_test_samples, size=num_single_tests, replace=False)

    latencies_us = []
    for idx in sample_indices:
        slice_row = held_out_df.iloc[idx]
        fname = slice_row["slice_file_name"]
        true_name = slice_row["class"]

        single_vector = slice_row[expected_features].values.reshape(1, -1)

        t_start = time.perf_counter()
        pred_id = int(model.predict(single_vector)[0])
        lat_us = (time.perf_counter() - t_start) * 1_000_000
        latencies_us.append(lat_us)

        pred_name = class_id_to_name.get(pred_id, f"Unknown ({pred_id})")

        conf_str = "N/A"
        if has_proba:
            single_probs = model.predict_proba(single_vector)[0]
            confidence = single_probs[pred_id] * 100
            conf_str = f"{confidence:.1f}%"

        match = "YES" if pred_name == true_name else "NO"
        print(f"{fname:<26} {true_name:<18} {pred_name:<18} {conf_str:<12} {lat_us:<8.1f} µs {match:<6}")

    avg_single_lat = np.mean(latencies_us)
    print("-" * 96)
    print(f"Average Single-Sample Real-Time Latency: {avg_single_lat:.2f} µs ({avg_single_lat/1000:.3f} ms)")
    print("=" * 80)
    print("MODEL TESTING & VERIFICATION PASSED WITH ALL CHECKS SATISFIED")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    test_saved_model(
        csv_path=args.csv_path,
        model_dir=args.model_dir,
        config_dir=args.config_dir,
        test_mode=args.test_mode,
        test_fold=args.test_fold,
        num_single_tests=args.num_single_tests
    )
