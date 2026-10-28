"""
Unit tests for VerificationService.
"""

import csv
import os
import tempfile
import pytest

from app.services.verification_service import (
    CANONICAL_62_FEATURE_COLUMNS,
    VerificationService,
)


def test_verify_csv_missing_feature_columns():
    """
    Test verifying a raw CSV that does NOT have the 62 feature columns yet.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "UrbanSound8K.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class"])
            writer.writerow(["1001-1.wav", "1001", "0.0", "1.0", "1", "1", "0", "air_conditioner"])

        report = VerificationService.verify_csv_feature_mapping(csv_path=csv_path)
        assert report.is_valid is False
        assert report.feature_columns_found == 0
        assert report.unmapped_or_incomplete_rows == 1
        assert report.discrepancy_count == 1
        assert report.discrepancies[0].issue_type == "missing_columns"


def test_verify_csv_valid_62_features():
    """
    Test verifying a CSV that has all 62 canonical columns populated with valid floats.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "UrbanSound8K.csv")
        base_cols = ["slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class"]
        all_cols = base_cols + CANONICAL_62_FEATURE_COLUMNS

        row_data = ["1001-1.wav", "1001", "0.0", "1.0", "1", "1", "0", "air_conditioner"]
        # Populate all 62 features with valid floats
        row_data += ["0.12345" for _ in range(62)]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(all_cols)
            writer.writerow(row_data)

        report = VerificationService.verify_csv_feature_mapping(csv_path=csv_path)
        assert report.is_valid is True
        assert report.feature_columns_found == 62
        assert report.fully_mapped_rows == 1
        assert report.unmapped_or_incomplete_rows == 0
        assert report.discrepancy_count == 0
        assert report.sample_mapped_record is not None
        assert report.sample_mapped_record["slice_file_name"] == "1001-1.wav"


def test_verify_csv_with_nan_or_empty_values():
    """
    Test verifying a CSV where a row has empty or NaN feature values.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "UrbanSound8K.csv")
        base_cols = ["slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class"]
        all_cols = base_cols + CANONICAL_62_FEATURE_COLUMNS

        # Row 1 has valid floats
        row1 = ["1001-1.wav", "1001", "0.0", "1.0", "1", "1", "0", "air_conditioner"] + ["1.5" for _ in range(62)]
        # Row 2 has an empty feature cell
        row2 = ["1002-2.wav", "1002", "0.0", "1.0", "1", "1", "1", "car_horn"] + ["1.5" for _ in range(61)] + [""]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(all_cols)
            writer.writerow(row1)
            writer.writerow(row2)

        report = VerificationService.verify_csv_feature_mapping(csv_path=csv_path)
        assert report.is_valid is False
        assert report.fully_mapped_rows == 1
        assert report.unmapped_or_incomplete_rows == 1
        assert report.discrepancy_count == 1
        assert report.discrepancies[0].slice_file_name == "1002-2.wav"
