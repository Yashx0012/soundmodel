"""
Unit tests for DatasetService.
"""

import os
import tempfile
import pytest

from app.config import DEFAULT_DATASET_ROOT
from app.services.dataset_service import DatasetService


def test_load_and_validate_real_dataset_fold3():
    """
    Test loading fold 3 from actual UrbanSound8K_Processed dataset.
    """
    valid_files, errors, summary = DatasetService.load_and_validate(
        dataset_root=DEFAULT_DATASET_ROOT,
        fold_filter=3,
    )

    assert len(errors) == 0, f"Expected 0 errors, got: {errors}"
    assert len(valid_files) == 925
    assert summary["valid_files_count"] == 925
    assert summary["fold_filter"] == 3

    # Check first item structure and path
    first = valid_files[0]
    assert first.fold == 3
    assert first.filename.endswith(".wav")
    assert os.path.isabs(first.wav_path)
    assert os.path.isfile(first.wav_path)
    assert f"fold3/{first.filename}" in first.wav_path
    assert first.classID >= 0
    assert bool(first.class_name)


def test_load_and_validate_all_folds():
    """
    Test loading all folds from actual UrbanSound8K_Processed dataset.
    """
    valid_files, errors, summary = DatasetService.load_and_validate(
        dataset_root=DEFAULT_DATASET_ROOT,
        fold_filter="all",
    )

    assert len(errors) == 0
    assert len(valid_files) == 8732
    assert summary["total_csv_rows"] == 8732
    assert len(summary["fold_distribution"]) == 10
    assert summary["fold_distribution"][1] == 873
    assert summary["fold_distribution"][10] == 837


def test_invalid_fold_filter():
    """
    Test that invalid fold numbers raise ValueError.
    """
    with pytest.raises(ValueError, match="Fold number must be between 1 and 10"):
        DatasetService.load_and_validate(
            dataset_root=DEFAULT_DATASET_ROOT,
            fold_filter=11,
        )

    with pytest.raises(ValueError, match="Invalid fold filter"):
        DatasetService.load_and_validate(
            dataset_root=DEFAULT_DATASET_ROOT,
            fold_filter="invalid_fold",
        )


def test_handling_missing_wav_files():
    """
    Test that missing files are recorded in validation_errors without crashing.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock CSV with one existing file and one missing file
        csv_path = os.path.join(tmp_dir, "UrbanSound8K.csv")
        fold1_dir = os.path.join(tmp_dir, "fold1")
        os.makedirs(fold1_dir, exist_ok=True)

        existing_wav = os.path.join(fold1_dir, "exists.wav")
        with open(existing_wav, "wb") as f:
            f.write(b"RIFFmockwavdata")

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("slice_file_name,fsID,start,end,salience,fold,classID,class\n")
            f.write("exists.wav,1001,0.0,1.0,1,1,3,dog_bark\n")
            f.write("missing.wav,1002,0.0,1.0,1,1,0,air_conditioner\n")

        valid_files, errors, summary = DatasetService.load_and_validate(
            dataset_root=tmp_dir,
            csv_filename="UrbanSound8K.csv",
        )

        assert len(valid_files) == 1
        assert valid_files[0].filename == "exists.wav"
        assert len(errors) == 1
        assert errors[0].error_type == "missing_wav_file"
        assert errors[0].filename == "missing.wav"
        assert "missing.wav" in errors[0].message


def test_handling_corrupted_csv_rows():
    """
    Test handling of non-integer fold and missing columns.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "UrbanSound8K.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("slice_file_name,fsID,start,end,salience,fold,classID,class\n")
            f.write("bad_fold.wav,1001,0.0,1.0,1,NOT_A_NUMBER,3,dog_bark\n")
            f.write("bad_cid.wav,1002,0.0,1.0,1,1,NOT_A_CID,air_conditioner\n")

        valid_files, errors, summary = DatasetService.load_and_validate(
            dataset_root=tmp_dir,
            csv_filename="UrbanSound8K.csv",
        )

        assert len(valid_files) == 0
        assert len(errors) == 2
        assert errors[0].error_type == "invalid_fold"
        assert errors[1].error_type == "invalid_class_id"
