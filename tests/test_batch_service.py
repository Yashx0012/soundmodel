"""
Unit tests for BatchService with C DSP tool compatibility.
"""

import os
import pytest
from app.models.schemas import AudioFileItem, BatchPayload
from app.services.batch_service import BatchService


@pytest.fixture
def sample_audio_files():
    """Generates 125 sample AudioFileItem objects."""
    files = []
    for i in range(125):
        files.append(
            AudioFileItem(
                filename=f"audio_{i}.wav",
                fold=3,
                wav_path=f"/fake/UrbanSound8K/fold3/audio_{i}.wav",
                classID=i % 10,
                class_name="sample_class",
                file_exists=True,
            )
        )
    return files


def test_divide_into_batches_exact_and_remainder(sample_audio_files):
    """
    Test dividing 125 files with batch size 50:
    Should produce 3 batches:
    Batch 1: 50 files
    Batch 2: 50 files
    Batch 3: 25 files
    """
    batches = BatchService.divide_into_batches(
        files=sample_audio_files,
        batch_size=50,
        csv_path="/fake/UrbanSound8K/UrbanSound8K.csv",
        fold_identifier=3,
    )

    assert len(batches) == 3
    assert batches[0].batch_number == 1
    assert batches[0].total_batches == 3
    assert batches[0].batch_size == 50
    assert batches[0].files_count == 50
    assert len(batches[0].files) == 50
    assert batches[0].files[0] == "/fake/UrbanSound8K/fold3/audio_0.wav"
    assert batches[0].files[-1] == "/fake/UrbanSound8K/fold3/audio_49.wav"
    assert batches[0].files_metadata[0].filename == "audio_0.wav"

    assert batches[1].batch_number == 2
    assert batches[1].files_count == 50
    assert batches[1].files[0] == "/fake/UrbanSound8K/fold3/audio_50.wav"

    assert batches[2].batch_number == 3
    assert batches[2].files_count == 25
    assert batches[2].files[0] == "/fake/UrbanSound8K/fold3/audio_100.wav"
    assert batches[2].files[-1] == "/fake/UrbanSound8K/fold3/audio_124.wav"


def test_c_tool_payload_format(sample_audio_files):
    """
    Test that to_c_tool_payload returns the exact structure expected by urban8k_tool /process.
    """
    batches = BatchService.divide_into_batches(
        files=sample_audio_files[:3],
        batch_size=50,
        csv_path="/path/to/UrbanSound8K.csv",
        fold_identifier=3,
    )
    c_payload = batches[0].to_c_tool_payload()
    assert "csv_path" in c_payload
    assert c_payload["csv_path"] == "/path/to/UrbanSound8K.csv"
    assert "files" in c_payload
    assert len(c_payload["files"]) == 3
    assert c_payload["files"][0] == "/fake/UrbanSound8K/fold3/audio_0.wav"


def test_submit_batches_simulated(sample_audio_files):
    """
    Test simulated batch submission when target_url is None.
    """
    batches = BatchService.divide_into_batches(
        files=sample_audio_files[:10],
        batch_size=5,
        csv_path="/path/to/UrbanSound8K.csv",
        fold_identifier=3,
    )
    results = BatchService.submit_batches_sequentially(
        batches=batches,
        target_url=None,
    )

    assert len(results) == 2
    for r in results:
        assert r.status == "simulated"
        assert r.target_url is None
        assert r.error_message is None
        assert r.processed_count == 5
        assert r.failed_count == 0
