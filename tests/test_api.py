"""
Integration tests for FastAPI endpoints with C DSP tool compatibility.
"""

from fastapi.testclient import TestClient
import pytest

from app.config import DEFAULT_DATASET_ROOT
from app.main import app

client = TestClient(app)


def test_root_and_health_endpoints():
    """Test health check and root endpoints."""
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert resp_root.json()["status"] == "online"

    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"
    assert resp_health.json()["dataset_available"] is True


def test_dataset_info_endpoint():
    """Test GET /api/v1/dataset/info."""
    resp = client.get("/api/v1/dataset/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_records"] == 8732
    assert data["verified_files_count"] == 8732
    assert data["missing_files_count"] == 0
    assert len(data["fold_distribution"]) == 10
    assert len(data["classes"]) == 10


def test_dataset_sample_endpoint():
    """Test GET /api/v1/dataset/sample."""
    resp = client.get("/api/v1/dataset/sample?fold=3&limit=3")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 3
    for it in items:
        assert it["fold"] == 3
        assert it["filename"].endswith(".wav")
        assert "fold3" in it["wav_path"]


def test_prepare_batches_single_fold():
    """
    Test POST /api/v1/batches/prepare for fold 3 with batch_size 50.
    Fold 3 has 925 files -> 925 / 50 = 19 batches.
    """
    payload = {
        "dataset_root": DEFAULT_DATASET_ROOT,
        "fold": 3,
        "batch_size": 50,
    }
    resp = client.post("/api/v1/batches/prepare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["fold_filter"] == "3" or data["fold_filter"] == 3
    assert data["batch_size"] == 50
    assert data["total_valid_files"] == 925
    assert data["total_batches"] == 19
    assert len(data["batches"]) == 19

    first_batch = data["batches"][0]
    assert first_batch["batch_number"] == 1
    assert first_batch["total_batches"] == 19
    assert first_batch["batch_size"] == 50
    assert first_batch["files_count"] == 50
    assert len(first_batch["files"]) == 50
    # Files array contains WAV path strings for C tool
    assert first_batch["files"][0].endswith(".wav")
    assert "fold3" in first_batch["files"][0]
    # Detailed metadata array contains rich items
    assert len(first_batch["files_metadata"]) == 50
    assert "class" in first_batch["files_metadata"][0]

    last_batch = data["batches"][-1]
    assert last_batch["batch_number"] == 19
    assert last_batch["files_count"] == 25


def test_mock_receiver_c_tool_format():
    """
    Test POST /api/v1/batches/mock-receiver with the exact urban8k_tool JSON payload:
    {"csv_path": "...", "files": ["/path/1.wav", "/path/2.wav"]}
    """
    c_payload = {
        "csv_path": "/fake/UrbanSound8K.csv",
        "files": [
            "/fake/fold1/100032-3-0-0.wav",
            "/fake/fold1/100263-2-0-117.wav",
        ],
    }
    resp = client.post("/api/v1/batches/mock-receiver", json=c_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["processed"] == 2
    assert body["failed"] == 0
    assert body["num_features"] == 62
    assert body["csv_path"] == "/fake/UrbanSound8K.csv"


def test_c_tool_status_probe():
    """Test GET /api/v1/batches/c-tool/status."""
    resp = client.get("/api/v1/batches/c-tool/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "c_tool_url" in data
    assert "c_tool_health_url" in data
    assert "c_tool_available" in data


def test_submit_batches_simulated():
    """
    Test POST /api/v1/batches/submit without target_url (simulated mode).
    """
    payload = {
        "dataset_root": DEFAULT_DATASET_ROOT,
        "fold": 1,
        "batch_size": 20,
        "max_batches": 1,
        "target_url": None,
    }
    resp = client.post("/api/v1/batches/submit", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_batches"] == 1
    assert data["successful_batches"] == 1
    assert data["failed_batches"] == 0
    assert len(data["results"]) == 1
    assert data["results"][0]["status"] == "simulated"
    assert data["results"][0]["files_count"] == 20
    # Check stats section
    assert "stats" in data
    stats = data["stats"]
    assert stats["total_files_requested"] == 873
    assert stats["total_files_processed"] == 20
    assert stats["total_files_failed"] == 0
    assert stats["total_batches"] == 1
    assert stats["total_duration_ms"] > 0
    assert stats["total_duration_seconds"] >= 0
    assert "c_dsp_engine_time_ms" in stats
    assert "c_dsp_engine_time_seconds" in stats
    assert "fastapi_overhead_ms" in stats


def test_verify_dataset_features_endpoint():
    """
    Test GET /api/v1/dataset/verify-features on dataset.
    """
    resp = client.get("/api/v1/dataset/verify-features?fold=1")
    assert resp.status_code == 200
    report = resp.json()
    assert "is_valid" in report
    assert "feature_columns_found" in report
    assert "total_rows_inspected" in report
    assert report["total_rows_inspected"] == 873


def test_invalid_fold_and_missing_directory_errors():
    """
    Test error responses for bad fold and non-existent dataset directory.
    """
    # Invalid fold
    resp = client.post(
        "/api/v1/batches/prepare",
        json={"dataset_root": DEFAULT_DATASET_ROOT, "fold": 99},
    )
    assert resp.status_code == 400
    assert "between 1 and 10" in resp.json()["detail"]

    # Non-existent dataset directory
    resp_dir = client.post(
        "/api/v1/batches/prepare",
        json={"dataset_root": "/non/existent/directory", "fold": 1},
    )
    assert resp_dir.status_code == 404
    assert "does not exist" in resp_dir.json()["detail"]
