"""
Live end-to-end verification script for the CSV-Based FastAPI server
integrated with the UrbanSound8K C DSP Feature Extraction Tool (`urban8k_tool`).

Tests:
1. Health and Dataset Info.
2. Batch preparation (formatted for urban8k_tool).
3. Small batch sequential submission to mock receiver (Fold 1, batch size 10).
4. Direct integration test with live urban8k_tool C binary on port 8085.
5. All folds batch preparation (8,732 files -> 175 batches of 50).
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
import requests

FASTAPI_BASE_URL = "http://127.0.0.1:8000"
MOCK_RECEIVER_URL = f"{FASTAPI_BASE_URL}/api/v1/batches/mock-receiver"
C_TOOL_BIN = "/home/ORBITAL/Documents/dsp_engine-1/data_processing_tools/urban8k_tool"
TEST_C_TOOL_PORT = 8085
TEST_C_TOOL_URL = f"http://127.0.0.1:{TEST_C_TOOL_PORT}/process"


def verify():
    print("=" * 65)
    print("1. Checking FastAPI Server Health and Dataset Info...")
    print("=" * 65)

    resp = requests.get(f"{FASTAPI_BASE_URL}/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    print("Health response:", resp.json())

    resp = requests.get(f"{FASTAPI_BASE_URL}/api/v1/dataset/info")
    assert resp.status_code == 200, f"Dataset info failed: {resp.text}"
    info = resp.json()
    print(f"Dataset root: {info['dataset_root']}")
    print(f"Total CSV records: {info['total_records']}")
    print(f"Verified files: {info['verified_files_count']}")
    print(f"Missing files: {info['missing_files_count']}")

    print("\n" + "=" * 65)
    print("2. Batch Preparation Test (C Tool Payload Format)")
    print("=" * 65)

    prep_req = {
        "fold": 1,
        "batch_size": 25,
        "max_batches": 2,
    }
    resp = requests.post(f"{FASTAPI_BASE_URL}/api/v1/batches/prepare", json=prep_req)
    assert resp.status_code == 200, f"Prepare failed: {resp.text}"
    prep_res = resp.json()
    b1 = prep_res["batches"][0]
    print(f"Batch 1 files count: {b1['files_count']}")
    print(f"Batch 1 C tool payload sample (first 2 files): {b1['files'][:2]}")
    print(f"Batch 1 metadata item sample: {b1['files_metadata'][0]['filename']}, class: {b1['files_metadata'][0]['class']}")
    assert len(b1["files"]) == 25
    assert isinstance(b1["files"][0], str)

    print("\n" + "=" * 65)
    print("3. Small Batch Sequential Submission Test (Fold 1, Size 10 -> Mock Receiver)")
    print("=" * 65)

    small_submit_req = {
        "fold": 1,
        "batch_size": 10,
        "max_batches": 1,
        "target_url": MOCK_RECEIVER_URL,
        "payload_format": "c_tool",
    }
    resp = requests.post(f"{FASTAPI_BASE_URL}/api/v1/batches/submit", json=small_submit_req)
    assert resp.status_code == 200, f"Small batch submission failed: {resp.text}"
    small_res = resp.json()
    print(f"Total batches submitted: {small_res['total_batches']}")
    print(f"Successful: {small_res['successful_batches']}, Failed: {small_res['failed_batches']}")
    print(f"Batch 1 result: {small_res['results'][0]}")
    assert small_res["successful_batches"] == 1
    assert small_res["results"][0]["status"] == "success"

    print("\n" + "=" * 65)
    print("4. Direct Sequential Batch Submission to Actual C DSP Tool Binary")
    print("=" * 65)

    if os.path.isfile(C_TOOL_BIN):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_csv = os.path.join(tmpdir, "UrbanSound8K.csv")
            # Select 20 rows from fold 5 to match exactly what is submitted
            with open("/home/ORBITAL/soundmodel/UrbanSound8K_Processed/UrbanSound8K.csv", "r") as src:
                reader = csv.DictReader(src)
                fold5_rows = [r for r in reader if r["fold"] == "5"][:20]
            with open(test_csv, "w", newline="") as dst:
                writer = csv.DictWriter(dst, fieldnames=list(fold5_rows[0].keys()))
                writer.writeheader()
                writer.writerows(fold5_rows)

            # Start urban8k_tool on port 8085
            proc = subprocess.Popen([C_TOOL_BIN, "--port", str(TEST_C_TOOL_PORT), "--csv", test_csv])
            time.sleep(1)

            try:
                # Probe C tool status via FastAPI server
                status_probe = requests.get(
                    f"{FASTAPI_BASE_URL}/api/v1/batches/c-tool/status?health_url=http://127.0.0.1:{TEST_C_TOOL_PORT}/health"
                )
                print("C Tool status probe via FastAPI:", status_probe.json())

                # Submit 2 batches of 10 files directly through FastAPI server to urban8k_tool
                c_submit_req = {
                    "dataset_root": "/home/ORBITAL/soundmodel/UrbanSound8K_Processed",
                    "csv_filename": "UrbanSound8K.csv",
                    "fold": 5,
                    "batch_size": 10,
                    "max_batches": 2,
                    "target_url": TEST_C_TOOL_URL,
                    "payload_format": "c_tool",
                }
                c_resp = requests.post(f"{FASTAPI_BASE_URL}/api/v1/batches/submit", json=c_submit_req)
                assert c_resp.status_code == 200, f"C tool submission failed: {c_resp.text}"
                c_res = c_resp.json()
                print(f"Batches submitted to C tool: {c_res['total_batches']}")
                print(f"Successful batches: {c_res['successful_batches']}, Failed: {c_res['failed_batches']}")
                print("Processing Stats Section:")
                for k, v in c_res["stats"].items():
                    print(f"   {k}: {v}")
                
                # Check cross-checking / verification report directly
                verify_report = requests.get(
                    f"{FASTAPI_BASE_URL}/api/v1/dataset/verify-features?dataset_root={tmpdir}&csv_filename=UrbanSound8K.csv"
                ).json()
                print("Feature Cross-Checking Report on updated CSV:")
                print(f"   is_valid: {verify_report['is_valid']}")
                print(f"   feature_columns_found: {verify_report['feature_columns_found']}/62")
                print(f"   fully_mapped_rows: {verify_report['fully_mapped_rows']}/20")
                print(f"   unmapped_rows: {verify_report['unmapped_or_incomplete_rows']}")
                print(f"   discrepancy_count: {verify_report['discrepancy_count']}")
                if verify_report.get("sample_mapped_record"):
                    print(f"   sample_record: {verify_report['sample_mapped_record']['slice_file_name']} -> {verify_report['sample_mapped_record']['feature_sample']}")

                assert verify_report["feature_columns_found"] == 62
                assert verify_report["fully_mapped_rows"] == 20
                assert verify_report["is_valid"] is True
                print("Direct C DSP Engine Tool processing & CSV feature mapping verified successfully!")
            finally:
                proc.terminate()
                proc.wait()
    else:
        print(f"C tool binary not found at {C_TOOL_BIN}, skipped live binary test.")

    print("\n" + "=" * 65)
    print("5. All Folds Batch Preparation Test (8,732 files, batch size 50)")
    print("=" * 65)

    all_folds_req = {
        "fold": "all",
        "batch_size": 50,
    }
    start_time = time.perf_counter()
    resp = requests.post(f"{FASTAPI_BASE_URL}/api/v1/batches/prepare", json=all_folds_req)
    assert resp.status_code == 200, f"All folds preparation failed: {resp.text}"
    all_res = resp.json()
    elapsed = (time.perf_counter() - start_time) * 1000

    print(f"Total files: {all_res['total_valid_files']}")
    print(f"Total batches prepared: {all_res['total_batches']}")
    print(f"Preparation time: {round(elapsed, 2)} ms")
    assert all_res["total_valid_files"] == 8732
    assert all_res["total_batches"] == 175
    assert len(all_res["batches"]) == 175

    print("\n" + "=" * 65)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    verify()
