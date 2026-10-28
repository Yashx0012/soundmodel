#!/usr/bin/env python3
"""
CLI script to trigger batch processing and display complete stats and cross-checking reports.
Usage:
    python3 trigger_batch_processing.py --fold 1
    python3 trigger_batch_processing.py --fold all --batch-size 50
"""

import argparse
import json
import sys
import time
import requests

API_URL = "http://127.0.0.1:8000/api/v1/batches/submit"
HEALTH_URL = "http://127.0.0.1:8000/api/v1/batches/c-tool/status"


def main():
    parser = argparse.ArgumentParser(
        description="Trigger UrbanSound8K batch processing and print stats & verification report"
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="1",
        help="Fold number (1-10) or 'all' to process all folds (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size (number of WAV files per batch, default: 50)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit on number of batches (e.g., 2 for a quick test)",
    )
    parser.add_argument(
        "--target-url",
        type=str,
        default="http://127.0.0.1:8080/process",
        help="C tool process endpoint URL (default: http://127.0.0.1:8080/process)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" 🚀 TRIGGERING URBANSOUND8K BATCH PROCESSING PIPELINE")
    print("=" * 70)

    # 1. Probe C Tool connection
    print("🔍 Checking C DSP Engine server status...")
    try:
        probe = requests.get(HEALTH_URL, timeout=3.0)
        p_data = probe.json()
        if p_data.get("c_tool_available"):
            print("   ✅ C DSP Tool is ONLINE and READY on http://127.0.0.1:8080")
        else:
            print("   ⚠️ C DSP Tool probe failed:", p_data.get("error"))
    except Exception as e:
        print(f"   ⚠️ Could not reach FastAPI server: {e}")
        print("   Make sure 'python3 run_server.py --port 8000' is running.")
        sys.exit(1)

    # 2. Build payload
    fold_param = int(args.fold) if args.fold.isdigit() else args.fold
    payload = {
        "fold": fold_param,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "use_c_tool": True,
        "target_url": args.target_url,
        "payload_format": "c_tool",
        "auto_cross_check": True,
    }

    print(f"📦 Submitting job: fold={fold_param}, batch_size={args.batch_size}, target={args.target_url}")
    print("⏳ Processing batches sequentially (please wait)...")

    start_time = time.perf_counter()
    resp = requests.post(API_URL, json=payload, timeout=600)
    roundtrip_sec = time.perf_counter() - start_time

    if resp.status_code != 200:
        print("❌ Error from server:", resp.status_code, resp.text)
        sys.exit(1)

    data = resp.json()
    stats = data.get("stats", {})
    verification = data.get("verification", {})

    # 3. Print Stats Section
    print("\n" + "=" * 70)
    print(" 📊 EXECUTION STATISTICS SECTION")
    print("=" * 70)
    print(f" • Total Files Requested:      {stats.get('total_files_requested')}")
    print(f" • Total Files Processed:      {stats.get('total_files_processed')} / {stats.get('total_files_requested')}")
    print(f" • Total Files Failed:         {stats.get('total_files_failed')}")
    print(f" • Total Batches Submitted:    {stats.get('total_batches')} (Success: {stats.get('successful_batches')}, Failed: {stats.get('failed_batches')})")
    print(f" • Total Overall Time:         {stats.get('total_duration_seconds'):.3f} seconds ({stats.get('total_duration_ms'):.1f} ms)")
    print(f" • ⚡ C DSP Engine Time:        {stats.get('c_dsp_engine_time_seconds'):.3f} seconds ({stats.get('c_dsp_engine_time_ms'):.1f} ms)")
    print(f" • 🐍 FastAPI Server Overhead:  {stats.get('fastapi_overhead_ms'):.1f} ms")
    print(f" • Average Time Per File:      {stats.get('average_time_per_file_ms'):.2f} ms / file")
    print(f" • Throughput Speed:           {stats.get('throughput_files_per_second'):.1f} files / second")
    print(f" • Target CSV Updated:         {stats.get('csv_target_path')}")

    # 4. Print Feature Cross-Checking Report
    print("\n" + "=" * 70)
    print(" 🔬 AUTOMATED FEATURE CROSS-CHECKING & MAPPING REPORT")
    print("=" * 70)
    if verification:
        status_icon = "✅ PASSED" if verification.get("is_valid") else "⚠️ ANOMALIES DETECTED"
        print(f" • Cross-Check Status:         {status_icon}")
        print(f" • Feature Columns Found:      {verification.get('feature_columns_found')} / 62 canonical columns")
        print(f" • Inspected CSV Rows:         {verification.get('total_rows_inspected')}")
        print(f" • Fully Mapped Rows:          {verification.get('fully_mapped_rows')}")
        print(f" • Unmapped / Incomplete Rows: {verification.get('unmapped_or_incomplete_rows')}")
        print(f" • Discrepancies Count:        {verification.get('discrepancy_count')}")

        sample = verification.get("sample_mapped_record")
        if sample:
            print(f"\n   Sample Verified Record in CSV:")
            print(f"   - File:  {sample.get('slice_file_name')} (Fold {sample.get('fold')}, Class: {sample.get('class')})")
            print(f"   - Sample Extracted Features: {json.dumps(sample.get('feature_sample'), indent=6)}")
    else:
        print("   Verification report not requested or unavailable.")

    print("\n" + "=" * 70)
    print(" ✨ BATCH PROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
