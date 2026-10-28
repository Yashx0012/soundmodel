"""
Batch service for dividing validated audio records into configurable batches,
generating API payloads compatible with the UrbanSound8K C DSP Feature Extraction Tool (`urban8k_tool`),
and executing strictly sequential batch HTTP submissions without multithreading or multiprocessing.
"""

import logging
import math
import time
from typing import Any, List, Optional, Union
import httpx
from pydantic import ValidationError

from app.models.schemas import (
    AudioFileItem,
    BatchPayload,
    BatchSubmissionResult,
)

logger = logging.getLogger(__name__)


class BatchService:
    """
    Manages batch creation and strictly sequential batch submission.
    """

    @staticmethod
    def divide_into_batches(
        files: List[AudioFileItem],
        batch_size: int,
        csv_path: str = "",
        fold_identifier: Optional[Union[int, str]] = None,
        max_batches: Optional[int] = None,
    ) -> List[BatchPayload]:
        """
        Sequentially divides validated audio files into batches.

        Args:
            files: List of validated AudioFileItem records.
            batch_size: Size of each batch (number of files).
            csv_path: Absolute path to the UrbanSound8K.csv dataset file.
            fold_identifier: Fold number or 'all' for labeling the payload.
            max_batches: Optional limit on the number of batches to prepare.

        Returns:
            List of BatchPayload objects.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be greater than 0, got {batch_size}")

        total_files = len(files)
        if total_files == 0:
            return []

        total_batches_possible = math.ceil(total_files / batch_size)
        if max_batches is not None and max_batches > 0:
            total_batches = min(total_batches_possible, max_batches)
        else:
            total_batches = total_batches_possible

        batches: List[BatchPayload] = []

        # Strictly sequential division
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_files)
            batch_items = files[start_idx:end_idx]

            # Construct verified WAV path strings for the C DSP Tool
            wav_paths = [item.wav_path for item in batch_items]

            payload = BatchPayload(
                fold=fold_identifier if fold_identifier is not None else "all",
                batch_number=batch_idx + 1,
                total_batches=total_batches,
                batch_size=batch_size,
                files_count=len(batch_items),
                csv_path=csv_path,
                files=wav_paths,
                files_metadata=batch_items,
            )
            batches.append(payload)

        return batches

    @staticmethod
    def submit_batches_sequentially(
        batches: List[BatchPayload],
        target_url: Optional[str] = None,
        payload_format: str = "c_tool",
        timeout_seconds: float = 60.0,
    ) -> List[BatchSubmissionResult]:
        """
        Submits prepared batches strictly sequentially to the target API endpoint.
        NO multithreading, NO multiprocessing, NO ThreadPoolExecutor.
        Each batch submission must complete before the next batch begins.

        Args:
            batches: List of prepared BatchPayload objects.
            target_url: Destination URL for HTTP POST submission (e.g. http://127.0.0.1:8080/process).
            payload_format: 'c_tool' (exact urban8k_tool JSON contract) or 'detailed'.
            timeout_seconds: Timeout per batch request in seconds.

        Returns:
            List of BatchSubmissionResult objects.
        """
        results: List[BatchSubmissionResult] = []

        # If no target URL provided, mark batches as simulated
        if not target_url:
            for b in batches:
                results.append(
                    BatchSubmissionResult(
                        batch_number=b.batch_number,
                        total_batches=b.total_batches,
                        files_count=b.files_count,
                        status="simulated",
                        target_url=None,
                        http_status_code=None,
                        processed_count=b.files_count,
                        failed_count=0,
                        response_body={
                            "message": "Batch prepared successfully. No target_url specified for submission.",
                            "c_tool_payload_sample": b.to_c_tool_payload(),
                        },
                        error_message=None,
                        duration_ms=0.0,
                    )
                )
            return results

        # Create sequential HTTP client
        with httpx.Client(timeout=timeout_seconds) as client:
            for b in batches:
                start_time = time.perf_counter()

                # Step 1: Prepare payload format matching target endpoint contract
                try:
                    if payload_format == "c_tool":
                        # Exact contract expected by urban8k_tool C DSP engine:
                        # {"csv_path": "/path/to/UrbanSound8K.csv", "files": ["/path/1.wav", "/path/2.wav", ...]}
                        payload_data = b.to_c_tool_payload()
                    else:
                        payload_data = b.model_dump(by_alias=True)

                    # Verify that files are present
                    if not payload_data.get("files"):
                        raise ValueError(f"Batch {b.batch_number} contains 0 files.")

                except (ValidationError, ValueError, Exception) as ve:
                    elapsed = (time.perf_counter() - start_time) * 1000
                    results.append(
                        BatchSubmissionResult(
                            batch_number=b.batch_number,
                            total_batches=b.total_batches,
                            files_count=b.files_count,
                            status="failed",
                            target_url=target_url,
                            http_status_code=None,
                            processed_count=0,
                            failed_count=b.files_count,
                            response_body=None,
                            error_message=f"Batch payload preparation failed: {ve}",
                            duration_ms=round(elapsed, 2),
                        )
                    )
                    continue

                # Step 2: Sequential HTTP POST submission
                try:
                    response = client.post(
                        target_url,
                        json=payload_data,
                        headers={"Content-Type": "application/json"},
                    )
                    elapsed = (time.perf_counter() - start_time) * 1000

                    try:
                        resp_body = response.json()
                    except Exception:
                        resp_body = response.text[:500]

                    # Parse response attributes returned by urban8k_tool or API
                    processed_count = None
                    failed_count = None
                    is_success = 200 <= response.status_code < 300

                    if isinstance(resp_body, dict):
                        processed_count = resp_body.get("processed")
                        if processed_count is None:
                            processed_count = resp_body.get("processed_count")

                        failed_count = resp_body.get("failed")
                        if failed_count is None:
                            failed_count = resp_body.get("error_count", 0)

                        # Check if tool reported failure in JSON
                        status_field = str(resp_body.get("status", "")).lower()
                        if status_field in ("error", "failed"):
                            is_success = False
                        elif failed_count and failed_count > 0:
                            is_success = False

                    status_str = "success" if is_success else "failed"
                    err_msg = None if is_success else f"HTTP {response.status_code}: {resp_body}"

                    results.append(
                        BatchSubmissionResult(
                            batch_number=b.batch_number,
                            total_batches=b.total_batches,
                            files_count=b.files_count,
                            status=status_str,
                            target_url=target_url,
                            http_status_code=response.status_code,
                            processed_count=processed_count if processed_count is not None else (b.files_count if is_success else 0),
                            failed_count=failed_count if failed_count is not None else (0 if is_success else b.files_count),
                            response_body=resp_body,
                            error_message=err_msg,
                            duration_ms=round(elapsed, 2),
                        )
                    )

                except httpx.RequestError as exc:
                    elapsed = (time.perf_counter() - start_time) * 1000
                    results.append(
                        BatchSubmissionResult(
                            batch_number=b.batch_number,
                            total_batches=b.total_batches,
                            files_count=b.files_count,
                            status="failed",
                            target_url=target_url,
                            http_status_code=None,
                            processed_count=0,
                            failed_count=b.files_count,
                            response_body=None,
                            error_message=f"Request error submitting batch to {target_url}: {exc}",
                            duration_ms=round(elapsed, 2),
                        )
                    )

        return results
