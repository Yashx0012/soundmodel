"""
Router for batch preparation, sequential submission to the C DSP tool (urban8k_tool),
automated cross-checking, and mock receiver testing.
"""

from datetime import datetime, timezone
import logging
import time
from typing import Optional, Union
from fastapi import APIRouter, HTTPException, Query, Request, status
import httpx

from app.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CSV_FILENAME,
    DEFAULT_C_TOOL_HEALTH_URL,
    DEFAULT_C_TOOL_URL,
    DEFAULT_TIMEOUT_SECONDS,
    resolve_dataset_root,
)
from app.models.schemas import (
    BatchPayload,
    CToolBatchRequest,
    CToolStatusResponse,
    MockReceiverResponse,
    PrepareBatchRequest,
    PrepareBatchResponse,
    ProcessingStats,
    SubmitBatchRequest,
    SubmitBatchResponse,
)
from app.services.batch_service import BatchService
from app.services.dataset_service import DatasetService
from app.services.verification_service import VerificationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/batches", tags=["Batches"])


@router.post(
    "/prepare",
    response_model=PrepareBatchResponse,
    summary="Prepare batches of validated WAV paths with CSV metadata",
    description="""
    Reads the UrbanSound8K CSV metadata, dynamically constructs complete WAV paths,
    verifies file existence on disk, and divides them into sequential batches.
    Each batch payload is pre-formatted for seamless submission to the C DSP tool (`urban8k_tool`).
    """,
)
def prepare_batches(request: PrepareBatchRequest) -> PrepareBatchResponse:
    try:
        root = resolve_dataset_root(request.dataset_root)
        valid_files, validation_errors, summary = DatasetService.load_and_validate(
            dataset_root=root,
            csv_filename=request.csv_filename or DEFAULT_CSV_FILENAME,
            fold_filter=request.fold,
        )

        batches = BatchService.divide_into_batches(
            files=valid_files,
            batch_size=request.batch_size,
            csv_path=summary["csv_path"],
            fold_identifier=request.fold if request.fold is not None else "all",
            max_batches=request.max_batches,
        )

        return PrepareBatchResponse(
            dataset_root=root,
            csv_path=summary["csv_path"],
            fold_filter=request.fold if request.fold is not None else "all",
            batch_size=request.batch_size,
            total_csv_records=summary["total_csv_rows"],
            total_valid_files=len(valid_files),
            total_invalid_or_missing=len(validation_errors),
            total_batches=len(batches),
            batches=batches,
            validation_errors=validation_errors,
        )
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        logger.exception("Unexpected error preparing batches: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error preparing batches: {exc}",
        )


@router.post(
    "/submit",
    response_model=SubmitBatchResponse,
    summary="Sequentially submit batch requests to the C DSP tool or custom target",
    description="""
    Reads the CSV, validates WAV paths, partitions them into batches, and submits each batch
    sequentially via HTTP POST to the target endpoint (e.g., http://127.0.0.1:8080/process).
    Batches are submitted strictly one-by-one without multithreading or multiprocessing.
    Includes comprehensive execution statistics and automated post-processing CSV feature cross-checking.
    """,
)
def submit_batches(request: SubmitBatchRequest) -> SubmitBatchResponse:
    start_total_time = time.perf_counter()

    try:
        root = resolve_dataset_root(request.dataset_root)
        valid_files, validation_errors, summary = DatasetService.load_and_validate(
            dataset_root=root,
            csv_filename=request.csv_filename or DEFAULT_CSV_FILENAME,
            fold_filter=request.fold,
        )

        batches = BatchService.divide_into_batches(
            files=valid_files,
            batch_size=request.batch_size,
            csv_path=summary["csv_path"],
            fold_identifier=request.fold if request.fold is not None else "all",
            max_batches=request.max_batches,
        )

        # Resolve destination target URL
        effective_target_url = request.target_url
        if not effective_target_url and request.use_c_tool:
            effective_target_url = DEFAULT_C_TOOL_URL

        # Execute strictly sequential submission
        results = BatchService.submit_batches_sequentially(
            batches=batches,
            target_url=effective_target_url,
            payload_format=request.payload_format,
            timeout_seconds=request.timeout_seconds,
        )

        total_duration_ms = (time.perf_counter() - start_total_time) * 1000
        total_duration_sec = total_duration_ms / 1000.0

        successful_batches = sum(1 for r in results if r.status in ("success", "simulated"))
        failed_batches = sum(1 for r in results if r.status == "failed")

        total_files_processed = sum(
            r.processed_count or 0 for r in results if r.status in ("success", "simulated")
        )
        total_files_failed = sum(
            r.failed_count or 0 for r in results if r.status == "failed"
        )

        avg_time_per_file = (
            (total_duration_ms / total_files_processed) if total_files_processed > 0 else 0.0
        )
        avg_time_per_batch = (
            (total_duration_ms / len(batches)) if len(batches) > 0 else 0.0
        )
        throughput = (
            (total_files_processed / total_duration_sec) if total_duration_sec > 0 else 0.0
        )

        # Compute C DSP engine time vs FastAPI overhead
        c_dsp_time_ms = sum(r.duration_ms for r in results)
        c_dsp_time_sec = c_dsp_time_ms / 1000.0
        fastapi_overhead_ms = max(0.0, total_duration_ms - c_dsp_time_ms)

        stats = ProcessingStats(
            total_files_requested=len(valid_files),
            total_files_processed=total_files_processed,
            total_files_failed=total_files_failed,
            total_batches=len(batches),
            successful_batches=successful_batches,
            failed_batches=failed_batches,
            total_duration_ms=round(total_duration_ms, 2),
            total_duration_seconds=round(total_duration_sec, 3),
            c_dsp_engine_time_ms=round(c_dsp_time_ms, 2),
            c_dsp_engine_time_seconds=round(c_dsp_time_sec, 3),
            fastapi_overhead_ms=round(fastapi_overhead_ms, 2),
            average_time_per_file_ms=round(avg_time_per_file, 2),
            average_time_per_batch_ms=round(avg_time_per_batch, 2),
            throughput_files_per_second=round(throughput, 1),
            csv_target_path=summary["csv_path"],
        )

        # Optional automated cross-checking of updated CSV
        verification_report = None
        if request.auto_cross_check and effective_target_url:
            try:
                submitted_filenames = {
                    item.filename for b in batches for item in b.files_metadata
                }
                verification_report = VerificationService.verify_csv_feature_mapping(
                    csv_path=summary["csv_path"],
                    fold_filter=request.fold,
                    expected_filenames=submitted_filenames,
                )
            except Exception as v_err:
                logger.warning("Automated feature cross-checking encountered error: %s", v_err)

        return SubmitBatchResponse(
            dataset_root=root,
            csv_path=summary["csv_path"],
            fold_filter=request.fold if request.fold is not None else "all",
            batch_size=request.batch_size,
            total_valid_files=len(valid_files),
            total_batches=len(batches),
            successful_batches=successful_batches,
            failed_batches=failed_batches,
            total_duration_ms=round(total_duration_ms, 2),
            stats=stats,
            verification=verification_report,
            results=results,
            validation_errors=validation_errors,
        )

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        logger.exception("Unexpected error submitting batches: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error submitting batches: {exc}",
        )


@router.post(
    "/mock-receiver",
    response_model=MockReceiverResponse,
    summary="Built-in mock receiver endpoint adhering to the urban8k_tool C API contract",
    description="""
    Accepts the C DSP tool batch JSON payload:
    {"csv_path": "...", "files": ["/path/1.wav", "/path/2.wav", ...]}
    Validates the structure and responds with the urban8k_tool response format.
    """,
)
async def mock_receiver(request: Request) -> MockReceiverResponse:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request body must be valid JSON",
        )

    # Extract files list from either CToolBatchRequest or BatchPayload
    files_list = body.get("files", [])
    csv_path = body.get("csv_path")

    # If passed as detailed objects, normalize
    normalized_files = []
    for f in files_list:
        if isinstance(f, str):
            normalized_files.append(f)
        elif isinstance(f, dict) and "wav_path" in f:
            normalized_files.append(f["wav_path"])

    if not normalized_files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Payload must contain a non-empty 'files' list.",
        )

    return MockReceiverResponse(
        status="ok",
        processed=len(normalized_files),
        failed=0,
        csv_path=csv_path,
        num_features=62,
        message=f"Mock receiver processed: {len(normalized_files)} succeeded, 0 failed",
    )


@router.get(
    "/c-tool/status",
    response_model=CToolStatusResponse,
    summary="Check status and connectivity of the C DSP Engine tool (urban8k_tool)",
)
def check_c_tool_status(
    health_url: Optional[str] = Query(None, description="Custom C tool health check URL"),
) -> CToolStatusResponse:
    target_health_url = health_url or DEFAULT_C_TOOL_HEALTH_URL
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(target_health_url)
            is_ok = resp.status_code == 200
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            return CToolStatusResponse(
                c_tool_url=DEFAULT_C_TOOL_URL,
                c_tool_health_url=target_health_url,
                c_tool_available=is_ok,
                status_code=resp.status_code,
                response=body,
                error=None if is_ok else f"HTTP {resp.status_code}",
            )
    except Exception as exc:
        return CToolStatusResponse(
            c_tool_url=DEFAULT_C_TOOL_URL,
            c_tool_health_url=target_health_url,
            c_tool_available=False,
            status_code=None,
            response=None,
            error=str(exc),
        )
