"""
Pydantic schemas for UrbanSound8K CSV parsing, batch preparation,
C DSP tool integration, sequential batch API submission,
and post-processing cross-checking / verification.
"""

from typing import Any, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class AudioFileItem(BaseModel):
    """
    Metadata and constructed WAV path for a single audio file.
    """
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    filename: str = Field(..., description="WAV file name (slice_file_name)")
    fold: int = Field(..., description="Fold directory number (1-10)")
    wav_path: str = Field(..., description="Absolute verified path to the WAV file")
    classID: int = Field(..., description="Numeric class identifier (0-9)")
    class_name: str = Field(..., alias="class", description="Class label name")

    # Optional metadata from CSV
    fsID: Optional[str] = Field(None, description="Freesound ID")
    start: Optional[float] = Field(None, description="Start time in original recording")
    end: Optional[float] = Field(None, description="End time in original recording")
    salience: Optional[int] = Field(None, description="Salience indicator")
    file_exists: bool = Field(True, description="Whether the file exists on filesystem")


class ValidationErrorItem(BaseModel):
    """
    Record of a validation failure encountered when reading CSV or checking paths.
    """
    row_index: int = Field(..., description="Zero-based or one-based CSV row number")
    filename: Optional[str] = Field(None, description="Filename from CSV if available")
    fold: Optional[Union[int, str]] = Field(None, description="Fold value from CSV if available")
    error_type: str = Field(..., description="Category of error (missing_file, invalid_format, etc.)")
    message: str = Field(..., description="Detailed description of the error")


class CToolBatchRequest(BaseModel):
    """
    Exact payload expected by the UrbanSound8K C DSP Feature Extraction Tool (`urban8k_tool`).
    POST /process
    """
    csv_path: Optional[str] = Field(None, description="Path to UrbanSound8K.csv")
    files: List[str] = Field(..., description="List of absolute WAV file paths to process")


class CToolBatchResponse(BaseModel):
    """
    Response returned by the UrbanSound8K C DSP Feature Extraction Tool (`urban8k_tool`).
    """
    status: str = Field(..., description="Status ('ok' or 'success')")
    processed: Optional[int] = Field(None, description="Count of successfully processed files")
    processed_count: Optional[int] = Field(None, description="Alias for processed files count")
    failed: Optional[int] = Field(None, description="Count of failed files")
    error_count: Optional[int] = Field(None, description="Alias for failed files count")
    csv_path: Optional[str] = Field(None, description="Path to updated CSV file")
    csv_updated: Optional[str] = Field(None, description="Alias for updated CSV file path")
    num_features: Optional[int] = Field(None, description="Number of acoustic features extracted (62)")
    message: Optional[str] = Field(None, description="Status/summary message from C tool")


class BatchPayload(BaseModel):
    """
    Payload for an individual batch prepared for API submission.
    Provides compatibility with both urban8k_tool C API and rich metadata clients.
    """
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    fold: Optional[Union[int, str]] = Field(..., description="Selected fold number (1-10) or 'all'")
    batch_number: int = Field(..., description="Sequential batch number (1-indexed)")
    total_batches: int = Field(..., description="Total number of batches in this sequence")
    batch_size: int = Field(..., description="Configured batch size limit")
    files_count: int = Field(..., description="Actual number of files included in this batch")
    csv_path: str = Field(..., description="Absolute path to UrbanSound8K.csv")

    # Formatted specifically for the C DSP Tool:
    files: List[str] = Field(..., description="List of verified WAV paths matching urban8k_tool contract")

    # Associated rich metadata for caller transparency:
    files_metadata: List[AudioFileItem] = Field(
        default_factory=list,
        description="Detailed metadata for each file in this batch"
    )

    def to_c_tool_payload(self) -> dict:
        """Returns the exact JSON dictionary expected by urban8k_tool /process."""
        return {
            "csv_path": self.csv_path,
            "files": self.files,
        }


class PrepareBatchRequest(BaseModel):
    """
    Input parameters to prepare batches without submitting.
    """
    dataset_root: Optional[str] = Field(
        None,
        description="Path to dataset root directory (defaults to configured UrbanSound8K_Processed)"
    )
    csv_filename: Optional[str] = Field(
        "UrbanSound8K.csv",
        description="CSV filename located in dataset root"
    )
    fold: Optional[Union[int, str]] = Field(
        None,
        description="Fold to process (1-10, or 'all', or None for all folds)"
    )
    batch_size: int = Field(
        50,
        ge=1,
        le=5000,
        description="Number of WAV paths per batch (default 50)"
    )
    max_batches: Optional[int] = Field(
        None,
        ge=1,
        description="Optional limit on number of batches to prepare"
    )


class PrepareBatchResponse(BaseModel):
    """
    Response returned by the batch preparation endpoint.
    """
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    dataset_root: str
    csv_path: str
    fold_filter: Optional[Union[int, str]]
    batch_size: int
    total_csv_records: int
    total_valid_files: int
    total_invalid_or_missing: int
    total_batches: int
    batches: List[BatchPayload]
    validation_errors: List[ValidationErrorItem]


class SubmitBatchRequest(BaseModel):
    """
    Input parameters to prepare and sequentially submit batch requests.
    """
    dataset_root: Optional[str] = Field(
        None,
        description="Dataset root directory"
    )
    csv_filename: Optional[str] = Field(
        "UrbanSound8K.csv",
        description="CSV filename located in dataset root"
    )
    fold: Optional[Union[int, str]] = Field(
        None,
        description="Fold to process (1-10, or 'all', or None for all folds)"
    )
    batch_size: int = Field(
        50,
        ge=1,
        le=5000,
        description="Configured batch size (default 50)"
    )
    max_batches: Optional[int] = Field(
        None,
        ge=1,
        description="Limit number of batches to submit (e.g. 1 or 2 for testing)"
    )
    target_url: Optional[str] = Field(
        None,
        description="Target receiving endpoint URL. If None and use_c_tool=True, defaults to C tool (http://127.0.0.1:8080/process)."
    )
    use_c_tool: bool = Field(
        False,
        description="Set to True to target the default local C DSP Engine tool (http://127.0.0.1:8080/process)"
    )
    payload_format: str = Field(
        "c_tool",
        description="Payload format: 'c_tool' (exact urban8k_tool contract: csv_path + files) or 'detailed'"
    )
    auto_cross_check: bool = Field(
        True,
        description="Whether to automatically cross-check and verify updated CSV features after submission"
    )
    timeout_seconds: float = Field(
        60.0,
        gt=0,
        le=600.0,
        description="HTTP request timeout per batch submission in seconds"
    )


class BatchSubmissionResult(BaseModel):
    """
    Status and details of a single submitted batch.
    """
    batch_number: int
    total_batches: int
    files_count: int
    status: str = Field(..., description="'success', 'failed', or 'simulated'")
    target_url: Optional[str] = None
    http_status_code: Optional[int] = None
    processed_count: Optional[int] = None
    failed_count: Optional[int] = None
    response_body: Optional[Any] = None
    error_message: Optional[str] = None
    duration_ms: float = Field(0.0, description="Submission duration in milliseconds")


class ProcessingStats(BaseModel):
    """
    Comprehensive performance and progress metrics for the batch submission run.
    """
    total_files_requested: int = Field(..., description="Total files partitioned for submission")
    total_files_processed: int = Field(..., description="Total files successfully processed")
    total_files_failed: int = Field(..., description="Total files that failed processing")
    total_batches: int = Field(..., description="Total number of batches in run")
    successful_batches: int = Field(..., description="Number of successful batches")
    failed_batches: int = Field(..., description="Number of failed batches")
    total_duration_ms: float = Field(..., description="Total execution duration in milliseconds")
    total_duration_seconds: float = Field(..., description="Total execution duration in seconds")
    c_dsp_engine_time_ms: float = Field(0.0, description="Total time spent on C DSP engine execution across all batches (ms)")
    c_dsp_engine_time_seconds: float = Field(0.0, description="Total time spent on C DSP engine execution in seconds")
    fastapi_overhead_ms: float = Field(0.0, description="FastAPI server overhead (CSV parsing, path validation, payload formatting) in ms")
    average_time_per_file_ms: float = Field(..., description="Average processing time per file in milliseconds")
    average_time_per_batch_ms: float = Field(..., description="Average duration per batch in milliseconds")
    throughput_files_per_second: float = Field(..., description="Processing throughput in audio files per second")
    csv_target_path: str = Field(..., description="Path to the targeted CSV file")


class FeatureDiscrepancyItem(BaseModel):
    """
    Details of a single row cross-checking failure or feature mapping anomaly.
    """
    row_index: int
    slice_file_name: str
    fold: int
    issue_type: str = Field(..., description="missing_columns, null_features, nan_inf_values, missing_row")
    details: str


class FeatureVerificationReport(BaseModel):
    """
    Comprehensive cross-checking report verifying that the C DSP Tool correctly
    injected and mapped 62 acoustic feature columns to the exact corresponding CSV rows.
    """
    is_valid: bool = Field(..., description="True if all inspected rows have complete, valid 62-D features")
    csv_path: str = Field(..., description="Path to the inspected CSV file")
    total_rows_inspected: int = Field(..., description="Total CSV rows inspected")
    feature_columns_found: int = Field(..., description="Number of canonical feature columns found (expected 62)")
    feature_columns_list: List[str] = Field(default_factory=list, description="List of feature column names")
    fully_mapped_rows: int = Field(..., description="Rows with complete and valid numeric feature values")
    unmapped_or_incomplete_rows: int = Field(..., description="Rows missing some or all feature columns")
    discrepancy_count: int = Field(..., description="Total discrepancies detected")
    discrepancies: List[FeatureDiscrepancyItem] = Field(
        default_factory=list,
        description="Detailed list of discrepancies found (capped at 50 for readability)"
    )
    sample_mapped_record: Optional[dict] = Field(
        None,
        description="Sample verified row with its mapped features for inspection"
    )


class SubmitBatchResponse(BaseModel):
    """
    Summary, stats, and cross-checking breakdown of sequential batch submission.
    """
    dataset_root: str
    csv_path: str
    fold_filter: Optional[Union[int, str]]
    batch_size: int
    total_valid_files: int
    total_batches: int
    successful_batches: int
    failed_batches: int
    total_duration_ms: float

    # Dedicated statistics section
    stats: ProcessingStats = Field(
        ...,
        description="Comprehensive statistics: total files, total time consumed, avg latency, throughput"
    )

    # Automated cross-checking & feature mapping verification report
    verification: Optional[FeatureVerificationReport] = Field(
        None,
        description="Cross-checking report confirming features are correctly mapped to corresponding rows"
    )

    results: List[BatchSubmissionResult]
    validation_errors: List[ValidationErrorItem]


class MockReceiverResponse(BaseModel):
    """
    Response returned by the built-in mock receiver endpoint (matches urban8k_tool output format).
    """
    status: str = "ok"
    processed: int
    failed: int = 0
    csv_path: Optional[str] = None
    num_features: int = 62
    message: str


class DatasetSummaryResponse(BaseModel):
    """
    Information regarding the dataset directory, folds, and metadata.
    """
    dataset_root: str
    csv_path: str
    total_records: int
    verified_files_count: int
    missing_files_count: int
    fold_distribution: dict[int, int]
    classes: dict[int, str]


class CToolStatusResponse(BaseModel):
    """
    Health and connectivity status of the C DSP feature extraction tool.
    """
    c_tool_url: str
    c_tool_health_url: str
    c_tool_available: bool
    status_code: Optional[int] = None
    response: Optional[Any] = None
    error: Optional[str] = None
