"""
Router for dataset inspection, summary information, sample preview,
and feature verification / cross-checking.
"""

import os
from typing import List, Optional, Union
from fastapi import APIRouter, HTTPException, Query, status

from app.config import DEFAULT_CSV_FILENAME, resolve_dataset_root
from app.models.schemas import (
    AudioFileItem,
    DatasetSummaryResponse,
    FeatureVerificationReport,
)
from app.services.dataset_service import DatasetService
from app.services.verification_service import VerificationService

router = APIRouter(prefix="/api/v1/dataset", tags=["Dataset"])


@router.get(
    "/info",
    response_model=DatasetSummaryResponse,
    summary="Get summary information about the UrbanSound8K dataset",
    description="Inspects dataset directory, CSV row count, verified files, and fold breakdown.",
)
def get_dataset_info(
    dataset_root: Optional[str] = Query(None, description="Path to dataset root"),
    csv_filename: Optional[str] = Query(DEFAULT_CSV_FILENAME, description="CSV filename"),
) -> DatasetSummaryResponse:
    try:
        root = resolve_dataset_root(dataset_root)
        valid_files, validation_errors, summary = DatasetService.load_and_validate(
            dataset_root=root,
            csv_filename=csv_filename or DEFAULT_CSV_FILENAME,
            fold_filter=None,
        )

        return DatasetSummaryResponse(
            dataset_root=root,
            csv_path=summary["csv_path"],
            total_records=summary["total_csv_rows"],
            verified_files_count=len(valid_files),
            missing_files_count=len(validation_errors),
            fold_distribution=summary["fold_distribution"],
            classes=summary["classes"],
        )
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))


@router.get(
    "/sample",
    response_model=List[AudioFileItem],
    summary="Preview a sample of records with constructed WAV paths and metadata",
)
def get_sample_records(
    fold: Optional[Union[int, str]] = Query(None, description="Fold number (1-10) or 'all'"),
    limit: int = Query(5, ge=1, le=100, description="Number of sample records to return"),
    dataset_root: Optional[str] = Query(None, description="Path to dataset root"),
) -> List[AudioFileItem]:
    try:
        root = resolve_dataset_root(dataset_root)
        valid_files, _, _ = DatasetService.load_and_validate(
            dataset_root=root,
            fold_filter=fold,
        )
        return valid_files[:limit]
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))


@router.get(
    "/verify-features",
    response_model=FeatureVerificationReport,
    summary="Cross-check and verify that the 62 acoustic features are correctly mapped to CSV rows",
    description="""
    Cross-checks the dataset CSV file to verify:
    1. That all 62 canonical acoustic feature columns exist.
    2. That each file row has non-empty, non-null, valid finite floating-point values for all features.
    3. Pinpoints any discrepancies, missing columns, or unmapped rows.
    """,
)
def verify_dataset_features(
    dataset_root: Optional[str] = Query(None, description="Path to dataset root"),
    csv_filename: Optional[str] = Query(DEFAULT_CSV_FILENAME, description="CSV filename"),
    fold: Optional[Union[int, str]] = Query(None, description="Fold number (1-10) or 'all'"),
) -> FeatureVerificationReport:
    try:
        root = resolve_dataset_root(dataset_root)
        csv_path = os.path.join(root, csv_filename or DEFAULT_CSV_FILENAME)
        report = VerificationService.verify_csv_feature_mapping(
            csv_path=csv_path,
            fold_filter=fold,
        )
        return report
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
