"""
Verification and cross-checking service for verifying that the C DSP Tool (`urban8k_tool`)
has correctly updated and mapped the 62 acoustic feature vectors to the corresponding CSV rows.
"""

import csv
import logging
import math
import os
from typing import List, Optional, Set, Union

from app.models.schemas import FeatureDiscrepancyItem, FeatureVerificationReport

logger = logging.getLogger(__name__)

# Canonical 62 feature columns produced by the C DSP Engine (`urban8k_tool`)
CANONICAL_62_FEATURE_COLUMNS = [
    # Spectral moments (Mean)
    "mean_spectral_centroid",
    "mean_spectral_spread",
    "mean_spectral_rolloff",
    "mean_spectral_flatness",
    "mean_spectral_flux",
    "mean_spectral_zcr",
    # MFCC 0-12 (Mean)
    "mean_mfcc_0",
    "mean_mfcc_1",
    "mean_mfcc_2",
    "mean_mfcc_3",
    "mean_mfcc_4",
    "mean_mfcc_5",
    "mean_mfcc_6",
    "mean_mfcc_7",
    "mean_mfcc_8",
    "mean_mfcc_9",
    "mean_mfcc_10",
    "mean_mfcc_11",
    "mean_mfcc_12",
    # Chroma 0-11 (Mean)
    "mean_chroma_0",
    "mean_chroma_1",
    "mean_chroma_2",
    "mean_chroma_3",
    "mean_chroma_4",
    "mean_chroma_5",
    "mean_chroma_6",
    "mean_chroma_7",
    "mean_chroma_8",
    "mean_chroma_9",
    "mean_chroma_10",
    "mean_chroma_11",
    # Spectral moments (Std/Variance)
    "std_spectral_centroid",
    "std_spectral_spread",
    "std_spectral_rolloff",
    "std_spectral_flatness",
    "std_spectral_flux",
    "std_spectral_zcr",
    # MFCC 0-12 (Std/Variance)
    "std_mfcc_0",
    "std_mfcc_1",
    "std_mfcc_2",
    "std_mfcc_3",
    "std_mfcc_4",
    "std_mfcc_5",
    "std_mfcc_6",
    "std_mfcc_7",
    "std_mfcc_8",
    "std_mfcc_9",
    "std_mfcc_10",
    "std_mfcc_11",
    "std_mfcc_12",
    # Chroma 0-11 (Std/Variance)
    "std_chroma_0",
    "std_chroma_1",
    "std_chroma_2",
    "std_chroma_3",
    "std_chroma_4",
    "std_chroma_5",
    "std_chroma_6",
    "std_chroma_7",
    "std_chroma_8",
    "std_chroma_9",
    "std_chroma_10",
    "std_chroma_11",
]


class VerificationService:
    """
    Validates CSV feature column injection, row matching, and numeric integrity.
    """

    @staticmethod
    def verify_csv_feature_mapping(
        csv_path: str,
        fold_filter: Optional[Union[int, str]] = None,
        expected_filenames: Optional[Set[str]] = None,
        max_discrepancies: int = 50,
    ) -> FeatureVerificationReport:
        """
        Cross-checks that all expected WAV records in the CSV have their 62 acoustic
        features correctly populated with finite floating-point numbers.

        Args:
            csv_path: Path to the UrbanSound8K CSV file.
            fold_filter: Optional fold number (1-10) or 'all' to restrict check.
            expected_filenames: Optional set of filenames expected to be populated.
            max_discrepancies: Max number of discrepancies to store in report.

        Returns:
            FeatureVerificationReport with full cross-checking breakdown.
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file to verify does not exist: {csv_path}")

        target_fold: Optional[int] = None
        if fold_filter is not None and str(fold_filter).strip().lower() not in ("all", "none", ""):
            try:
                target_fold = int(fold_filter)
            except ValueError:
                pass

        total_rows_inspected = 0
        fully_mapped_rows = 0
        unmapped_rows = 0
        discrepancies: List[FeatureDiscrepancyItem] = []
        sample_record: Optional[dict] = None

        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            # Check which canonical feature columns exist in the CSV header
            found_feature_cols = [c for c in CANONICAL_62_FEATURE_COLUMNS if c in fieldnames]

            # If canonical names aren't present, check for alternative alias names
            if not found_feature_cols:
                # check for non-prefix or 'var_' prefix style
                found_feature_cols = [
                    c for c in fieldnames
                    if any(c.startswith(p) for p in ("spec_", "mfcc_", "chroma_", "var_", "rms_", "zcr"))
                ]

            for row_idx, row in enumerate(reader, start=1):
                raw_fold = row.get("fold", "").strip()
                try:
                    row_fold = int(raw_fold)
                except (ValueError, TypeError):
                    row_fold = -1

                # Apply fold filter
                if target_fold is not None and row_fold != target_fold:
                    continue

                filename = row.get("slice_file_name", "").strip()

                # If an expected filename set is passed, skip other rows
                if expected_filenames is not None and filename not in expected_filenames:
                    continue

                total_rows_inspected += 1

                # Check if features exist and are populated with valid numbers
                if len(found_feature_cols) < 62:
                    if len(discrepancies) < max_discrepancies:
                        discrepancies.append(
                            FeatureDiscrepancyItem(
                                row_index=row_idx,
                                slice_file_name=filename,
                                fold=row_fold,
                                issue_type="missing_columns",
                                details=f"CSV header only contains {len(found_feature_cols)}/62 feature columns.",
                            )
                        )
                    unmapped_rows += 1
                    continue

                # Verify every feature column in this row
                missing_or_invalid = []
                for col in found_feature_cols:
                    val = row.get(col, "").strip()
                    if not val:
                        missing_or_invalid.append(f"{col}=empty")
                    else:
                        try:
                            fval = float(val)
                            if math.isnan(fval) or math.isinf(fval):
                                missing_or_invalid.append(f"{col}=nan_or_inf")
                        except ValueError:
                            missing_or_invalid.append(f"{col}=not_a_number")

                if missing_or_invalid:
                    unmapped_rows += 1
                    if len(discrepancies) < max_discrepancies:
                        discrepancies.append(
                            FeatureDiscrepancyItem(
                                row_index=row_idx,
                                slice_file_name=filename,
                                fold=row_fold,
                                issue_type="null_or_invalid_features",
                                details=f"{len(missing_or_invalid)} features invalid. Samples: {', '.join(missing_or_invalid[:3])}",
                            )
                        )
                else:
                    fully_mapped_rows += 1
                    if sample_record is None:
                        # Capture a sample mapped record
                        sample_features = {c: float(row[c]) for c in found_feature_cols[:6]}
                        sample_record = {
                            "slice_file_name": filename,
                            "fold": row_fold,
                            "class": row.get("class", ""),
                            "feature_sample": sample_features,
                        }

        is_valid = (
            len(found_feature_cols) == 62
            and unmapped_rows == 0
            and fully_mapped_rows > 0
            and total_rows_inspected > 0
        )

        return FeatureVerificationReport(
            is_valid=is_valid,
            csv_path=csv_path,
            total_rows_inspected=total_rows_inspected,
            feature_columns_found=len(found_feature_cols),
            feature_columns_list=found_feature_cols,
            fully_mapped_rows=fully_mapped_rows,
            unmapped_or_incomplete_rows=unmapped_rows,
            discrepancy_count=len(discrepancies),
            discrepancies=discrepancies,
            sample_mapped_record=sample_record,
        )
