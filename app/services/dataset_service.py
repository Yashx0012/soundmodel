"""
Dataset service for reading UrbanSound8K CSV metadata,
constructing absolute WAV paths, validating file existence,
and associating metadata without modifying source files.
"""

import csv
import logging
import os
from typing import List, Optional, Tuple, Union

from app.config import resolve_dataset_root
from app.models.schemas import AudioFileItem, ValidationErrorItem

logger = logging.getLogger(__name__)


class DatasetService:
    """
    Handles CSV reading, dynamic WAV path construction, and path verification.
    """

    @staticmethod
    def load_and_validate(
        dataset_root: Optional[str] = None,
        csv_filename: str = "UrbanSound8K.csv",
        fold_filter: Optional[Union[int, str]] = None,
    ) -> Tuple[List[AudioFileItem], List[ValidationErrorItem], dict]:
        """
        Reads CSV metadata, constructs WAV paths, verifies file existence,
        and returns validated items alongside any validation errors.

        Args:
            dataset_root: Root directory of dataset. If None, resolves from default config.
            csv_filename: CSV file name inside dataset_root (default UrbanSound8K.csv).
            fold_filter: Optional fold number (1-10) or 'all' / None for all folds.

        Returns:
            Tuple of (valid_items, validation_errors, summary_dict)
        """
        root = resolve_dataset_root(dataset_root)
        csv_path = os.path.join(root, csv_filename)

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        # Normalize fold_filter
        target_fold: Optional[int] = None
        if fold_filter is not None and str(fold_filter).strip().lower() not in ("all", "none", ""):
            try:
                target_fold = int(fold_filter)
                if not (1 <= target_fold <= 10):
                    raise ValueError(f"Fold number must be between 1 and 10, got {target_fold}")
            except ValueError as e:
                raise ValueError(f"Invalid fold filter '{fold_filter}': {e}") from e

        valid_items: List[AudioFileItem] = []
        validation_errors: List[ValidationErrorItem] = []
        total_rows = 0
        fold_counts: dict[int, int] = {}
        class_mapping: dict[int, str] = {}

        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            required_fields = {"slice_file_name", "fold", "classID", "class"}
            missing_required = required_fields - fieldnames
            if missing_required:
                raise ValueError(
                    f"CSV missing mandatory columns: {missing_required}. Present columns: {fieldnames}"
                )

            for row_idx, row in enumerate(reader, start=1):
                total_rows += 1

                # Parse and validate fold
                raw_fold = row.get("fold", "").strip()
                try:
                    fold_num = int(raw_fold)
                except (ValueError, TypeError):
                    validation_errors.append(
                        ValidationErrorItem(
                            row_index=row_idx,
                            filename=row.get("slice_file_name"),
                            fold=raw_fold,
                            error_type="invalid_fold",
                            message=f"Row {row_idx}: fold '{raw_fold}' is not a valid integer",
                        )
                    )
                    continue

                # Filter by fold if requested
                if target_fold is not None and fold_num != target_fold:
                    continue

                # Parse slice_file_name
                filename = row.get("slice_file_name", "").strip()
                if not filename:
                    validation_errors.append(
                        ValidationErrorItem(
                            row_index=row_idx,
                            filename=filename,
                            fold=fold_num,
                            error_type="missing_filename",
                            message=f"Row {row_idx}: slice_file_name is empty",
                        )
                    )
                    continue

                # Parse classID
                raw_cid = row.get("classID", "").strip()
                try:
                    class_id = int(raw_cid)
                except (ValueError, TypeError):
                    validation_errors.append(
                        ValidationErrorItem(
                            row_index=row_idx,
                            filename=filename,
                            fold=fold_num,
                            error_type="invalid_class_id",
                            message=f"Row {row_idx}: classID '{raw_cid}' is not a valid integer",
                        )
                    )
                    continue

                # Parse class label name
                class_label = row.get("class", "").strip()
                if not class_label:
                    validation_errors.append(
                        ValidationErrorItem(
                            row_index=row_idx,
                            filename=filename,
                            fold=fold_num,
                            error_type="missing_class_name",
                            message=f"Row {row_idx}: class name is empty",
                        )
                    )
                    continue

                # Record class mapping
                class_mapping[class_id] = class_label

                # Construct dynamic full WAV path using dataset root, fold, and slice_file_name
                constructed_wav_path = os.path.abspath(
                    os.path.join(root, f"fold{fold_num}", filename)
                )

                # Verify file existence on disk without positional indexing
                if not os.path.isfile(constructed_wav_path):
                    validation_errors.append(
                        ValidationErrorItem(
                            row_index=row_idx,
                            filename=filename,
                            fold=fold_num,
                            error_type="missing_wav_file",
                            message=f"Row {row_idx}: WAV file not found at '{constructed_wav_path}'",
                        )
                    )
                    continue

                # Optional fields
                fs_id = row.get("fsID", "").strip() or None
                start_val = None
                end_val = None
                salience_val = None

                try:
                    if row.get("start"):
                        start_val = float(row["start"])
                except ValueError:
                    pass

                try:
                    if row.get("end"):
                        end_val = float(row["end"])
                except ValueError:
                    pass

                try:
                    if row.get("salience"):
                        salience_val = int(row["salience"])
                except ValueError:
                    pass

                # Build validated AudioFileItem
                audio_item = AudioFileItem(
                    filename=filename,
                    fold=fold_num,
                    wav_path=constructed_wav_path,
                    classID=class_id,
                    class_name=class_label,
                    fsID=fs_id,
                    start=start_val,
                    end=end_val,
                    salience=salience_val,
                    file_exists=True,
                )

                valid_items.append(audio_item)
                fold_counts[fold_num] = fold_counts.get(fold_num, 0) + 1

        summary = {
            "dataset_root": root,
            "csv_path": csv_path,
            "total_csv_rows": total_rows,
            "fold_filter": target_fold if target_fold is not None else "all",
            "valid_files_count": len(valid_items),
            "validation_errors_count": len(validation_errors),
            "fold_distribution": fold_counts,
            "classes": class_mapping,
        }

        return valid_items, validation_errors, summary
