#!/usr/bin/env python3
"""
trim_audio.py

Task: WAV Audio Duration Checking and Conditional Trimming for UrbanSound8K dataset.

Requirements:
1. Automatically scan all 10 fold directories (fold1 to fold10).
2. Read every WAV file and calculate its duration.
3. Apply conditions:
   - Greater than 1 second: Trim to exactly 1 second.
   - Equal to 1 second: Keep unchanged.
   - Less than 1 second: Pad with zeros (silence) to exactly 1 second.
4. Preserve original filenames.
5. Preserve original fold directory structure.
6. Preserve original sample rate and audio channels.
7. Do not modify or overwrite original dataset.
8. Save all processed files in a separate directory named 'UrbanSound8K_Processed'.
9. Keep 'UrbanSound8K.csv' unchanged.
10. Print the filename and original duration of each trimmed file.
11. Final summary with required counts.
12. Comprehensive output validation.
"""

import os
import shutil
import sys
import soundfile as sf
import numpy as np


def resolve_dataset_root(custom_path=None):
    """
    Locates the UrbanSound8K dataset directory.
    Checks user-specified path, 'UrbanSound8K', 'sound data', and current directory.
    """
    if custom_path and os.path.isdir(custom_path):
        return custom_path

    candidates = ["UrbanSound8K", "sound data", "."]
    for candidate in candidates:
        if os.path.isdir(candidate):
            # Verify if fold1 exists inside
            if os.path.isdir(os.path.join(candidate, "fold1")):
                return candidate

    raise FileNotFoundError(
        "Could not locate UrbanSound8K dataset folder containing fold1-fold10."
    )


def get_audio_info(file_path):
    """
    Retrieves audio metadata using soundfile.
    Returns (duration_seconds, samplerate, channels, frames, subtype).
    """
    info = sf.info(file_path)
    duration = info.frames / info.samplerate
    return duration, info.samplerate, info.channels, info.frames, info.subtype


def trim_and_save_audio(src_path, dst_path, samplerate, channels, original_subtype):
    """
    Trims the audio file to exactly 1 second (samplerate frames) and writes to dst_path.
    Preserves original sample rate and channels.
    """
    # Read exactly 1 second of audio (samplerate frames)
    data, sr = sf.read(src_path, frames=samplerate, dtype="float32")

    # If file had more frames, ensure we take exactly samplerate frames
    if len(data) > samplerate:
        data = data[:samplerate]

    # Target subtype: preserve original if compatible, otherwise fallback to PCM_16
    # Note: ADPCM subtypes have fixed block sizes that prevent arbitrary frame counts
    target_subtype = original_subtype
    if target_subtype in ("MS_ADPCM", "IMA_ADPCM"):
        target_subtype = "PCM_16"

    try:
        sf.write(dst_path, data, sr, subtype=target_subtype)
        # Verify written frame count matches exactly samplerate
        written_info = sf.info(dst_path)
        if written_info.frames != samplerate:
            # Fallback to standard PCM_16 if subtype caused frame alignment discrepancy
            sf.write(dst_path, data, sr, subtype="PCM_16")
    except Exception:
        # Fallback to PCM_16 on any subtype-specific encoding failure
        sf.write(dst_path, data, sr, subtype="PCM_16")


def pad_and_save_audio(src_path, dst_path, samplerate, channels, original_subtype):
    """
    Pads the audio file with zeros (silence) to reach exactly 1 second (samplerate samples)
    and writes to dst_path. Preserves original sample rate and number of channels.
    """
    data, sr = sf.read(src_path, dtype="float32")
    current_samples = len(data)
    missing_samples = samplerate - current_samples

    # Append zeros (silence) to the end of the audio
    if missing_samples > 0:
        if data.ndim == 1:
            silence = np.zeros(missing_samples, dtype=data.dtype)
            padded_data = np.concatenate((data, silence))
        else:
            silence = np.zeros((missing_samples, data.shape[1]), dtype=data.dtype)
            padded_data = np.concatenate((data, silence), axis=0)
    else:
        padded_data = data

    target_subtype = original_subtype
    if target_subtype in ("MS_ADPCM", "IMA_ADPCM"):
        target_subtype = "PCM_16"

    try:
        sf.write(dst_path, padded_data, sr, subtype=target_subtype)
        written_info = sf.info(dst_path)
        if written_info.frames != samplerate:
            sf.write(dst_path, padded_data, sr, subtype="PCM_16")
    except Exception:
        sf.write(dst_path, padded_data, sr, subtype="PCM_16")


def process_dataset(dataset_root, output_root):
    """
    Scans all 10 folds, conditionally trims audio files > 1 second,
    copies unchanged files == 1 second, and zero-pads files < 1 second.
    """
    os.makedirs(output_root, exist_ok=True)

    # Copy UrbanSound8K.csv if present, preserving original unchanged
    csv_src = os.path.join(dataset_root, "UrbanSound8K.csv")
    csv_dst = os.path.join(output_root, "UrbanSound8K.csv")
    if os.path.isfile(csv_src):
        shutil.copy2(csv_src, csv_dst)

    stats = {
        "total_scanned": 0,
        "greater_than_1s": 0,
        "equal_to_1s": 0,
        "less_than_1s": 0,
        "successfully_processed": 0,
        "processing_errors": 0,
    }

    trimmed_files_record = []

    print(f"\n--- Starting UrbanSound8K Audio Processing ---")
    print(f"Source Dataset: {os.path.abspath(dataset_root)}")
    print(f"Output Directory: {os.path.abspath(output_root)}\n")

    for fold_num in range(1, 11):
        fold_name = f"fold{fold_num}"
        src_fold_dir = os.path.join(dataset_root, fold_name)
        dst_fold_dir = os.path.join(output_root, fold_name)

        if not os.path.isdir(src_fold_dir):
            print(f"Warning: Directory {src_fold_dir} not found. Skipping.")
            continue

        os.makedirs(dst_fold_dir, exist_ok=True)
        wav_files = sorted(
            [f for f in os.listdir(src_fold_dir) if f.lower().endswith(".wav")]
        )

        for filename in wav_files:
            stats["total_scanned"] += 1
            src_file_path = os.path.join(src_fold_dir, filename)
            dst_file_path = os.path.join(dst_fold_dir, filename)

            try:
                duration, samplerate, channels, frames, subtype = get_audio_info(
                    src_file_path
                )

                if frames > samplerate:
                    # Greater than 1 second: Trim to exactly 1 second
                    stats["greater_than_1s"] += 1
                    # Requirement 10: Print filename and original duration
                    print(
                        f"Trimmed: {filename} (fold: {fold_name}) | Original Duration: {duration:.4f}s"
                    )
                    trimmed_files_record.append((filename, fold_name, duration))

                    trim_and_save_audio(
                        src_file_path, dst_file_path, samplerate, channels, subtype
                    )
                    stats["successfully_processed"] += 1

                elif frames == samplerate:
                    # Equal to 1 second: Keep unchanged
                    stats["equal_to_1s"] += 1
                    shutil.copy2(src_file_path, dst_file_path)
                    stats["successfully_processed"] += 1

                else:
                    # Less than 1 second: Calculate missing samples and append zeros
                    stats["less_than_1s"] += 1
                    pad_and_save_audio(
                        src_file_path, dst_file_path, samplerate, channels, subtype
                    )
                    stats["successfully_processed"] += 1

            except Exception as e:
                stats["processing_errors"] += 1
                print(f"Error processing {src_file_path}: {e}", file=sys.stderr)

    return stats, trimmed_files_record


def display_summary(stats):
    """
    Displays the final summary of processing results.
    """
    print("\n" + "=" * 55)
    print("                FINAL PROCESSING SUMMARY")
    print("=" * 55)
    print(f"Total WAV files scanned:            {stats['total_scanned']}")
    print(f"Number of files greater than 1s:    {stats['greater_than_1s']}")
    print(f"Number of files equal to 1s:        {stats['equal_to_1s']}")
    print(f"Number of files shorter than 1s:    {stats['less_than_1s']}")
    print(f"Number of successfully processed:   {stats['successfully_processed']}")
    print(f"Number of processing errors:        {stats['processing_errors']}")
    print("=" * 55 + "\n")


def validate_results(dataset_root, output_root):
    """
    Validates output according to all required criteria:
    - Every original WAV file has a corresponding processed file.
    - No processed WAV file is longer than 1 second.
    - Files originally shorter than or equal to 1 second remain unchanged.
    - Every trimmed file has exactly 1 second of audio.
    - Original filenames and folder structures are preserved.
    """
    print("=" * 55)
    print("              OUTPUT VALIDATION REPORT")
    print("=" * 55)

    all_tests_passed = True
    missing_files = []
    longer_than_1s = []
    untouched_mismatches = []
    trimmed_not_exact_1s = []
    padded_not_exact_1s = []
    structure_mismatches = []

    for fold_num in range(1, 11):
        fold_name = f"fold{fold_num}"
        src_fold = os.path.join(dataset_root, fold_name)
        dst_fold = os.path.join(output_root, fold_name)

        if not os.path.isdir(dst_fold):
            structure_mismatches.append(f"Missing folder {fold_name}")
            continue

        src_files = sorted(
            [f for f in os.listdir(src_fold) if f.lower().endswith(".wav")]
        )
        dst_files = sorted(
            [f for f in os.listdir(dst_fold) if f.lower().endswith(".wav")]
        )

        if src_files != dst_files:
            diff = set(src_files).symmetric_difference(set(dst_files))
            structure_mismatches.append(
                f"{fold_name} file mismatch (difference: {len(diff)} files)"
            )

        for filename in src_files:
            src_path = os.path.join(src_fold, filename)
            dst_path = os.path.join(dst_fold, filename)

            if not os.path.isfile(dst_path):
                missing_files.append(os.path.join(fold_name, filename))
                continue

            # Original file info
            src_info = sf.info(src_path)
            # Processed file info
            dst_info = sf.info(dst_path)

            # Check 1: Sample rate and channels preserved
            if dst_info.samplerate != src_info.samplerate:
                all_tests_passed = False
                print(
                    f"Sample rate mismatch for {filename}: {src_info.samplerate} vs {dst_info.samplerate}"
                )
            if dst_info.channels != src_info.channels:
                all_tests_passed = False
                print(
                    f"Channels mismatch for {filename}: {src_info.channels} vs {dst_info.channels}"
                )

            # Check 2: No processed WAV file is longer than 1 second
            if dst_info.frames > dst_info.samplerate:
                longer_than_1s.append((filename, dst_info.frames, dst_info.samplerate))

            # Check 3: Files originally == 1s remain unchanged
            if src_info.frames == src_info.samplerate:
                if (
                    dst_info.frames != src_info.frames
                    or os.path.getsize(dst_path) != os.path.getsize(src_path)
                ):
                    untouched_mismatches.append(filename)

            # Check 4: Every trimmed file has exactly 1 second of audio
            if src_info.frames > src_info.samplerate:
                if dst_info.frames != dst_info.samplerate:
                    trimmed_not_exact_1s.append(
                        (filename, dst_info.frames, dst_info.samplerate)
                    )

            # Check 5: Every padded file has exactly 1 second of audio
            if src_info.frames < src_info.samplerate:
                if dst_info.frames != dst_info.samplerate:
                    padded_not_exact_1s.append(
                        (filename, dst_info.frames, dst_info.samplerate)
                    )

    # Report results
    test_1 = len(missing_files) == 0
    test_2 = len(longer_than_1s) == 0
    test_3 = len(untouched_mismatches) == 0
    test_4 = len(trimmed_not_exact_1s) == 0
    test_5 = len(padded_not_exact_1s) == 0
    test_6 = len(structure_mismatches) == 0

    print(
        f"1. Every original WAV has processed file:    {'PASSED' if test_1 else 'FAILED (' + str(len(missing_files)) + ' missing)'}"
    )
    print(
        f"2. No processed WAV longer than 1s:          {'PASSED' if test_2 else 'FAILED (' + str(len(longer_than_1s)) + ' violations)'}"
    )
    print(
        f"3. Files == 1s remain unchanged:             {'PASSED' if test_3 else 'FAILED (' + str(len(untouched_mismatches)) + ' modified)'}"
    )
    print(
        f"4. Trimmed files have exactly 1s audio:      {'PASSED' if test_4 else 'FAILED (' + str(len(trimmed_not_exact_1s)) + ' incorrect)'}"
    )
    print(
        f"5. Padded files have exactly 1s audio:       {'PASSED' if test_5 else 'FAILED (' + str(len(padded_not_exact_1s)) + ' incorrect)'}"
    )
    print(
        f"6. Filenames and folder structures preserved:{'PASSED' if test_6 else 'FAILED (' + str(len(structure_mismatches)) + ' errors)'}"
    )

    all_passed = test_1 and test_2 and test_3 and test_4 and test_5 and test_6 and all_tests_passed
    print("=" * 55)
    if all_passed:
        print(">>> ALL VALIDATION CHECKS PASSED SUCCESSFULLY! <<<")
    else:
        print(">>> VALIDATION DETECTED ISSUES! <<<")
    print("=" * 55 + "\n")

    return all_passed


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="UrbanSound8K WAV Duration Checking and Conditional Trimming"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to UrbanSound8K dataset directory (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="UrbanSound8K_Processed",
        help="Path to output processed dataset directory (default: UrbanSound8K_Processed)",
    )
    args = parser.parse_args()

    # 1. Resolve dataset directory
    dataset_root = resolve_dataset_root(args.input)
    output_root = args.output

    # 2. Process dataset
    stats, _ = process_dataset(dataset_root, output_root)

    # 3. Final summary
    display_summary(stats)

    # 4. Validation
    success = validate_results(dataset_root, output_root)

    if not success or stats["processing_errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
