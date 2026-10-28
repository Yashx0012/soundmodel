"""
Configuration module for the CSV-Based UrbanSound8K FastAPI server.
"""

import os
from pathlib import Path

DEFAULT_WORKSPACE = "/home/ORBITAL/soundmodel"
DEFAULT_DATASET_ROOT = os.path.join(DEFAULT_WORKSPACE, "UrbanSound8K_Processed")
DEFAULT_CSV_FILENAME = "UrbanSound8K.csv"
DEFAULT_BATCH_SIZE = 50
DEFAULT_TIMEOUT_SECONDS = 30.0

SERVER_HOST = os.getenv("HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", "8000"))

# UrbanSound8K C DSP Feature Extraction Tool (`urban8k_tool`) configuration
DEFAULT_C_TOOL_HOST = os.getenv("C_TOOL_HOST", "127.0.0.1")
DEFAULT_C_TOOL_PORT = int(os.getenv("C_TOOL_PORT", "8080"))
DEFAULT_C_TOOL_URL = os.getenv(
    "C_TOOL_URL", f"http://{DEFAULT_C_TOOL_HOST}:{DEFAULT_C_TOOL_PORT}/process"
)
DEFAULT_C_TOOL_HEALTH_URL = os.getenv(
    "C_TOOL_HEALTH_URL", f"http://{DEFAULT_C_TOOL_HOST}:{DEFAULT_C_TOOL_PORT}/health"
)


def resolve_dataset_root(custom_path: str | None = None) -> str:
    """
    Resolves the dataset root directory.
    Checks:
    1. custom_path if provided
    2. DEFAULT_DATASET_ROOT (/home/ORBITAL/soundmodel/UrbanSound8K_Processed)
    3. Fallback candidates: 'UrbanSound8K_Processed', 'sound data', 'UrbanSound8K'
    """
    if custom_path:
        abs_path = os.path.abspath(custom_path)
        if os.path.isdir(abs_path):
            return abs_path
        raise FileNotFoundError(f"Specified dataset root directory does not exist: {custom_path}")

    if os.path.isdir(DEFAULT_DATASET_ROOT):
        return os.path.abspath(DEFAULT_DATASET_ROOT)

    candidates = [
        "UrbanSound8K_Processed",
        "sound data",
        "UrbanSound8K",
        os.path.join(DEFAULT_WORKSPACE, "sound data"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)

    raise FileNotFoundError("Could not locate UrbanSound8K dataset directory.")
