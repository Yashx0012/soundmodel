"""
Main FastAPI application entry point for UrbanSound8K WAV path construction
and sequential batch API request preparation & submission.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import DEFAULT_DATASET_ROOT, resolve_dataset_root
from app.routers.batches import router as batches_router
from app.routers.dataset import router as dataset_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("soundmodel-api")

app = FastAPI(
    title="UrbanSound8K CSV Batch Request Server",
    description="""
    CSV-Based FastAPI Server for WAV Path Construction and Sequential Batch API Requests.
    
    Capabilities:
    - Reads UrbanSound8K CSV metadata and validates row integrity.
    - Dynamically constructs verified WAV file paths without positional indexing.
    - Associates CSV metadata (labels, classIDs, folds) with each file.
    - Partitions files into configurable batches.
    - Prepares structured batch API payloads.
    - Sequentially submits batches to destination endpoints without parallel threads or processes.
    """,
    version="1.0.0",
)

# Enable CORS for open API usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(batches_router)
app.include_router(dataset_router)


@app.get("/", tags=["Health"])
def root_status():
    """
    Root status check.
    """
    return {
        "service": "UrbanSound8K CSV Batch Request Server",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check verifying service status and dataset root availability.
    """
    dataset_found = False
    try:
        root = resolve_dataset_root()
        dataset_found = bool(root)
    except Exception:
        root = None

    return {
        "status": "healthy" if dataset_found else "degraded",
        "dataset_root_resolved": root,
        "dataset_available": dataset_found,
    }


if __name__ == "__main__":
    import uvicorn
    from app.config import SERVER_HOST, SERVER_PORT

    uvicorn.run("app.main:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)
