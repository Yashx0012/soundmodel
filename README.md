# CSV-Based FastAPI Server for UrbanSound8K WAV Path Construction & Sequential C DSP Batch Requests

A Python FastAPI server that reads the UrbanSound8K dataset metadata, dynamically constructs and verifies complete WAV audio file paths, partitions them into sequential batches, and submits them strictly one-by-one to the **UrbanSound8K C DSP Feature Extraction Tool (`urban8k_tool`)** or a mock receiver.

> [!IMPORTANT]
> **Strict Scope Boundary**:
> The C DSP Tool (`urban8k_tool`) handles DSP processing, framing, FFT, MFCC, Chroma extraction, and writing 62-D feature columns to `UrbanSound8K.csv`.
> **No changes were made to the C tool codebase**. All modifications and integrations are strictly contained within the Python FastAPI server.

---

## 1. End-to-End System Architecture

```text
       UrbanSound8K Dataset (/home/ORBITAL/soundmodel/UrbanSound8K_Processed)
                                 |
                 +---------------+---------------+
                 |                               |
          UrbanSound8K.csv                  WAV Fold Folders
                 |                          (fold1 to fold10)
                 +---------------+---------------+
                                 |
                                 v
                     DatasetService (dataset_service.py)
                                 |
               - Read CSV: slice_file_name, fold, classID, class
               - Construct full absolute WAV paths
               - Verify file existence on disk (os.path.isfile)
               - Associate metadata without positional indexing
                                 |
                                 v
                      BatchService (batch_service.py)
                                 |
               - Divide validated records into configurable batches
               - Format payload to exact C tool contract:
                 {"csv_path": "...", "files": ["/path/1.wav", "/path/2.wav", ...]}
               - Maintain detailed metadata for transparent inspection
                                 |
                                 v
                    Strict Sequential Batch Submission
               - Strict sequential order: Batch 1 -> Batch 2 -> ... -> Batch N
               - NO multithreading, NO multiprocessing, NO ThreadPoolExecutor
               - Current batch submission must complete before next begins
                                 |
                                 v
                 +---------------+---------------+
                 |                               |
                 v                               v
       urban8k_tool (C Server)          Mock Receiver (FastAPI)
       (http://127.0.0.1:8080/process)  (http://127.0.0.1:8000/api/v1/batches/mock-receiver)
                 |
                 v
       Acoustic Feature Extraction
       (Framing -> FFT -> Spectral -> MFCC -> Chroma -> 62-D Assembler)
                 |
                 v
       UrbanSound8K.csv Updated with 62 Canonical Feature Columns
```

---

## 2. Directory Layout

```text
/home/ORBITAL/soundmodel/
│
├── UrbanSound8K_Processed/
│   ├── UrbanSound8K.csv        # Metadata CSV (8,732 records)
│   └── fold1/ ... fold10/      # 10 Fold directories with WAV files
│
├── app/
│   ├── __init__.py
│   ├── config.py               # Dataset paths & C tool connection defaults
│   ├── main.py                 # FastAPI application & middleware
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic schemas (CToolBatchRequest, BatchPayload, etc.)
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── batches.py          # /prepare, /submit, /mock-receiver, /c-tool/status
│   │   └── dataset.py          # /info and /sample inspection endpoints
│   └── services/
│       ├── __init__.py
│       ├── dataset_service.py   # CSV reader and WAV path validator
│       └── batch_service.py     # Sequential batch divider & HTTP submission loop
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset_service.py # Tests for CSV parsing and path verification
│   ├── test_batch_service.py   # Tests for C tool payload formatting & sequential batches
│   └── test_api.py             # Integration tests for FastAPI endpoints
│
├── run_server.py               # Launcher script for FastAPI server
├── verify_live_batches.py      # Live end-to-end verification script
└── README.md
```

---

## 3. C DSP Tool (`urban8k_tool`) Integration Contract

### Request Payload (`POST /process`)
```json
{
  "csv_path": "/home/ORBITAL/soundmodel/UrbanSound8K_Processed/UrbanSound8K.csv",
  "files": [
    "/home/ORBITAL/soundmodel/UrbanSound8K_Processed/fold1/100032-3-0-0.wav",
    "/home/ORBITAL/soundmodel/UrbanSound8K_Processed/fold1/100263-2-0-117.wav",
    "/home/ORBITAL/soundmodel/UrbanSound8K_Processed/fold1/100263-2-0-121.wav"
  ]
}
```

### Response Payload from C Tool
```json
{
  "status": "ok",
  "processed": 3,
  "failed": 0,
  "csv_path": "/home/ORBITAL/soundmodel/UrbanSound8K_Processed/UrbanSound8K.csv",
  "num_features": 62,
  "message": "Batch processed: 3 succeeded, 0 failed, 3 rows updated in CSV"
}
```

---

## 4. FastAPI Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/batches/prepare` | Prepares batches formatted for the C tool without submitting. |
| `POST` | `/api/v1/batches/submit` | Sequentially submits batches via HTTP POST to `target_url` (or C tool default `http://127.0.0.1:8080/process`). |
| `POST` | `/api/v1/batches/mock-receiver` | Test receiving endpoint mirroring the `urban8k_tool` API contract. |
| `GET` | `/api/v1/batches/c-tool/status` | Probes connectivity and readiness of the `urban8k_tool` server. |
| `GET` | `/api/v1/dataset/info` | Returns dataset root, row count, verified files, and fold breakdown. |
| `GET` | `/api/v1/dataset/sample` | Returns preview records with verified paths. |
| `GET` | `/health` & `/` | Service status checks. |

---

## 5. Usage Examples

### 1. Preparing Batches
```bash
curl -X POST http://127.0.0.1:8000/api/v1/batches/prepare \
  -H "Content-Type: application/json" \
  -d '{
    "fold": 1,
    "batch_size": 50,
    "max_batches": 2
  }'
```

### 2. Submitting Sequentially to the C DSP Tool (`urban8k_tool`)
```bash
curl -X POST http://127.0.0.1:8000/api/v1/batches/submit \
  -H "Content-Type: application/json" \
  -d '{
    "fold": 1,
    "batch_size": 50,
    "use_c_tool": true,
    "target_url": "http://127.0.0.1:8080/process"
  }'
```

### 3. Checking C DSP Tool Connectivity
```bash
curl http://127.0.0.1:8000/api/v1/batches/c-tool/status
```

---

## 6. Testing

### Run Pytest Suite
```bash
python3 -m pytest -v
```

### Run Live End-to-End Verification
```bash
# Terminal 1: Start FastAPI server
python3 run_server.py --port 8000

# Terminal 2: Run verification script
python3 verify_live_batches.py
```
This tests both mock receiver sequential submission and direct batch submission against the compiled `urban8k_tool` binary!
