#!/usr/bin/env python3
"""
Convenience launcher for the UrbanSound8K CSV Batch Request FastAPI server.
"""

import argparse
import sys
import uvicorn
from app.config import SERVER_HOST, SERVER_PORT


def main():
    parser = argparse.ArgumentParser(
        description="Run UrbanSound8K CSV Batch Request FastAPI Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=SERVER_HOST,
        help=f"Host interface to bind (default: {SERVER_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=SERVER_PORT,
        help=f"Port number to bind (default: {SERVER_PORT})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code change",
    )
    args = parser.parse_args()

    print(f"Starting UrbanSound8K FastAPI Server on http://{args.host}:{args.port}")
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
