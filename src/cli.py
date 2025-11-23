"""Convenience entrypoints for uv/project scripts."""
from __future__ import annotations

import uvicorn


def run_api() -> None:
    """Run the FastAPI application with sensible defaults."""
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)


def main_api() -> None:  # backward-compat alias if needed
    run_api()


if __name__ == "__main__":
    run_api()
