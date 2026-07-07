"""
Shared runtime helpers — one source of truth for config loading, device selection,
and opening the vector store. Imported by both the data_pipeline scripts and the
rag/ modules, so these details are never duplicated or written two different ways.

See markdown/CODING_STANDARDS.md.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from qdrant_client import QdrantClient

CONFIG_PATH = Path("configs/config.yaml")


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Parse config.yaml. Always safe_load (never load) — no code execution from YAML."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def get_device() -> str:
    """'cuda' if a GPU is available, else 'cpu'. Single source for device selection."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_qdrant_client(vs: dict) -> QdrantClient:
    """Open the embedded Qdrant store from the vector_store config.

    Embedded (on-disk) mode holds an exclusive folder lock for the client's lifetime,
    so the caller is responsible for close() — use `with` where possible.
    """
    return QdrantClient(path=vs["path"])
