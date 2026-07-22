"""Shared paths for objective metric calculation."""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR.parent / "data"
OUTPUT_DIR = ROOT_DIR / "output"

INDEX_CSV = DATA_DIR / "waymo_data index.csv"

LOCAL_LIGHTWEIGHT_CACHE_ROOT = DATA_DIR / "waymo_lightweight_cache"

def _cache_path(folder: str, env_var: str) -> str:
    specific_path = os.environ.get(env_var)
    if specific_path:
        return specific_path

    cache_root = os.environ.get("WAYMO_CACHE_ROOT")
    if cache_root:
        return str(Path(cache_root).expanduser() / folder)

    return str(LOCAL_LIGHTWEIGHT_CACHE_ROOT / folder)


FOLDER_CACHE_MAP = {
    "waymo_0-299": _cache_path("waymo_0-299", "WAYMO_CACHE_0_299"),
    "waymo_300-499": _cache_path("waymo_300-499", "WAYMO_CACHE_300_499"),
    "waymo_500-799": _cache_path("waymo_500-799", "WAYMO_CACHE_500_799"),
    "waymo_800-999": _cache_path("waymo_800-999", "WAYMO_CACHE_800_999"),
}
