"""JSON in and out, as UTF-8."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ujson as json

__all__ = ["load_json", "save_json"]


def load_json(path: str | Path) -> Any:
    """The JSON at `path`; a missing file is an error that names it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 4) -> None:
    """Write `data` to `path` as indented JSON, creating its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
