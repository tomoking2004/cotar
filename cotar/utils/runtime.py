from __future__ import annotations

from datetime import datetime

__all__ = ["timestamp", "make_run_id"]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_id(
    name: str | None = None,
    debug: bool = False,
    ts: str | None = None,
) -> str:
    """A run's directory name. Passing one `ts` to several runs files them under a
    common timestamp, which is how the arms of a single experiment stay together.
    """
    parts = [ts or timestamp()]
    if name:
        parts.append(name)
    if debug:
        parts.append("debug")
    return "_".join(parts)
