from __future__ import annotations

from datetime import datetime

__all__ = ["make_run_id", "timestamp"]


def timestamp() -> str:
    """The local wall-clock time, as a run directory reads it.

    Deliberately naive: this names a directory the person at the machine has to
    recognise, so it must agree with the clock they are looking at. Nothing compares
    two of these across machines.
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005


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
