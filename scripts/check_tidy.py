"""Keep the working tree tidy: what `tmp/` may hold, and no `__pycache__` anywhere.

CLAUDE.md fixes what belongs in `tmp/`, which git does not track and nothing else looks at.
The Stop hook in `.claude/settings.json` runs this with `--hook` when a turn ends, so a stray
crop or a dated copy of a working folder is cleared by the session that left it. Run by hand
it prints what is out of place and exits 1.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP  = ROOT / "tmp"

# The only files at the top of tmp/: its index, and work still in progress.
FILES = re.compile(r"README\.md|handoff-\d{4}-\d{2}-\d{2}\.md|run-[\w-]+\.sh")
# A dated folder is a copy of a working folder; versions belong in tmp/professor/ or in git.
DATED = re.compile(r"\d{4}-\d{2}-\d{2}")
SKIP  = {".git", ".venv", "datasets", "runs", "snapshots"}


def caches(folder: Path) -> list[Path]:
    found = []
    for path in folder.iterdir():
        if path.is_dir() and path.name not in SKIP:
            found += [path] if path.name == "__pycache__" else caches(path)
    return found


def problems() -> list[str]:
    found = [f"{path.relative_to(ROOT)}: bytecode cache" for path in caches(ROOT)]
    if TMP.is_dir():
        for path in sorted(TMP.iterdir()):
            if path.is_file() and not FILES.fullmatch(path.name):
                found.append(f"tmp/{path.name}: not a file tmp/ keeps")
            if path.is_dir() and DATED.search(path.name):
                found.append(f"tmp/{path.name}/: a dated copy of a working folder")
    return found


def main() -> int:
    hook = "--hook" in sys.argv
    # A hook that blocks again after its own block would hold the session forever.
    if hook and json.load(sys.stdin).get("stop_hook_active"):
        return 0

    found = problems()
    if not found:
        return 0
    if hook:
        reason = "CLAUDE.md の「tmp/ は置くものを決めて保つ」から外れている：" + "／".join(found)
        print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=False))
        return 0
    print("\n".join(found))
    return 1


if __name__ == "__main__":
    sys.exit(main())
