"""发布结果必须来自干净的 git 工作区。"""

from __future__ import annotations

import json
from pathlib import Path


def dirty_run_files(root: Path) -> list[Path]:
    hits = []
    releases = root / "data" / "releases"
    if not releases.is_dir():
        return hits
    for path in sorted(releases.rglob("*.json")):
        text = path.read_text(encoding="utf-8")
        if "source_commit" in text and "-dirty" in text:
            hits.append(path)
    return hits


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    hits = dirty_run_files(root)
    if hits:
        lines = "\n".join(str(path.relative_to(root)) for path in hits)
        raise SystemExit(f"研究结果带有 -dirty，不能发布：\n{lines}")
    print("published runs are clean")


if __name__ == "__main__":
    main()
