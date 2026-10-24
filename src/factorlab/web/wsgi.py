"""生产环境入口。本机有最新结果时用 output/demo，线上用随仓库发布的 demo。"""

from __future__ import annotations

import os
from pathlib import Path

from factorlab.web.app import create_app


def default_out() -> Path:
    override = os.environ.get("FACTORLAB_OUT")
    root = Path(__file__).resolve().parents[3]
    if override:
        return Path(override)
    release = root / "data" / "releases" / "cn_pit_v1"
    if (release / "study.json").is_file():
        return release
    local = root / "output" / "demo"
    if (local / "summary.json").is_file():
        return local
    return release


app = create_app(default_out())
