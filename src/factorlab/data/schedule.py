"""按 config/schedule.json 判断今天是不是更新日。导入时不拉数据。"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_schedule(path: Path | None = None) -> dict:
    source = path or project_root() / "config" / "schedule.json"
    return json.loads(source.read_text(encoding="utf-8"))


def is_due(today: date, spec: dict) -> bool:
    names = {str(item).lower() for item in spec.get("weekdays") or []}
    return _WEEKDAYS[today.weekday()] in names


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    spec = load_schedule()
    today = date.today()
    due = is_due(today, spec)
    if "--run" not in args:
        if due:
            print("今天在更新日。加上 --run 才会续拉行情。这一步不改研究结论。")
        else:
            print("今天不在更新日，没有拉取。")
        return 0
    cache = os.environ.get("FACTORLAB_CACHE", "").replace("\\", "/").lower()
    if "csi500" in cache or "csi800" in cache:
        print("当前环境指向中证500或800的缓存，定时更新不会往里写。")
        return 2
    if not due and "--force" not in args:
        print("今天不在更新日。要忽略日历请加 --force。")
        return 0
    stage = str(spec.get("stage") or "prices")
    os.environ["FACTORLAB_CSI300_STAGE"] = stage
    from factorlab.data.csi300_pit import main as fetch_main

    fetch_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
