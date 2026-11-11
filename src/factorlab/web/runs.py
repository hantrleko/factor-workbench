"""列出已有 run.json，并对比两次运行的配置字段。"""

from __future__ import annotations

import json
from pathlib import Path


def list_runs(project: Path) -> list[dict]:
    root = project.resolve()
    found: list[dict] = []
    releases = root / "data" / "releases"
    if releases.is_dir():
        for path in sorted(releases.glob("*/run.json")):
            card = _card(root, path, path.parent.name)
            if card:
                found.append(card)
    archive = root / "data" / "runs"
    if archive.is_dir():
        for path in sorted(archive.glob("*.json")):
            card = _card(root, path, "归档")
            if card:
                found.append(card)
    kept: dict[str, dict] = {}
    for card in found:
        token = card["run_id"] or card["key"]
        current = kept.get(token)
        if current is None or (current["place"] == "归档" and card["place"] != "归档"):
            kept[token] = card
    rows = list(kept.values())
    rows.sort(key=lambda item: item["generated_at"], reverse=True)
    return rows


def diff_runs(left: dict, right: dict) -> list[dict]:
    fields = _flat(left)
    other = _flat(right)
    rows = []
    for key, label in _FIELDS:
        if str(fields.get(key, "")) != str(other.get(key, "")):
            rows.append({"field": label, "left": fields.get(key, ""), "right": other.get(key, "")})
    return rows


_FIELDS = (
    ("run_id", "运行编号"),
    ("generated_at", "生成时间"),
    ("source_commit", "代码版本"),
    ("mode", "样本模式"),
    ("start", "起始"),
    ("end", "结束"),
    ("n_stocks", "股票数"),
    ("n_groups", "分组"),
    ("ic_min_names", "截面最少股票"),
    ("book_min_names", "组合最少股票"),
    ("factors", "因子方向"),
    ("full_cost", "完整成本假设"),
)


def _card(project: Path, path: Path, place: str) -> dict | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        key = path.resolve().relative_to(project).as_posix()
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    config = record.get("config") if isinstance(record.get("config"), dict) else {}
    return {
        "key": key,
        "place": place,
        "run_id": str(record.get("run_id") or ""),
        "generated_at": str(record.get("generated_at") or ""),
        "source_commit": str(record.get("source_commit") or ""),
        "config": config,
        "mode": str(config.get("mode") or ""),
        "start": str(config.get("start") or ""),
        "end": str(config.get("end") or ""),
        "n_stocks": config.get("n_stocks", ""),
    }


def _flat(card: dict) -> dict:
    config = card.get("config") or {}
    cost = config.get("full_cost_bps") or {}
    cost_text = " ".join(
        f"{key} {cost[key]}" for key in ("commission", "half_spread", "impact", "borrow") if key in cost
    )
    return {
        "run_id": card.get("run_id", ""),
        "generated_at": card.get("generated_at", ""),
        "source_commit": card.get("source_commit", ""),
        "mode": config.get("mode", ""),
        "start": config.get("start", ""),
        "end": config.get("end", ""),
        "n_stocks": config.get("n_stocks", ""),
        "n_groups": config.get("n_groups", ""),
        "ic_min_names": config.get("ic_min_names", ""),
        "book_min_names": config.get("book_min_names", ""),
        "factors": _factor_text(config.get("factors")),
        "full_cost": cost_text,
    }


def _factor_text(factors) -> str:
    parts = []
    for item in factors or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            parts.append(f"{item[0]} {int(item[1]):+d}")
    return ", ".join(parts)
