"""一次运行的编号、代码版本和复算容差。编号只取决于配置，不取决于生成时间。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from factorlab.schema import FACTORS

TOLERANCE_ABS = 1e-8
TOLERANCE_NOTE = (
    "同一配置、同一代码版本重跑后，首页的策略 IC、Newey–West t、几何年化、换手，"
    "以及 Brinson 配置、选股、交互的累计值，绝对误差应小于 1e-8。"
    "逐月成本文件里的净收益按同一公式年化后，也应落在这个范围内。"
    "换一台机器时，把容差放宽到 1e-6。"
)


def _package_version() -> str:
    try:
        from factorlab import __version__
    except ImportError:
        return "unknown"
    return str(__version__)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_commit() -> str:
    root = project_root()
    try:
        output = subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        dirty = subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    commit = output.strip() or "unknown"
    if dirty.strip():
        return f"{commit}-dirty"
    return commit


def frozen_factors() -> list[list]:
    return [[spec.name, spec.direction] for spec in FACTORS]


def run_config(
    *,
    mode: str,
    seed: int | None,
    embed_premium: bool | None,
    start: str,
    end: str,
    n_stocks: int,
    ic_min_names: int,
    book_min_names: int,
) -> dict:
    return {
        "mode": mode,
        "seed": seed,
        "embed_premium": embed_premium,
        "start": start,
        "end": end,
        "n_stocks": int(n_stocks),
        "factors": frozen_factors(),
        "n_groups": 5,
        "ic_min_names": int(ic_min_names),
        "book_min_names": int(book_min_names),
        "full_cost_bps": {"commission": 3, "half_spread": 5, "impact": 10, "borrow_monthly_bps": 30},
        "flat_side_bps": [5, 10, 20],
    }


def run_id(config: dict) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def manifest(
    config: dict,
    *,
    generated_at: str,
    brinson: str,
) -> dict:
    return {
        "run_id": run_id(config),
        "generated_at": generated_at,
        "source_commit": source_commit(),
        "package_version": _package_version(),
        "config": config,
        "brinson": brinson,
        "tolerance_abs": TOLERANCE_ABS,
        "tolerance_note": TOLERANCE_NOTE,
        "files": [
            "periods_ic.csv",
            "periods_long_short.csv",
            "periods_weights.csv",
            "periods_brinson.csv",
        ],
    }
