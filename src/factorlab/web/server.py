"""启动本地工作台。"""

from __future__ import annotations

import argparse
from pathlib import Path

from factorlab.web.app import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="打开因子工作台网站")
    parser.add_argument("--out", default="output/demo", help="分析结果目录")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5057)
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    app = create_app(out)
    print(f"工作台  http://{args.host}:{args.port}")
    print(f"数据目录  {out}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
