"""Run the website locally with its deployed Modal model workers."""

import argparse
from pathlib import Path

import uvicorn

from nonslop.runtime.local_app import preview_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=8742, help="loopback TCP port (default: 8742)"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="absolute private data directory outside the checkout",
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    uvicorn.run(
        preview_app(arguments.data_root),
        host="127.0.0.1",
        port=arguments.port,
        workers=1,
        proxy_headers=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
