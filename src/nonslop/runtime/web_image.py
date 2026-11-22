from pathlib import Path

import modal


def build_web_image() -> modal.Image:
    web_root = Path(__file__).parents[1] / "web"
    return (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("tesseract-ocr", "ffmpeg")
        .uv_pip_install(
            "fastapi[standard]",
            "duckdb",
            "httpx",
            "itsdangerous",
            "jinja2",
            "pillow",
            "pytesseract",
            "python-multipart",
            "python-dotenv",
        )
        .add_local_dir(web_root / "templates", "/opt/nonslop-web/templates", copy=True)
        .add_local_dir(web_root / "static", "/opt/nonslop-web/static", copy=True)
    )
