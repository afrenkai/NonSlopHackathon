from pathlib import Path

import modal

from nonslop.web.demo_visits import (
    DEMO_VISITS,
    LOCAL_RESEARCH_ROOT,
    MODAL_RESEARCH_ROOT,
)


def build_web_image() -> modal.Image:
    web_root = Path(__file__).parents[1] / "web"
    image = (
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
    for demo in DEMO_VISITS.values():
        for filename in (demo.research_file, demo.observation_file):
            image = image.add_local_file(
                LOCAL_RESEARCH_ROOT / filename,
                str(MODAL_RESEARCH_ROOT / filename),
                copy=True,
            )
    return image
