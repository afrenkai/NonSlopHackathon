import os
import secrets
import tempfile
from functools import partial
from pathlib import Path

import modal
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import SecretStr
from pytesseract import TesseractNotFoundError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from nonslop.runtime.archivist import ModalArchivistGateway
from nonslop.runtime.gateway import ModalGenerationGateway, ModalReconstructionGateway
from nonslop.web.app import create_app
from nonslop.web.schemas import WebSettings


def private_root(directory: Path | None = None) -> Path:
    repository = Path(__file__).resolve().parents[3]
    if directory is None:
        base = (
            Path(os.environ["LOCALAPPDATA"])
            if os.name == "nt"
            else Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        )
        directory = base / "NonSlop" / "cpu-preview"
    if not directory.is_absolute():
        raise ValueError("The local user data directory must be an absolute path.")
    root = directory.resolve()
    if root.is_relative_to(repository.parent) or repository.parent.is_relative_to(root):
        raise ValueError(
            "Preview data must be outside the checkout and its served parent directory."
        )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def session_secret(root: Path) -> SecretStr:
    path = root / "session.key"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return SecretStr(path.read_text(encoding="ascii"))
    secret = secrets.token_urlsafe(48)
    with os.fdopen(descriptor, "w", encoding="ascii") as file:
        file.write(secret)
    return SecretStr(secret)


def no_volume_sync() -> None:
    pass


def download_volume_artifact(
    volume: modal.Volume, relative_path: Path, destination: Path
) -> None:
    with destination.open("wb") as file:
        for chunk in volume.read_file(relative_path.as_posix()):
            file.write(chunk)


def preview_app(data_root: Path | None = None) -> FastAPI:
    repository = Path(__file__).resolve().parents[3]
    load_dotenv(repository / ".env", override=False)
    root = private_root(data_root)
    temporary = root / "tmp"
    temporary.mkdir(mode=0o700, exist_ok=True)
    tempfile.tempdir = str(temporary)
    os.environ.update(dict.fromkeys(("TMPDIR", "TEMP", "TMP"), str(temporary)))
    settings = WebSettings(
        session_secret=session_secret(root),
        api_key=SecretStr(os.environ.get("APP_API_KEY") or secrets.token_urlsafe(32)),
        password=SecretStr(os.environ.get("APP_PASSWORD") or secrets.token_urlsafe(32)),
        secure_cookies=False,
        assets_root=repository / "src/nonslop/web",
        data_root=root / "data",
        output_root=root / "outputs",
        serving_root=root / "serving",
        world_root=root / "worlds",
        world_serving_root=root / "world-serving",
    )
    app = create_app(
        settings,
        ModalGenerationGateway(
            modal.Function.from_name("nonslop-streets", "prepare_visit")
        ),
        ModalReconstructionGateway(
            modal.Cls.from_name("nonslop-world", "WorldReconstructor")
        ),
        ModalArchivistGateway(
            modal.Cls.from_name("nonslop-inference", "MemoryArchivist")
        ),
        no_volume_sync,
        no_volume_sync,
        no_volume_sync,
        download_video=partial(
            download_volume_artifact, modal.Volume.from_name("nonslop-videos")
        ),
        download_world=partial(
            download_volume_artifact, modal.Volume.from_name("nonslop-worlds")
        ),
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1"])

    @app.exception_handler(TesseractNotFoundError)
    async def no_ocr(
        request: Request, error: TesseractNotFoundError
    ) -> PlainTextResponse:
        return PlainTextResponse(
            "Scan OCR needs local Tesseract. Paste the text or upload a .txt or .md file instead.",
            status_code=422,
        )

    return app
