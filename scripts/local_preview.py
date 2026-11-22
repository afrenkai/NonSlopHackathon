"""Launch the real web app with private local stores and no remote workers."""

import argparse
import os
import secrets
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Never
from uuid import uuid4

import uvicorn
from anyio import to_thread
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import SecretStr
from pytesseract import TesseractNotFoundError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from nonslop.generation.prompts import build_scene_prompt
from nonslop.generation.schemas import JobRecord, SceneRequest
from nonslop.generation.service import SceneService
from nonslop.web import app as web
from nonslop.web.schemas import WebSettings

REPO = Path(__file__).resolve().parents[1]


class OfflineWorker:
    """Never return a call ID or an artifact for work that did not run."""

    def submit(self, *args: object, **kwargs: object) -> Never:
        raise RuntimeError("Model workers are offline in the local CPU preview.")

    def result(self, call_id: str) -> Never:
        raise RuntimeError("Model workers are offline in the local CPU preview.")


class PreviewScenes(SceneService):
    async def submit(
        self,
        scene: SceneRequest,
        reference_images: list[bytes],
        owner_id: str,
        reference_names: list[str],
    ) -> JobRecord:
        # The normal submit path queries Internet Archive *before* the gateway.
        # Keep the validated evidence, but do not retrieve, enqueue, or poll.
        job = JobRecord(
            job_id=str(uuid4()),
            owner_id=owner_id,
            created_at=datetime.now(UTC),
            status="failed",
            error="Your sources are saved. Video generation and 3D reconstruction are offline in this CPU preview.",
            scene=scene.model_copy(update={"use_web": False}),
            render_prompt=build_scene_prompt(scene, [], bool(reference_images)),
            notices=(
                ["Web search is disabled in the local preview."]
                if scene.use_web
                else []
            ),
            has_reference=bool(reference_images),
            reference_count=len(reference_images),
            reference_names=reference_names,
        )
        await to_thread.run_sync(self.store.save, job, reference_images)
        return job


def private_root() -> Path:
    base = (
        Path(os.environ["LOCALAPPDATA"])
        if os.name == "nt"
        else Path.home() / ".local/share"
    )
    if not base.is_absolute():
        raise ValueError("The local user data directory must be an absolute path.")
    root = (base / "NonSlop" / "cpu-preview").resolve()
    if root.is_relative_to(REPO.parent) or REPO.parent.is_relative_to(root):
        raise ValueError(
            "Preview data must be outside the checkout and its served parent directory."
        )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def session_secret(root: Path) -> SecretStr:
    # Keep returning browsers in the same workspace after a server restart.
    path = root / "session.key"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return SecretStr(path.read_text(encoding="ascii"))
    secret = secrets.token_urlsafe(48)
    with os.fdopen(fd, "w", encoding="ascii") as file:
        file.write(secret)
    return SecretStr(secret)


def no_volume_sync() -> None:
    pass


def preview_app() -> FastAPI:
    root = private_root()
    # Spools, OCR and ffprobe inputs should not inherit a served temp directory.
    tmp = root / "tmp"
    tmp.mkdir(mode=0o700, exist_ok=True)
    tempfile.tempdir = str(tmp)
    os.environ.update(dict.fromkeys(("TMPDIR", "TEMP", "TMP"), str(tmp)))
    settings = WebSettings(
        session_secret=session_secret(root),
        api_key=SecretStr(secrets.token_urlsafe(32)),
        password=SecretStr(secrets.token_urlsafe(32)),
        secure_cookies=False,  # This launcher binds to IPv4 loopback over HTTP only.
        assets_root=REPO / "src/nonslop/web",
        data_root=root / "data",
        output_root=root / "outputs",
        serving_root=root / "serving",
        world_root=root / "worlds",
        world_serving_root=root / "world-serving",
    )
    worker = OfflineWorker()
    app = web.create_app(
        settings,
        worker,
        worker,
        worker,
        no_volume_sync,
        no_volume_sync,
        no_volume_sync,
        scene_service=PreviewScenes,
    )
    app.state.local_preview = True
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=8742, help="loopback TCP port (default: 8742)"
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    uvicorn.run(
        preview_app(),
        host="127.0.0.1",
        port=args.port,
        workers=1,
        proxy_headers=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
