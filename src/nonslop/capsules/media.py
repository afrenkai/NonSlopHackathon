import subprocess
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from anyio import to_thread
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

from nonslop.capsules.schemas import CapsuleUpload, EntryKind, MediaProbe
from nonslop.models.inference_schemas import MediaType
from nonslop.web.uploads import normalize_image, upload_bytes


def probe_recording(content: bytes, suffix: str) -> MediaProbe:
    formats = {
        "wav": "wav",
        "mp3": "mp3",
        "m4a": "mov",
        "ogg": "ogg",
        "flac": "flac",
        "mp4": "mov",
        "webm": "matroska",
        "mov": "mov",
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / f"recording.{suffix}"
        path.write_bytes(content)
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "json",
                "-protocol_whitelist",
                "file",
                "-f",
                formats[suffix],
                "-i",
                str(path),
            ],
            capture_output=True,
            check=True,
            timeout=15,
        )
    return MediaProbe.model_validate_json(result.stdout)


async def read_memory_media(upload: UploadFile, limit: int) -> CapsuleUpload:
    content = await upload_bytes(upload, limit)
    if not content:
        raise HTTPException(400, "The uploaded file is empty.")
    if upload.content_type in ("image/png", "image/jpeg", "image/webp"):
        normalize_image(content)
        with Image.open(BytesIO(content)) as image:
            if image.format == "PNG":
                return CapsuleUpload(
                    content=content, kind="image", media_type="image/png", suffix="png"
                )
            if image.format == "JPEG":
                return CapsuleUpload(
                    content=content, kind="image", media_type="image/jpeg", suffix="jpg"
                )
        return CapsuleUpload(
            content=content, kind="image", media_type="image/webp", suffix="webp"
        )
    supported: dict[str, tuple[EntryKind, MediaType, str]] = {
        "audio/wav": ("audio", "audio/wav", "wav"),
        "audio/x-wav": ("audio", "audio/wav", "wav"),
        "audio/mpeg": ("audio", "audio/mpeg", "mp3"),
        "audio/mp4": ("audio", "audio/mp4", "m4a"),
        "audio/ogg": ("audio", "audio/ogg", "ogg"),
        "audio/flac": ("audio", "audio/flac", "flac"),
        "video/mp4": ("video", "video/mp4", "mp4"),
        "video/webm": ("video", "video/webm", "webm"),
        "video/quicktime": ("video", "video/quicktime", "mov"),
    }
    if upload.content_type not in supported:
        raise HTTPException(
            400,
            "Use a PNG/JPEG/WebP image, WAV/MP3/M4A/OGG/FLAC audio, or MP4/WebM/MOV video.",
        )
    kind, media_type, suffix = supported[upload.content_type]
    try:
        streams = await to_thread.run_sync(probe_recording, content, suffix)
        if not any(stream.codec_type == kind for stream in streams.streams):
            raise HTTPException(
                400, "The file does not contain the selected media type."
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise HTTPException(400, "The uploaded recording could not be read.") from error
    return CapsuleUpload(
        content=content, kind=kind, media_type=media_type, suffix=suffix
    )
