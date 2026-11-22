from io import BytesIO

import pytesseract
from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile

from nonslop.web.schemas import WebSettings


async def upload_bytes(upload: UploadFile, limit: int) -> bytes:
    payload = await upload.read(limit + 1)
    await upload.close()
    if len(payload) > limit:
        raise HTTPException(
            413, f"Each uploaded file must be {limit // (1024 * 1024)} MB or smaller."
        )
    return payload


def normalize_image(payload: bytes, max_dimension: int = 2048) -> bytes:
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format not in ("PNG", "JPEG", "WEBP"):
                raise HTTPException(400, "Use a PNG, JPEG, or WebP image.")
            if image.width * image.height > 20000000:
                raise HTTPException(413, "Use an image smaller than 20 megapixels.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((max_dimension, max_dimension))
            image.info.clear()
            output = BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(400, "The image could not be read.") from error


async def extract_source_text(upload: UploadFile, settings: WebSettings) -> str:
    payload = await upload_bytes(upload, settings.upload_limit)
    if not payload:
        return ""
    if (upload.filename or "").lower().endswith((".txt", ".md")):
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise HTTPException(400, "Text files must use UTF-8 encoding.") from error
    else:
        image_bytes = normalize_image(payload)
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                text = pytesseract.image_to_string(image, timeout=30)
        except RuntimeError as error:
            raise HTTPException(
                400, "OCR could not read this scan. Try a text file."
            ) from error
        if not text.strip():
            raise HTTPException(
                400, "No text was found in the scan. Add context manually."
            )
    if len(text) > settings.text_limit:
        raise HTTPException(413, "Source text must be 20,000 characters or fewer.")
    return text.strip()
