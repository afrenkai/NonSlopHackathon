from pathlib import Path

from anyio import to_thread
from fastapi import HTTPException
from starlette.datastructures import FormData, UploadFile

from nonslop.grounding.photos import photo_metadata
from nonslop.web.schemas import ReferenceBundle, WebSettings
from nonslop.web.uploads import normalize_image, upload_bytes


async def add_reference(
    references: ReferenceBundle, content: bytes, filename: str
) -> None:
    references.images.append(await to_thread.run_sync(normalize_image, content))
    references.filenames.append(filename)
    references.metadata.append(
        await to_thread.run_sync(photo_metadata, content, filename)
    )


async def read_references(
    form: FormData, settings: WebSettings, *, demo_photo: Path | None = None
) -> ReferenceBundle:
    references = ReferenceBundle()
    if demo_photo is not None:
        content = await to_thread.run_sync(demo_photo.read_bytes)
        await add_reference(references, content, demo_photo.name)
        return references

    opening = form.get("reference_image")
    selected: list[UploadFile] = []
    if isinstance(opening, UploadFile) and opening.filename:
        selected.append(opening)
    additional = [
        upload
        for upload in form.getlist("reference_images")
        if isinstance(upload, UploadFile) and upload.filename
    ]
    selected.extend(
        sorted(additional, key=lambda upload: str(upload.filename).casefold())
    )
    if len(selected) > settings.reference_limit:
        raise HTTPException(
            413, f"Choose at most {settings.reference_limit} reference images."
        )
    total = 0
    for upload in selected:
        content = await upload_bytes(upload, settings.upload_limit)
        total += len(content)
        if total > settings.total_upload_limit:
            raise HTTPException(413, "Keep the complete upload under 40 MB.")
        filename = Path(upload.filename or "reference.png").name[:250]
        await add_reference(references, content, filename)
    return references
