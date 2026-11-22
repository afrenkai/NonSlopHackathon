from pathlib import Path

from anyio import to_thread
from fastapi import HTTPException
from starlette.datastructures import FormData, UploadFile

from nonslop.web.schemas import ReferenceBundle, WebSettings
from nonslop.web.uploads import normalize_image, upload_bytes


async def read_references(form: FormData, settings: WebSettings) -> ReferenceBundle:
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
    references = ReferenceBundle()
    total = 0
    for upload in selected:
        content = await upload_bytes(upload, settings.upload_limit)
        total += len(content)
        if total > settings.total_upload_limit:
            raise HTTPException(413, "Keep the complete upload under 40 MB.")
        references.images.append(
            await to_thread.run_sync(
                normalize_image, content, 768 if len(selected) > 1 else 2048
            )
        )
        references.filenames.append(Path(upload.filename or "reference.png").name[:250])
    return references
