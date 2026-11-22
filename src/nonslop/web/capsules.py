from datetime import UTC, date, datetime
from uuid import UUID, uuid4

from anyio import to_thread
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from nonslop.capsules.media import read_memory_media
from nonslop.capsules.schemas import CapsuleEntry, CapsuleRecord, CapsuleView
from nonslop.capsules.service import CapsuleService
from nonslop.capsules.store import CapsuleStore
from nonslop.data.jobs import JobStore
from nonslop.web.schemas import WebSettings
from nonslop.web.security import browser_workspace, csrf_token, require_csrf


def install_capsule_routes(
    application: FastAPI,
    settings: WebSettings,
    store: CapsuleStore,
    service: CapsuleService,
    jobs: JobStore,
    templates: Jinja2Templates,
) -> None:
    def find_capsule(request: Request, capsule_id: UUID) -> CapsuleRecord:
        capsule = store.get(capsule_id, browser_workspace(request))
        if capsule is None:
            raise HTTPException(404, "Capsule not found.")
        return capsule

    def timeline(
        request: Request, capsule: CapsuleRecord, view: CapsuleView
    ) -> HTMLResponse:
        entries = [
            service.refresh(entry)
            for entry in store.entries(capsule.capsule_id, capsule.owner_id)
        ]
        return templates.TemplateResponse(
            request=request,
            name="capsule_entries.html",
            context={
                "capsule": capsule,
                "entries": entries,
                "view": view,
                "pending": any(
                    entry.enrichment is not None
                    and entry.enrichment.status == "running"
                    for entry in entries
                ),
            },
        )

    @application.get("/capsules", response_class=HTMLResponse)
    def capsule_list(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="capsules.html",
            context={
                "capsules": store.recent(browser_workspace(request)),
                "csrf": csrf_token(request),
            },
        )

    @application.post("/capsules", response_class=HTMLResponse)
    async def create_capsule(request: Request) -> HTMLResponse:
        async with request.form(max_files=1, max_fields=3, max_part_size=8192) as form:
            require_csrf(request, str(form.get("csrf", "")))
            try:
                capsule = CapsuleRecord(
                    capsule_id=uuid4(),
                    owner_id=browser_workspace(request),
                    created_at=datetime.now(UTC),
                    title=str(form.get("title", "")).strip(),
                    description=str(form.get("description", "")),
                )
            except ValidationError as error:
                raise HTTPException(
                    422,
                    "Add a title up to 120 characters and a description up to 4,000 characters.",
                ) from error
            anchor = form.get("anchor")
            anchor_bytes = None
            if isinstance(anchor, UploadFile) and anchor.filename:
                uploaded = await read_memory_media(anchor, settings.upload_limit)
                if uploaded.kind != "image":
                    raise HTTPException(400, "Use an image as the capsule's anchor.")
                capsule.anchor_filename = f"{uuid4()}.{uploaded.suffix}"
                capsule.anchor_media_type = uploaded.media_type
                anchor_bytes = uploaded.content
        await to_thread.run_sync(store.create, capsule, anchor_bytes)
        return templates.TemplateResponse(
            request=request, name="capsule_card.html", context={"capsule": capsule}
        )

    @application.get("/capsules/{capsule_id}", response_class=HTMLResponse)
    def capsule_page(request: Request, capsule_id: UUID) -> HTMLResponse:
        capsule = find_capsule(request, capsule_id)
        entries = [
            service.refresh(entry)
            for entry in store.entries(capsule_id, capsule.owner_id)
        ]
        return templates.TemplateResponse(
            request=request,
            name="capsule.html",
            context={
                "capsule": capsule,
                "entries": entries,
                "jobs": jobs.recent(capsule.owner_id),
                "csrf": csrf_token(request),
                "today": date.today(),
                "view": "timeline",
                "pending": any(
                    entry.enrichment is not None
                    and entry.enrichment.status == "running"
                    for entry in entries
                ),
            },
        )

    @application.get("/capsules/{capsule_id}/entries", response_class=HTMLResponse)
    def capsule_entries(
        request: Request, capsule_id: UUID, view: CapsuleView = "timeline"
    ) -> HTMLResponse:
        return timeline(request, find_capsule(request, capsule_id), view)

    @application.post("/capsules/{capsule_id}/entries", response_class=HTMLResponse)
    async def append_entry(request: Request, capsule_id: UUID) -> HTMLResponse:
        capsule = find_capsule(request, capsule_id)
        async with request.form(max_files=1, max_fields=7, max_part_size=65536) as form:
            require_csrf(request, str(form.get("csrf", "")))
            try:
                linked = str(form.get("linked_job_id", ""))
                entry = CapsuleEntry(
                    entry_id=uuid4(),
                    capsule_id=capsule_id,
                    owner_id=capsule.owner_id,
                    created_at=datetime.now(UTC),
                    occurred_on=date.fromisoformat(str(form.get("occurred_on", ""))),
                    kind="correction" if form.get("correction") == "true" else "text",
                    text=str(form.get("text", "")),
                    linked_job_id=UUID(linked) if linked else None,
                )
            except (ValueError, ValidationError) as error:
                raise HTTPException(
                    422, "Add a valid date and keep your words under 20,000 characters."
                ) from error
            if entry.linked_job_id is not None:
                linked_job = jobs.get(str(entry.linked_job_id))
                if linked_job is None or linked_job.owner_id != capsule.owner_id:
                    raise HTTPException(404, "Linked visit not found.")
            if entry.kind == "correction" and (
                not entry.text.strip() or entry.linked_job_id is None
            ):
                raise HTTPException(
                    422, "A correction needs your words and a linked visit."
                )
            media_bytes = None
            media = form.get("media")
            if isinstance(media, UploadFile) and media.filename:
                uploaded = await read_memory_media(media, settings.upload_limit)
                if entry.kind == "correction":
                    raise HTTPException(
                        422,
                        "Save a correction as text; attach media in a separate entry.",
                    )
                entry.kind = uploaded.kind
                entry.media_type = uploaded.media_type
                entry.media_filename = f"{entry.entry_id}.{uploaded.suffix}"
                entry.original_filename = media.filename
                media_bytes = uploaded.content
            if (
                not entry.text.strip()
                and media_bytes is None
                and entry.linked_job_id is None
            ):
                raise HTTPException(
                    422, "Add your words, a recording, an image, or a linked visit."
                )
            use_ai = form.get("use_ai") == "true"
        await to_thread.run_sync(service.append, entry, media_bytes, use_ai)
        return await to_thread.run_sync(timeline, request, capsule, "timeline")

    @application.get("/capsules/{capsule_id}/anchor")
    def capsule_anchor(request: Request, capsule_id: UUID) -> FileResponse:
        capsule = find_capsule(request, capsule_id)
        if capsule.anchor_filename is None:
            raise HTTPException(404, "Anchor image not found.")
        return FileResponse(
            store.media_path(capsule_id, capsule.anchor_filename),
            media_type=capsule.anchor_media_type,
        )

    @application.get("/capsules/{capsule_id}/entries/{entry_id}/media")
    def entry_media(request: Request, capsule_id: UUID, entry_id: UUID) -> FileResponse:
        capsule = find_capsule(request, capsule_id)
        entry = store.get_entry(entry_id, capsule.owner_id)
        if (
            entry is None
            or entry.capsule_id != capsule_id
            or entry.media_filename is None
        ):
            raise HTTPException(404, "Recording not found.")
        return FileResponse(
            store.media_path(capsule_id, entry.media_filename),
            media_type=entry.media_type,
            filename=entry.original_filename,
            content_disposition_type="inline",
        )
