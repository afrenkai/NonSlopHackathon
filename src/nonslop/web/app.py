from base64 import b64encode
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

from anyio import to_thread
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.datastructures import UploadFile
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response

from nonslop.capsules.protocols import ArchivistGateway
from nonslop.capsules.service import CapsuleService
from nonslop.capsules.store import CapsuleStore
from nonslop.data.jobs import JobStore
from nonslop.data.videos import ArtifactStore
from nonslop.generation.protocols import GenerationGateway
from nonslop.generation.schemas import JobRecord, SceneRequest
from nonslop.generation.service import QueueFullError, SceneService
from nonslop.reconstruction.protocols import ReconstructionGateway
from nonslop.reconstruction.service import ReconstructionService
from nonslop.web.body_limit import BodyLimitMiddleware
from nonslop.web.capsules import install_capsule_routes
from nonslop.web.image_search import (
    GALLERY_IMAGE_LIMIT,
    IMAGE_REFERENCE_LIMIT,
    ImageSearchError,
    image_query,
    search_images,
)
from nonslop.web.references import read_references
from nonslop.web.schemas import WebSettings
from nonslop.web.security import (
    browser_workspace,
    csrf_token,
    require_api,
    require_csrf,
)
from nonslop.web.uploads import extract_source_text
from nonslop.web.worlds import install_world_routes


def create_app(
    settings: WebSettings,
    gateway: GenerationGateway,
    world_gateway: ReconstructionGateway,
    archivist_gateway: ArchivistGateway,
    commit_data: Callable[[], None],
    reload_outputs: Callable[[], None],
    reload_worlds: Callable[[], None],
) -> FastAPI:
    application = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    application.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret.get_secret_value(),
        session_cookie="again_here",
        max_age=31536000,
        same_site="strict",
        https_only=settings.secure_cookies,
    )
    application.add_middleware(
        BodyLimitMiddleware, limit=settings.total_upload_limit + 65536
    )
    assets = settings.assets_root
    templates = Jinja2Templates(directory=assets / "templates")
    application.mount(
        "/static", StaticFiles(directory=assets / "static"), name="static"
    )

    @application.middleware("http")
    async def response_headers(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; "
            "img-src 'self' data:; media-src 'self'; connect-src 'self'; "
            "worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        return response

    @application.exception_handler(HTTPException)
    async def http_error(
        request: Request, error: HTTPException
    ) -> HTMLResponse | JSONResponse:
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                {"detail": str(error.detail)}, status_code=error.status_code
            )
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error": str(error.detail)},
            status_code=error.status_code,
        )

    if settings.references_only:
        current_year = datetime.now().year

        @application.post("/", response_class=HTMLResponse)
        @application.get("/", response_class=HTMLResponse)
        async def references(
            request: Request,
            location: Annotated[str, Form(max_length=250)] = "",
            year: Annotated[int | None, Form(ge=1800, le=current_year)] = None,
            description: Annotated[str, Form(max_length=4000)] = "",
            csrf: Annotated[str, Form()] = "",
        ) -> HTMLResponse:
            results = None
            error = None
            if request.method == "POST":
                require_csrf(request, csrf)
                if len(description.strip()) < 5:
                    raise HTTPException(422, "Describe the images you want to find.")
                try:
                    images = await search_images(
                        image_query(description, location, year),
                        GALLERY_IMAGE_LIMIT,
                        location=location,
                    )
                    results = [
                        (image.caption, b64encode(image.content).decode("ascii"))
                        for image in images
                    ]
                except ImageSearchError as failure:
                    error = str(failure)
            return templates.TemplateResponse(
                request=request,
                name="references.html",
                context={
                    "location": location,
                    "year": year,
                    "description": description,
                    "current_year": current_year,
                    "results": results,
                    "error": error,
                    "csrf": csrf_token(request),
                },
            )

        return application

    store = JobStore(settings.data_root, commit_data)
    capsules = CapsuleStore(store)
    capsule_service = CapsuleService(capsules, archivist_gateway)
    service = SceneService(store, gateway)
    videos = ArtifactStore(settings.output_root, settings.serving_root, reload_outputs)
    worlds = ReconstructionService(store, world_gateway)
    world_artifacts = ArtifactStore(
        settings.world_root, settings.world_serving_root, reload_worlds
    )

    def find_job(job_id: UUID, owner_id: str | None = None) -> JobRecord:
        job = store.get(str(job_id))
        if job is None or (owner_id is not None and job.owner_id != owner_id):
            raise HTTPException(404, "Scene not found.")
        return worlds.refresh(service.refresh(job))

    async def submit_form(request: Request, browser: bool) -> JobRecord:
        if browser:
            owner_id = browser_workspace(request)
        else:
            require_api(request, settings)
            owner_id = "api"
        async with request.form(
            max_files=settings.reference_limit + 2, max_fields=12, max_part_size=65536
        ) as form:
            if browser:
                require_csrf(request, str(form.get("csrf", "")))
            source_text = str(form.get("source_text", ""))
            source_file = form.get("source_file")
            if isinstance(source_file, UploadFile) and source_file.filename:
                extracted = await extract_source_text(source_file, settings)
                source_text = "\n\n".join(filter(None, [source_text, extracted]))
            try:
                scene = SceneRequest(
                    location=str(form.get("location", "")),
                    year=int(str(form.get("year", "0"))),
                    description=str(form.get("description", "")),
                    source_text=source_text,
                    use_web=str(form.get("use_web", "" if browser else "true")).lower()
                    in ("true", "on", "1"),
                    seed=int(str(form.get("seed", "42"))),
                )
            except (ValueError, ValidationError) as error:
                raise HTTPException(
                    422,
                    "Add a location, a valid year, and a description. Keep context under 20,000 characters.",
                ) from error
            references = await read_references(form, settings)
        notices = []
        remaining = settings.reference_limit - len(references.images)
        if scene.use_web and remaining:
            try:
                images = await search_images(
                    image_query(scene.description, scene.location, scene.year),
                    min(IMAGE_REFERENCE_LIMIT, remaining),
                    location=scene.location,
                )
            except ImageSearchError as failure:
                notices.append(str(failure))
            else:
                position = len(references.images)
                if position > 1:
                    position -= 1
                references.images[position:position] = [
                    image.content for image in images
                ]
                references.filenames[position:position] = [
                    image.caption for image in images
                ]
                if not images:
                    notices.append(
                        "The image collections returned no usable reference images."
                    )
        for recent in store.recent():
            await to_thread.run_sync(service.refresh, recent)
        try:
            return await service.submit(
                scene, references.images, owner_id, references.filenames, notices
            )
        except QueueFullError as error:
            raise HTTPException(409, str(error)) from error

    def video_response(job_id: UUID, owner_id: str | None = None) -> FileResponse:
        job = find_job(job_id, owner_id)
        if job.status != "complete":
            raise HTTPException(409, "This scene is not ready yet.")
        try:
            path = videos.materialize(Path(f"{job.job_id}.mp4"))
        except FileNotFoundError as error:
            raise HTTPException(
                404, "The generated video is not available yet."
            ) from error
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"street-{job.job_id}.mp4",
            content_disposition_type="inline",
        )

    @application.get("/login", response_class=HTMLResponse)
    def login_page() -> RedirectResponse:
        return RedirectResponse("/", status_code=303)

    @application.get("/", response_class=HTMLResponse)
    def home(request: Request) -> Response:
        jobs = [
            worlds.refresh(service.refresh(job))
            for job in store.recent(owner_id=browser_workspace(request))
        ]
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"csrf": csrf_token(request), "jobs": jobs},
        )

    @application.get("/visits/{job_id}", response_class=HTMLResponse)
    def visit_page(request: Request, job_id: UUID) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "csrf": csrf_token(request),
                "jobs": [find_job(job_id, browser_workspace(request))],
                "linked_memories": capsules.linked_entries(
                    job_id, browser_workspace(request)
                ),
            },
        )

    @application.post("/jobs", response_class=HTMLResponse)
    async def submit_browser(request: Request) -> HTMLResponse:
        job = await submit_form(request, browser=True)
        return templates.TemplateResponse(
            request=request,
            name="job.html",
            context={"job": job, "csrf": csrf_token(request)},
        )

    @application.get("/jobs/{job_id}", response_class=HTMLResponse)
    def browser_job(request: Request, job_id: UUID) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="job.html",
            context={
                "job": find_job(job_id, browser_workspace(request)),
                "csrf": csrf_token(request),
            },
        )

    @application.get("/jobs/{job_id}/video")
    def browser_video(request: Request, job_id: UUID) -> FileResponse:
        return video_response(job_id, browser_workspace(request))

    @application.get("/jobs/{job_id}/reference")
    def browser_reference(request: Request, job_id: UUID) -> FileResponse:
        job = find_job(job_id, browser_workspace(request))
        path = settings.data_root / "uploads" / f"{job.job_id}.png"
        if not job.has_reference or not path.is_file():
            raise HTTPException(404, "Reference image not found.")
        return FileResponse(path, media_type="image/png")

    @application.get("/jobs/{job_id}/references/{index}")
    def browser_reference_image(
        request: Request, job_id: UUID, index: int
    ) -> FileResponse:
        job = find_job(job_id, browser_workspace(request))
        if not 0 <= index < job.reference_count:
            raise HTTPException(404, "Reference image not found.")
        path = settings.data_root / "uploads" / job.job_id / f"{index}.png"
        return FileResponse(path, media_type="image/png")

    @application.post("/api/jobs", response_model=JobRecord)
    async def api_submit(request: Request) -> JobRecord:
        return await submit_form(request, browser=False)

    @application.get("/api/jobs/{job_id}", response_model=JobRecord)
    def api_job(request: Request, job_id: UUID) -> JobRecord:
        require_api(request, settings)
        return find_job(job_id)

    @application.get("/api/jobs/{job_id}/video")
    def api_video(request: Request, job_id: UUID) -> FileResponse:
        require_api(request, settings)
        return video_response(job_id)

    install_world_routes(
        application, settings, worlds, world_artifacts, templates, find_job, capsules
    )
    install_capsule_routes(
        application, settings, capsules, capsule_service, store, templates
    )
    return application
