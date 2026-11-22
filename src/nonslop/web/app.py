from collections.abc import Callable
from pathlib import Path
from typing import Annotated
from uuid import UUID

from anyio import to_thread
from fastapi import FastAPI, HTTPException, Query, Request
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
from nonslop.data.jobs import API_OWNER, PUBLIC_WEB_OWNER, JobStore
from nonslop.data.videos import ArtifactStore
from nonslop.generation.protocols import GenerationGateway
from nonslop.generation.schemas import JobRecord, SceneRequest
from nonslop.generation.service import QueueFullError, SceneService
from nonslop.grounding.research import research_packet_sources
from nonslop.reconstruction.protocols import ReconstructionGateway
from nonslop.reconstruction.service import ReconstructionService
from nonslop.web.body_limit import BodyLimitMiddleware
from nonslop.web.capsules import install_capsule_routes
from nonslop.web.demo_visits import (
    DEMO_VISITS,
    bundled_observations,
    saved_demo_research,
)
from nonslop.web.references import read_references
from nonslop.web.schemas import WebSettings
from nonslop.web.security import (
    csrf_token,
    require_api,
    require_csrf,
)
from nonslop.web.uploads import extract_source_text, upload_bytes
from nonslop.web.worlds import install_world_routes


def create_app(
    settings: WebSettings,
    gateway: GenerationGateway,
    world_gateway: ReconstructionGateway,
    archivist_gateway: ArchivistGateway,
    commit_data: Callable[[], None],
    reload_outputs: Callable[[], None],
    reload_worlds: Callable[[], None],
    *,
    scene_service: type[SceneService] = SceneService,
    download_video: Callable[[Path, Path], None] | None = None,
    download_world: Callable[[Path, Path], None] | None = None,
) -> FastAPI:
    application = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    application.state.local_preview = False
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
    templates.env.globals["saved_demo_research"] = saved_demo_research
    application.mount(
        "/static", StaticFiles(directory=assets / "static"), name="static"
    )
    store = JobStore(settings.data_root, commit_data)
    capsules = CapsuleStore(store)
    capsule_service = CapsuleService(capsules, archivist_gateway)
    service = scene_service(store, gateway)
    videos = ArtifactStore(
        settings.output_root, settings.serving_root, reload_outputs, download_video
    )
    worlds = ReconstructionService(store, world_gateway)
    world_artifacts = ArtifactStore(
        settings.world_root, settings.world_serving_root, reload_worlds, download_world
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
            "img-src 'self' data: blob:; media-src 'self'; connect-src 'self'; "
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

    def refreshed_job(job: JobRecord | None) -> JobRecord:
        if job is None:
            raise HTTPException(404, "Scene not found.")
        return worlds.refresh(service.refresh(job))

    def find_public_job(job_id: UUID) -> JobRecord:
        return refreshed_job(store.get_public(str(job_id)))

    def find_api_job(job_id: UUID) -> JobRecord:
        return refreshed_job(store.get(str(job_id)))

    async def submit_form(
        request: Request, browser: bool, *, visit_intake: bool = False
    ) -> JobRecord:
        if browser:
            owner_id = PUBLIC_WEB_OWNER
        else:
            require_api(request, settings)
            owner_id = API_OWNER
        async with request.form(
            max_files=settings.reference_limit + 3, max_fields=13, max_part_size=65536
        ) as form:
            if browser:
                require_csrf(request, str(form.get("csrf", "")))
            demo_key = str(form.get("visit_preset", "")) if visit_intake else ""
            demo = DEMO_VISITS.get(demo_key)
            if demo_key and demo is None:
                raise HTTPException(
                    422, "Choose a listed demo visit or your own scene."
                )
            references = await read_references(
                form,
                settings,
                demo_photo=assets / "static" / demo.photo if demo else None,
            )
            if browser and not visit_intake and not references.images:
                raise HTTPException(
                    422, "Add at least one photograph to create a field note."
                )
            source_text = str(form.get("source_text", ""))
            source_file = form.get("source_file")
            if isinstance(source_file, UploadFile) and source_file.filename:
                extracted = await extract_source_text(source_file, settings)
                source_text = "\n\n".join(filter(None, [source_text, extracted]))
            research = list(demo.research) if demo else []
            research_file = form.get("research_file")
            if isinstance(research_file, UploadFile) and research_file.filename:
                try:
                    research.extend(
                        research_packet_sources(
                            await upload_bytes(research_file, settings.upload_limit)
                        )
                    )
                except (TypeError, ValueError) as error:
                    raise HTTPException(422, str(error)) from error
            try:
                scene = SceneRequest(
                    location=demo.location if demo else str(form.get("location", "")),
                    year=demo.year if demo else int(str(form.get("year", "0"))),
                    description=str(form.get("description", "")).strip(),
                    source_text=source_text,
                    use_web=demo is None
                    and str(form.get("use_web", "" if browser else "true")).lower()
                    in ("true", "on", "1"),
                    seed=int(str(form.get("seed", "42"))),
                )
            except (ValueError, ValidationError) as error:
                raise HTTPException(
                    422,
                    "Add a place and a valid year. Keep notes under 4,000 characters and source context under 20,000 characters.",
                ) from error
            if not references.images and not scene.description:
                raise HTTPException(422, "Add a photograph or a scene description.")
        for recent in store.recent():
            await to_thread.run_sync(service.refresh, recent)
        try:
            return await service.submit(
                scene,
                references.images,
                owner_id,
                references.filenames,
                references.metadata,
                research_sources=research,
                deerflow=bundled_observations(demo.observation_file) if demo else (),
            )
        except QueueFullError as error:
            raise HTTPException(409, str(error)) from error

    def video_response(job: JobRecord) -> FileResponse:
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
        return templates.TemplateResponse(request=request, name="index.html")

    @application.get("/dashboard", response_class=HTMLResponse)
    def dashboard(
        request: Request,
        box_page: Annotated[int, Query(ge=1, le=2**31 - 1)] = 1,
        capsule_page: Annotated[int, Query(ge=1, le=2**31 - 1)] = 1,
    ) -> Response:
        jobs, more_jobs = store.public_page(box_page)
        page_capsules, more_capsules = capsules.public_page(capsule_page)
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "csrf": csrf_token(request),
                "jobs": jobs,
                "demo_visits": DEMO_VISITS,
                "capsules": page_capsules,
                "box_page": box_page,
                "capsule_page": capsule_page,
                "more_jobs": more_jobs,
                "more_capsules": more_capsules,
            },
        )

    @application.get("/visits/new", response_class=RedirectResponse, status_code=303)
    def new_visit() -> RedirectResponse:
        return RedirectResponse("/dashboard?add=photo", status_code=303)

    @application.post("/visits/new", response_class=HTMLResponse)
    async def save_new_visit(request: Request) -> Response:
        job = await submit_form(request, browser=True, visit_intake=True)
        location = f"/visits/{job.job_id}"
        if request.headers.get("HX-Request") == "true":
            return Response(headers={"HX-Redirect": location})
        return RedirectResponse(location, status_code=303)

    @application.get("/visits/{job_id}", response_class=HTMLResponse)
    def visit_page(request: Request, job_id: UUID) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="visit.html",
            context={
                "csrf": csrf_token(request),
                "job": find_public_job(job_id),
                "linked_memories": capsules.linked_entries(job_id),
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
                "job": find_public_job(job_id),
                "csrf": csrf_token(request),
            },
        )

    @application.get("/jobs/{job_id}/video")
    def browser_video(job_id: UUID) -> FileResponse:
        return video_response(find_public_job(job_id))

    @application.get("/jobs/{job_id}/reference")
    def browser_reference(job_id: UUID) -> FileResponse:
        job = find_public_job(job_id)
        path = settings.data_root / "uploads" / f"{job.job_id}.png"
        if not job.has_reference or not path.is_file():
            raise HTTPException(404, "Reference image not found.")
        return FileResponse(path, media_type="image/png")

    @application.get("/jobs/{job_id}/references/{index}")
    def browser_reference_image(job_id: UUID, index: int) -> FileResponse:
        job = find_public_job(job_id)
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
        return find_api_job(job_id)

    @application.get("/api/jobs/{job_id}/video")
    def api_video(request: Request, job_id: UUID) -> FileResponse:
        require_api(request, settings)
        return video_response(find_api_job(job_id))

    install_world_routes(
        application,
        settings,
        worlds,
        world_artifacts,
        templates,
        find_public_job,
        find_api_job,
        capsules,
    )
    install_capsule_routes(
        application, settings, capsules, capsule_service, store, templates
    )
    return application
