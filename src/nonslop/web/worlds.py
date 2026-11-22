from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from anyio import to_thread
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from nonslop.capsules.store import CapsuleStore
from nonslop.data.videos import ArtifactStore
from nonslop.generation.schemas import JobRecord
from nonslop.reconstruction.service import ReconstructionService
from nonslop.web.schemas import WebSettings
from nonslop.web.security import (
    browser_workspace,
    csrf_token,
    require_api,
    require_csrf,
)


def install_world_routes(
    application: FastAPI,
    settings: WebSettings,
    service: ReconstructionService,
    artifacts: ArtifactStore,
    templates: Jinja2Templates,
    find_job: Callable[[UUID, str | None], JobRecord],
    capsules: CapsuleStore,
) -> None:
    def start_world(job: JobRecord) -> JobRecord:
        if job.status != "complete":
            raise HTTPException(409, "Wait for the video to finish first.")
        try:
            return service.start(job)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    def artifact_response(job: JobRecord, filename: str) -> FileResponse:
        if job.world is None or job.world.status != "complete":
            raise HTTPException(409, "The reconstructed space is not ready yet.")
        try:
            path = artifacts.materialize(Path(job.job_id) / filename)
        except FileNotFoundError as error:
            raise HTTPException(404, "The 3D artifact is not available yet.") from error
        return FileResponse(
            path,
            media_type=(
                "application/json"
                if filename.endswith(".json")
                else "application/octet-stream"
            ),
        )

    @application.post("/jobs/{job_id}/world", response_class=HTMLResponse)
    async def reconstruct_browser(request: Request, job_id: UUID) -> HTMLResponse:
        async with request.form(max_files=0, max_fields=1, max_part_size=1024) as form:
            require_csrf(request, str(form.get("csrf", "")))
        job = await to_thread.run_sync(find_job, job_id, browser_workspace(request))
        started = await to_thread.run_sync(start_world, job)
        return templates.TemplateResponse(
            request=request,
            name="job.html",
            context={"job": started, "csrf": csrf_token(request)},
        )

    @application.get("/jobs/{job_id}/world", response_class=HTMLResponse)
    def view_world(request: Request, job_id: UUID) -> HTMLResponse:
        job = find_job(job_id, browser_workspace(request))
        if job.world is None or job.world.status != "complete":
            raise HTTPException(409, "The reconstructed space is not ready yet.")
        return templates.TemplateResponse(
            request=request,
            name="world.html",
            context={
                "job": job,
                "linked_memories": capsules.linked_entries(job_id, job.owner_id),
            },
        )

    @application.get("/jobs/{job_id}/world/gaussians.ply")
    def browser_splats(request: Request, job_id: UUID) -> FileResponse:
        return artifact_response(
            find_job(job_id, browser_workspace(request)), "gaussians.ply"
        )

    @application.get("/jobs/{job_id}/world/cameras")
    def browser_cameras(request: Request, job_id: UUID) -> FileResponse:
        return artifact_response(
            find_job(job_id, browser_workspace(request)), "camera_params.json"
        )

    @application.post("/api/jobs/{job_id}/world", response_model=JobRecord)
    def reconstruct_api(request: Request, job_id: UUID) -> JobRecord:
        require_api(request, settings)
        return start_world(find_job(job_id, None))

    @application.get("/api/jobs/{job_id}/world/gaussians.ply")
    def api_splats(request: Request, job_id: UUID) -> FileResponse:
        require_api(request, settings)
        return artifact_response(find_job(job_id, None), "gaussians.ply")

    @application.get("/api/jobs/{job_id}/world/cameras")
    def api_cameras(request: Request, job_id: UUID) -> FileResponse:
        require_api(request, settings)
        return artifact_response(find_job(job_id, None), "camera_params.json")
