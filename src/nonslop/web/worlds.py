from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from nonslop.capsules.store import CapsuleStore
from nonslop.data.videos import ArtifactStore
from nonslop.generation.schemas import JobRecord
from nonslop.reconstruction.service import ReconstructionService
from nonslop.web.schemas import WebSettings
from nonslop.web.security import csrf_token, require_api, require_csrf


def install_world_routes(
    application: FastAPI,
    settings: WebSettings,
    service: ReconstructionService,
    artifacts: ArtifactStore,
    templates: Jinja2Templates,
    find_public_job: Callable[[UUID], JobRecord],
    find_api_job: Callable[[UUID], JobRecord],
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
        job = find_public_job(job_id)
        return templates.TemplateResponse(
            request=request,
            name="job.html",
            context={"job": start_world(job), "csrf": csrf_token(request)},
        )

    @application.get("/jobs/{job_id}/world", response_class=HTMLResponse)
    def view_world(request: Request, job_id: UUID) -> HTMLResponse:
        job = find_public_job(job_id)
        if job.world is None or job.world.status != "complete":
            raise HTTPException(409, "The reconstructed space is not ready yet.")
        return templates.TemplateResponse(
            request=request,
            name="world.html",
            context={
                "job": job,
                "linked_memories": capsules.linked_entries(job_id),
            },
        )

    @application.get("/jobs/{job_id}/world/gaussians.ply")
    def browser_splats(job_id: UUID) -> FileResponse:
        return artifact_response(find_public_job(job_id), "gaussians.ply")

    @application.get("/jobs/{job_id}/world/cameras")
    def browser_cameras(job_id: UUID) -> FileResponse:
        return artifact_response(find_public_job(job_id), "camera_params.json")

    @application.post("/api/jobs/{job_id}/world", response_model=JobRecord)
    def reconstruct_api(request: Request, job_id: UUID) -> JobRecord:
        require_api(request, settings)
        return start_world(find_api_job(job_id))

    @application.get("/api/jobs/{job_id}/world/gaussians.ply")
    def api_splats(request: Request, job_id: UUID) -> FileResponse:
        require_api(request, settings)
        return artifact_response(find_api_job(job_id), "gaussians.ply")

    @application.get("/api/jobs/{job_id}/world/cameras")
    def api_cameras(request: Request, job_id: UUID) -> FileResponse:
        require_api(request, settings)
        return artifact_response(find_api_job(job_id), "camera_params.json")
