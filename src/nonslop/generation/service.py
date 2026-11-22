import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

from anyio import to_thread

from nonslop.data.jobs import JobStore
from nonslop.generation.prompts import build_scene_prompt
from nonslop.generation.protocols import GenerationGateway
from nonslop.generation.schemas import (
    DeerFlowObservation,
    GenerationTask,
    JobRecord,
    SceneRequest,
    SourceEvidence,
    VisitEvidence,
)
from nonslop.grounding.photo_schemas import PhotoMetadata
from nonslop.grounding.retrieval import retrieve_sources


class QueueFullError(Exception):
    pass


class SceneService:
    def __init__(self, store: JobStore, gateway: GenerationGateway) -> None:
        self.store = store
        self.gateway = gateway
        self.logger = logging.getLogger(__name__)

    async def submit(
        self,
        scene: SceneRequest,
        reference_images: list[bytes],
        owner_id: str,
        reference_names: list[str],
        reference_metadata: list[PhotoMetadata],
        *,
        research_sources: Sequence[SourceEvidence] = (),
        deerflow: Sequence[DeerFlowObservation] = (),
    ) -> JobRecord:
        grounding = await retrieve_sources(scene)
        job = JobRecord(
            job_id=str(uuid4()),
            owner_id=owner_id,
            created_at=datetime.now(UTC),
            status="queued",
            scene=scene,
            render_prompt=build_scene_prompt(
                scene,
                grounding.sources,
                bool(reference_images),
                reference_metadata,
                research_sources=research_sources,
            ),
            sources=grounding.sources,
            research_sources=list(research_sources),
            notices=grounding.notices,
            has_reference=bool(reference_images),
            reference_count=len(reference_images),
            reference_names=reference_names,
            reference_metadata=reference_metadata,
        )
        if not await to_thread.run_sync(self.store.reserve, job, reference_images):
            raise QueueFullError(
                "Four scenes are already queued. Wait for one to finish."
            )
        try:
            job.call_id = await to_thread.run_sync(
                self.gateway.submit,
                GenerationTask(
                    job_id=job.job_id,
                    prompt=job.render_prompt,
                    seed=scene.seed,
                    evidence=VisitEvidence(
                        scene=scene,
                        reference_metadata=reference_metadata,
                        retrieved_sources=job.sources,
                        saved_research=job.research_sources,
                        deerflow_observations=list(deerflow),
                    ),
                ),
                reference_images,
            )
            job.status = "running"
        except Exception:
            self.logger.exception("Could not submit generation job %s", job.job_id)
            job.status = "failed"
            job.error = "The generation could not be started. Please try again."
        await to_thread.run_sync(self.store.save, job)
        return job

    def refresh(self, job: JobRecord) -> JobRecord:
        with self.store.lock:
            current = self.store.get(job.job_id)
            assert current is not None
            if current.status not in ("queued", "running") or current.call_id is None:
                return current
            try:
                result = self.gateway.result(current.call_id)
                if result is None:
                    return current
                current.result = result
                current.status = "complete"
            except Exception:
                self.logger.exception("Generation job %s failed", current.job_id)
                current.status = "failed"
                current.error = "Generation failed. The server recorded the error; you can try again."
            self.store.save(current)
            return current
