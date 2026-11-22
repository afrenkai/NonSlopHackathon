import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from nonslop.data.jobs import JobStore
from nonslop.generation.schemas import JobRecord
from nonslop.reconstruction.context_schemas import ReconstructionContext
from nonslop.reconstruction.protocols import ReconstructionGateway
from nonslop.reconstruction.schemas import ReconstructionRecord


class ReconstructionService:
    def __init__(self, store: JobStore, gateway: ReconstructionGateway) -> None:
        self.store = store
        self.gateway = gateway
        self.logger = logging.getLogger(__name__)

    def start(
        self, job: JobRecord, rebuild: bool = False, complete_world: bool = False
    ) -> JobRecord:
        with self.store.lock:
            current = self.store.get(job.job_id)
            assert current is not None
            if current.world is not None and not complete_world:
                if current.world.status == "running":
                    return self.refresh(current)
                if current.world.status == "complete" and not rebuild:
                    return current
            if current.status != "complete":
                raise ValueError(
                    "Wait for the generated video to finish before building the 3D space."
                )
            if self.store.active_world_count() >= 3:
                raise ValueError(
                    "Three spaces are already queued. Wait for one to finish."
                )
            try:
                references = self.store.reference_images(current)
                assert current.result is not None
                video_job_id = UUID(Path(current.result.video_path).stem)
                source_job_id = current.job_id if complete_world else None
                if complete_world:
                    current = current.model_copy(deep=True)
                    current.job_id = str(uuid4())
                    current.created_at = datetime.now(UTC)
                    current.world = None
                    self.store.save(current, references)
                current.world = ReconstructionRecord(
                    status="running",
                    pipeline_kind=(
                        "hy_world_generation"
                        if complete_world
                        else "worldmirror_reconstruction"
                    ),
                    source_job_id=source_job_id,
                    call_id=self.gateway.submit(
                        current.job_id,
                        references,
                        ReconstructionContext(
                            scene=current.scene,
                            sources=current.sources + current.research_sources,
                            video_job_id=video_job_id,
                            reference_metadata=current.reference_metadata,
                            reference_observations=(
                                current.result.reference_observations
                                if current.result
                                else ""
                            ),
                        ),
                        complete_world=complete_world,
                    ),
                    input_kind=(
                        "photos_and_generated_video"
                        if references
                        else "generated_video"
                    ),
                    input_count=len(references),
                )
            except Exception:
                self.logger.exception(
                    "Could not start reconstruction %s", current.job_id
                )
                if complete_world and current.job_id == job.job_id:
                    raise
                current.world = ReconstructionRecord(
                    status="failed",
                    pipeline_kind=(
                        "hy_world_generation"
                        if complete_world
                        else "worldmirror_reconstruction"
                    ),
                    source_job_id=job.job_id if complete_world else None,
                    error="The 3D reconstruction could not start. Please try again.",
                )
            self.store.save(current)
            return current

    def refresh(self, job: JobRecord) -> JobRecord:
        with self.store.lock:
            current = self.store.get(job.job_id)
            assert current is not None
            world = current.world
            if world is None or world.status != "running" or world.call_id is None:
                return current
            try:
                result = self.gateway.result(world.call_id)
                if result is None:
                    return current
                world.result = result
                world.status = "complete"
                world.input_kind = result.input_kind
                world.input_count = result.input_count
            except Exception:
                self.logger.exception("Reconstruction failed for %s", current.job_id)
                world.status = "failed"
                world.error = (
                    "The 3D reconstruction failed. Your source material is saved."
                )
            self.store.save(current)
            return current
