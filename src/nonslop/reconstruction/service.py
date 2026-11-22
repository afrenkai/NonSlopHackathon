import logging

from nonslop.data.jobs import JobStore
from nonslop.generation.schemas import JobRecord
from nonslop.reconstruction.protocols import ReconstructionGateway
from nonslop.reconstruction.schemas import ReconstructionRecord


class ReconstructionService:
    def __init__(self, store: JobStore, gateway: ReconstructionGateway) -> None:
        self.store = store
        self.gateway = gateway
        self.logger = logging.getLogger(__name__)

    def start(self, job: JobRecord, rebuild: bool = False) -> JobRecord:
        with self.store.lock:
            current = self.store.get(job.job_id)
            assert current is not None
            if current.world is not None:
                if current.world.status == "running":
                    return self.refresh(current)
                if current.world.status == "complete" and not rebuild:
                    return current
            if self.store.active_world_count() >= 2:
                raise ValueError(
                    "Two spaces are already queued. Wait for one to finish."
                )
            try:
                references = self.store.reference_images(current)
                current.world = ReconstructionRecord(
                    status="running",
                    call_id=self.gateway.submit(current.job_id, references),
                    input_kind="reference_photos" if references else "generated_video",
                    input_count=len(references),
                )
            except Exception:
                self.logger.exception(
                    "Could not start reconstruction %s", current.job_id
                )
                current.world = ReconstructionRecord(
                    status="failed",
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
