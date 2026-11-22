import modal

from nonslop.generation.schemas import GenerationResult, GenerationTask
from nonslop.models.world_schemas import WorldResult
from nonslop.reconstruction.context_schemas import ReconstructionContext


class ModalGenerationGateway:
    def __init__(self, worker: modal.Function) -> None:
        self.worker = worker

    def submit(self, task: GenerationTask, reference_images: list[bytes]) -> str:
        call = self.worker.spawn(task.model_dump_json(), reference_images)
        return call.object_id

    def result(self, call_id: str) -> GenerationResult | None:
        try:
            payload = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            return None
        return GenerationResult.model_validate_json(payload)


class ModalReconstructionGateway:
    def __init__(
        self, worker: modal.Function, world_worker: modal.Function | None = None
    ) -> None:
        self.worker = worker
        self.world_worker = world_worker

    def submit(
        self,
        job_id: str,
        reference_images: list[bytes],
        context: ReconstructionContext,
        *,
        complete_world: bool = False,
    ) -> str:
        worker = self.world_worker if complete_world else self.worker
        if worker is None:
            raise ValueError("The world-generation worker is not deployed.")
        return worker.spawn(
            job_id, reference_images, context.model_dump_json()
        ).object_id

    def result(self, call_id: str) -> WorldResult | None:
        try:
            payload = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            return None
        return WorldResult.model_validate_json(payload)
