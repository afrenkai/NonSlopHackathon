import modal

from nonslop.generation.schemas import GenerationResult, GenerationTask
from nonslop.models.world_schemas import WorldResult


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
    def __init__(self, worker: modal.Cls) -> None:
        self.worker = worker

    def submit(self, job_id: str, reference_images: list[bytes]) -> str:
        return self.worker().reconstruct.spawn(job_id, reference_images).object_id

    def result(self, call_id: str) -> WorldResult | None:
        try:
            payload = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            return None
        return WorldResult.model_validate_json(payload)
