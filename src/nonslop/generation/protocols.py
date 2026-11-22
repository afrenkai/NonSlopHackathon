from typing import Protocol

from nonslop.generation.schemas import GenerationResult, GenerationTask


class GenerationGateway(Protocol):
    def submit(self, task: GenerationTask, reference_images: list[bytes]) -> str: ...

    def result(self, call_id: str) -> GenerationResult | None: ...
