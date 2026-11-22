from typing import Protocol

from nonslop.models.world_schemas import WorldResult


class ReconstructionGateway(Protocol):
    def submit(self, job_id: str, reference_images: list[bytes]) -> str: ...

    def result(self, call_id: str) -> WorldResult | None: ...
