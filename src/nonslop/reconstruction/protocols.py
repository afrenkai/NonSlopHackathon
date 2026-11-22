from typing import Protocol

from nonslop.models.world_schemas import WorldResult
from nonslop.reconstruction.context_schemas import ReconstructionContext


class ReconstructionGateway(Protocol):
    def submit(
        self,
        job_id: str,
        reference_images: list[bytes],
        context: ReconstructionContext,
        *,
        complete_world: bool = False,
    ) -> str: ...

    def result(self, call_id: str) -> WorldResult | None: ...
