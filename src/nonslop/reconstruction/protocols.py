from typing import Protocol

from nonslop.models.world_schemas import WorldResult


class ReconstructionGateway(Protocol):
    def submit(self, job_id: str) -> str: ...

    def result(self, call_id: str) -> WorldResult | None: ...
