from typing import Literal

from pydantic import BaseModel

from nonslop.models.world_schemas import WorldResult


class ReconstructionRecord(BaseModel):
    status: Literal["running", "complete", "failed"]
    call_id: str | None = None
    result: WorldResult | None = None
    error: str | None = None
