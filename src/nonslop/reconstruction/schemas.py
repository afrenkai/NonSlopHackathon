from typing import Literal

from pydantic import BaseModel

from nonslop.models.world_schemas import ReconstructionInput, WorldResult


class ReconstructionRecord(BaseModel):
    status: Literal["running", "complete", "failed"]
    input_kind: ReconstructionInput = "generated_video"
    input_count: int = 0
    call_id: str | None = None
    result: WorldResult | None = None
    error: str | None = None
    pipeline_kind: Literal["worldmirror_reconstruction", "hy_world_generation"] = (
        "worldmirror_reconstruction"
    )
    source_job_id: str | None = None
