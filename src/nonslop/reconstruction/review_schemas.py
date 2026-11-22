from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FrameDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    usable: bool
    reason: str = Field(min_length=1, max_length=500)


class ReconstructionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: Literal["indoor", "outdoor", "uncertain"]
    frames: list[FrameDecision] = Field(min_length=1, max_length=6)
    summary: str = Field(min_length=1, max_length=2000)
