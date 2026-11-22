from uuid import UUID

from pydantic import BaseModel, Field

from nonslop.generation.schemas import SceneRequest, SourceEvidence
from nonslop.grounding.photo_schemas import PhotoMetadata


class ReconstructionContext(BaseModel):
    scene: SceneRequest
    sources: list[SourceEvidence] = Field(default_factory=list)
    reference_metadata: list[PhotoMetadata] = Field(default_factory=list)
    reference_observations: str = ""
    video_job_id: UUID | None = None
