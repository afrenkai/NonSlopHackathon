from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

from nonslop.grounding.photo_schemas import PhotoMetadata
from nonslop.reconstruction.schemas import ReconstructionRecord


class SourceEvidence(BaseModel):
    title: str
    url: HttpUrl
    excerpt: str
    kind: str
    date: str | None = None


class SceneRequest(BaseModel):
    location: str = Field(min_length=3, max_length=250)
    year: int = Field(ge=1800, le=datetime.now().year)
    description: str = Field(default="", max_length=4000)
    source_text: str = Field(default="", max_length=20000)
    use_web: bool = True
    seed: int = Field(default=42, ge=0, le=4294967295)


class DeerFlowObservation(BaseModel):
    receipt: str
    url: HttpUrl
    observation: str
    page: int
    crop: tuple[float, float, float, float]


class VisitEvidence(BaseModel):
    scene: SceneRequest
    reference_metadata: list[PhotoMetadata]
    retrieved_sources: list[SourceEvidence]
    saved_research: list[SourceEvidence]
    deerflow_observations: list[DeerFlowObservation] = Field(default_factory=list)


class GenerationTask(BaseModel):
    job_id: str
    prompt: str
    seed: int
    evidence: VisitEvidence | None = None


class PhotoReading(BaseModel):
    photo_number: int = Field(ge=1)
    visible_details: list[str] = Field(min_length=1, max_length=5)
    uncertainty: str


class GenerationResult(BaseModel):
    video_path: str
    width: int
    height: int
    frames: int
    reference_observations: str = ""
    prompt_model: str = ""
    used_prompt: str = ""
    photo_readings: list[PhotoReading] = Field(default_factory=list)


class JobRecord(BaseModel):
    job_id: str
    owner_id: str = ""
    created_at: datetime
    status: Literal["queued", "running", "complete", "failed"]
    scene: SceneRequest
    render_prompt: str
    sources: list[SourceEvidence] = Field(default_factory=list)
    research_sources: list[SourceEvidence] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    has_reference: bool = False
    reference_count: int = 0
    reference_names: list[str] = Field(default_factory=list)
    reference_metadata: list[PhotoMetadata] = Field(default_factory=list)
    call_id: str | None = None
    result: GenerationResult | None = None
    error: str | None = None
    world: ReconstructionRecord | None = None
