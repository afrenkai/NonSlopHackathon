from pydantic import BaseModel, Field

from nonslop.generation.schemas import SourceEvidence


class ArchiveItem(BaseModel):
    identifier: str
    title: str | list[str]
    description: str | list[str] = ""
    date: str | None = None


class ArchiveResults(BaseModel):
    numFound: int = 0
    docs: list[ArchiveItem] = Field(default_factory=list)


class ArchiveResponse(BaseModel):
    response: ArchiveResults


class GroundingResult(BaseModel):
    sources: list[SourceEvidence] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
