from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from nonslop.models.inference_schemas import MediaType

type EntryKind = Literal["text", "image", "audio", "video", "correction"]
type CapsuleView = Literal["timeline", "whole"]


class EntryEnrichment(BaseModel):
    status: Literal["running", "complete", "failed"]
    kind: Literal["transcription", "observations"]
    call_id: str | None = None
    text: str = ""
    error: str | None = None


class CapsuleRecord(BaseModel):
    capsule_id: UUID
    owner_id: str
    created_at: datetime
    title: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=4000)
    anchor_filename: str | None = None
    anchor_media_type: str | None = None


class CapsuleEntry(BaseModel):
    entry_id: UUID
    capsule_id: UUID
    owner_id: str
    created_at: datetime
    occurred_on: date
    kind: EntryKind
    text: str = Field(default="", max_length=20000)
    media_filename: str | None = None
    media_type: MediaType | None = None
    original_filename: str | None = None
    linked_job_id: UUID | None = None
    enrichment: EntryEnrichment | None = None


class CapsuleUpload(BaseModel):
    content: bytes
    kind: EntryKind
    media_type: MediaType
    suffix: str


class MediaStream(BaseModel):
    codec_type: str


class MediaProbe(BaseModel):
    streams: list[MediaStream]
