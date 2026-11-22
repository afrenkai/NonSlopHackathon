from typing import Protocol

from nonslop.models.inference_schemas import ArchivistResult, MediaType


class ArchivistGateway(Protocol):
    def submit(self, media: bytes, media_type: MediaType, filename: str) -> str: ...

    def result(self, call_id: str) -> ArchivistResult | None: ...
