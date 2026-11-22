import logging

from nonslop.capsules.protocols import ArchivistGateway
from nonslop.capsules.schemas import CapsuleEntry, EntryEnrichment
from nonslop.capsules.store import CapsuleStore


class CapsuleService:
    def __init__(self, store: CapsuleStore, gateway: ArchivistGateway) -> None:
        self.store = store
        self.gateway = gateway
        self.logger = logging.getLogger(__name__)

    def append(
        self, entry: CapsuleEntry, media: bytes | None, use_ai: bool
    ) -> CapsuleEntry:
        self.store.append(entry, media)
        if not use_ai or media is None or entry.media_type is None:
            return entry
        enrichment = EntryEnrichment(
            status="running",
            kind="transcription" if entry.kind == "audio" else "observations",
        )
        try:
            enrichment.call_id = self.gateway.submit(
                media, entry.media_type, entry.original_filename or "memory"
            )
        except Exception:
            self.logger.exception(
                "Could not submit capsule enrichment %s", entry.entry_id
            )
            enrichment.status = "failed"
            enrichment.error = (
                "AI processing could not start. Your original entry is saved."
            )
        return self.store.update_enrichment(entry.entry_id, entry.owner_id, enrichment)

    def refresh(self, entry: CapsuleEntry) -> CapsuleEntry:
        enrichment = entry.enrichment
        if (
            enrichment is None
            or enrichment.status != "running"
            or enrichment.call_id is None
        ):
            return entry
        try:
            result = self.gateway.result(enrichment.call_id)
            if result is None:
                return entry
            enrichment.text = result.text
            enrichment.status = "complete"
        except Exception:
            self.logger.exception("Capsule enrichment failed %s", entry.entry_id)
            enrichment.status = "failed"
            enrichment.error = "AI processing failed. Your original recording is saved."
        return self.store.update_enrichment(entry.entry_id, entry.owner_id, enrichment)
