import modal

from nonslop.models.inference_schemas import ArchivistResult, MediaType


class ModalArchivistGateway:
    def __init__(self, worker: modal.Cls) -> None:
        self.worker = worker

    def submit(self, media: bytes, media_type: MediaType, filename: str) -> str:
        if media_type.startswith("audio/"):
            return self.worker().transcribe.spawn(media, filename).object_id
        return (
            self.worker()
            .analyze.spawn(
                "Describe only details visible or audible in this source. Transcribe legible text. "
                "Separate observations from uncertainty. Do not invent dates, identities, or history. "
                "This is an AI observation for a personal archive; never write a memory on its owner's behalf.",
                media,
                media_type,
            )
            .object_id
        )

    def result(self, call_id: str) -> ArchivistResult | None:
        try:
            payload = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            return None
        return ArchivistResult.model_validate_json(payload)
