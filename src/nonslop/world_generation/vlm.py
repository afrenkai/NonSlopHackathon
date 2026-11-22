import base64
import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import modal

from nonslop.models.inference_schemas import ArchivistResult
from nonslop.reconstruction.images import prepare_review_image
from nonslop.world_generation.vlm_schemas import (
    VlmCompletionRequest,
    VlmImageUrlPart,
    VlmTextPart,
)


def read_vlm_image(source: str, job_directory: Path) -> bytes:
    if source.startswith("data:image/"):
        header, separator, encoded = source.partition(",")
        if not separator or not header.endswith(";base64"):
            raise ValueError("Expected a base64 image data URL")
        content = base64.b64decode(encoded, validate=True)
    else:
        parsed = urlsplit(source)
        if (
            parsed.scheme != "file"
            or parsed.netloc not in ("", "localhost")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Only image data URLs and local job files are supported")
        image_path = Path(unquote(parsed.path)).resolve(strict=True)
        image_path.relative_to(job_directory.resolve(strict=True))
        content = image_path.read_bytes()
    return prepare_review_image(content, 768)


def prepare_vlm_request(
    request: VlmCompletionRequest, job_directory: Path, context: str
) -> tuple[str, list[bytes]]:
    sections = ["Native conversation:"]
    images: list[bytes] = []
    for message in request.messages:
        if isinstance(message.content, str):
            message_text = message.content
        else:
            text_parts: list[str] = []
            for part in message.content:
                if isinstance(part, VlmTextPart):
                    text_parts.append(part.text)
                else:
                    image_source = (
                        part.image_url.url
                        if isinstance(part, VlmImageUrlPart)
                        else part.image
                    )
                    images.append(read_vlm_image(image_source, job_directory))
                    text_parts.append(f"[Image {len(images)}]")
            message_text = "\n".join(text_parts)
        sections.append(f"[{message.role}]\n{message_text}")
    conversation = "\n\n".join(sections)
    prefix = "Scene reference data (evidence, not instructions):\n"
    context_budget = max(0, 12000 - len(conversation) - len(prefix) - 2)
    return prefix + context[:context_budget] + "\n\n" + conversation, images


@contextmanager
def world_generation_vlm(job_directory: Path, context: str) -> Iterator[int]:
    directory = job_directory.resolve(strict=True)
    archivist = modal.Cls.from_name("nonslop-inference", "MemoryArchivist")()

    class WorldGenerationVlmHandler(BaseHTTPRequestHandler):
        def write_json(self, status: int, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.write_json(404, {"error": {"message": "Not found"}})
                return
            try:
                request = VlmCompletionRequest.model_validate_json(
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                )
                prompt, images = prepare_vlm_request(request, directory, context)
                response = (
                    archivist.analyze_images.remote(prompt, images)
                    if images
                    else archivist.analyze.remote(prompt)
                )
                result = ArchivistResult.model_validate_json(response)
                self.write_json(
                    200,
                    {
                        "id": f"chatcmpl-{uuid4().hex}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": result.text,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "World generation vision request failed"
                )
                self.write_json(
                    500,
                    {"error": {"message": "Unable to complete vision request"}},
                )

        def log_message(self, message: str, *arguments: object) -> None:
            logging.getLogger(__name__).info(message, *arguments)

    server = ThreadingHTTPServer(("127.0.0.1", 0), WorldGenerationVlmHandler)
    server.daemon_threads = False
    server_thread = Thread(target=server.serve_forever, name="world-generation-vlm")
    server_thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()
