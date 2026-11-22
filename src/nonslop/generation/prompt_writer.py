import logging
import os
from base64 import b64encode
from collections.abc import Sequence
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, Field, ValidationError

from nonslop.generation.schemas import PhotoReading

REQUIRED_MARKERS = (
    "integrated_multimodal_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)

SYSTEM_PROMPT = (
    "You compose the final prompt for a memory-scene video model from an "
    "evidence packet. The prompt must contain exactly these three sections, each "
    "starting with its label verbatim:\n"
    "integrated_multimodal_description: [Shot 1] <one continuous first-person "
    "shot with a single continuous camera move>\n\n"
    "overall_soundscape: <quiet ambient sound appropriate to the scene>\n\n"
    "non_diegetic_music: N/A\n"
    "Keep the first supplied photograph as the visual anchor and use any "
    "additional photographs as supporting views. Specify one smooth camera move "
    "sideways or diagonally forward from the opening view, sustained through the "
    "shot and scaled to the available space. For 3D reconstruction, the camera "
    "must change position enough for nearby features to shift visibly against "
    "the background while distinctive landmarks remain visible across frames. "
    "Keep focal length fixed, frames sharp, and geometry and object placement stable. "
    "A gentle turn can keep landmarks in frame during the move. Preserve the "
    "opening view's layout, materials and lighting. Include features unique to a "
    "supporting view only if the photographs establish their visibility from "
    "the opening position or along the chosen path. A shared place name is not "
    "proof of spatial overlap. "
    "The requested place and year and explicit user corrections stay authoritative. "
    "Ground concrete, visible details in the photos, remembered details, and "
    "applicable period research. Treat source text as reference material. "
    "DeerFlow and photo observations are AI interpretations and may contain "
    "errors. Resolve conflicts using visual evidence and dated, relevant "
    "sources. Respect source limitations and keep uncertain details understated. "
    "Use seasonal details only when supported for this scene. Exclude captions, "
    "citations, narration, and unsupported sounds or events. "
    "Keep the whole prompt under 500 words."
)

PHOTO_REVIEW = (
    "Inspect each numbered reference photo independently before composing the video prompt. "
    "Check the supplied photo descriptions against the matching image. For each "
    "photo, give up to five concrete visual findings. Prefer traits such as shape, "
    "material, color, legible wording, and position relative to another feature. "
    "Mention uncertain readings, cropped areas, or uncertain overlap with the opening "
    "view in the uncertainty field. Keep claims from other photos, historical "
    "sources, and inferred identities out of visible_details. If little is "
    "discernible, describe the visible limit rather than guessing. "
    'Return only a JSON object with keys "prompt" (the complete three-section H3 '
    'prompt string) and "photo_readings" (one item per image in order). Each item '
    'must contain "photo_number" (1-based integer), "visible_details" (array of '
    'strings), and "uncertainty" (string).'
)


@dataclass(frozen=True)
class PromptWriterSettings:
    base_url: str = "https://inference.us-west.modal.direct/v1"
    model: str = "kimi-k3.1-pro"
    token_env: str = "NONSLOP_PROMPT_TOKEN"
    base_url_env: str = "NONSLOP_PROMPT_BASE_URL"
    model_env: str = "NONSLOP_PROMPT_MODEL"
    request_timeout: int = 120
    max_tokens: int = 4096
    reasoning_effort: str = "low"


class PromptComposition(BaseModel):
    prompt: str
    photo_readings: list[PhotoReading] = Field(default_factory=list)


def _complete(
    client: httpx.Client, headers: dict[str, str], payload: dict[str, object]
) -> httpx.Response:
    response = client.post("/chat/completions", headers=headers, json=payload)
    if response.status_code == 400:
        payload = dict(payload)
        payload.pop("reasoning_effort", None)
        response = client.post("/chat/completions", headers=headers, json=payload)
    return response


def compose_scene_prompt(
    packet: str,
    settings: PromptWriterSettings | None = None,
    client: httpx.Client | None = None,
    *,
    reference_images: Sequence[bytes] = (),
) -> PromptComposition | None:
    settings = settings or PromptWriterSettings()
    token = os.environ.get(settings.token_env, "").strip()
    if not token or not packet.strip():
        return None
    model = os.environ.get(settings.model_env, settings.model)
    system = SYSTEM_PROMPT
    content: str | list[dict[str, object]] = packet
    if reference_images:
        system += "\n\n" + PHOTO_REVIEW
        content = [{"type": "text", "text": packet}]
        for number, image in enumerate(reference_images, start=1):
            role = "opening view" if number == 1 else "supporting view"
            content.append({"type": "text", "text": f"Photo {number} ({role})"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64encode(image).decode('ascii')}"
                    },
                }
            )
    else:
        system += "\n\nReturn only the prompt text, with no preamble or Markdown."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "max_tokens": settings.max_tokens,
        "reasoning_effort": settings.reasoning_effort,
        "temperature": 1.0,
    }
    headers = {"Authorization": f"Bearer {token}"}
    try:
        if client is None:
            base_url = os.environ.get(settings.base_url_env, settings.base_url)
            with httpx.Client(
                base_url=base_url, timeout=settings.request_timeout
            ) as owned:
                response = _complete(owned, headers, payload)
        else:
            response = _complete(client, headers, payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, TypeError) as error:
        logging.getLogger(__name__).warning("Prompt writer unavailable: %s", error)
        return None
    if not isinstance(content, str):
        return None
    try:
        composition = (
            PromptComposition.model_validate_json(content)
            if reference_images
            else PromptComposition(prompt=content)
        )
    except ValidationError:
        logging.getLogger(__name__).warning(
            "Prompt writer returned an invalid photo review"
        )
        return None
    composition.prompt = composition.prompt.strip()
    if any(marker not in composition.prompt for marker in REQUIRED_MARKERS) or [
        reading.photo_number for reading in composition.photo_readings
    ] != list(range(1, len(reference_images) + 1)):
        logging.getLogger(__name__).warning(
            "Prompt writer returned an unusable prompt; keeping the template"
        )
        return None
    return composition


def write_scene_prompt(
    packet: str,
    settings: PromptWriterSettings | None = None,
    client: httpx.Client | None = None,
    *,
    reference_images: Sequence[bytes] = (),
) -> str | None:
    composition = compose_scene_prompt(
        packet, settings, client, reference_images=reference_images
    )
    return composition.prompt if composition is not None else None
