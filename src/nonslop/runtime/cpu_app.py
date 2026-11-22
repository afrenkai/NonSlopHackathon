import logging
import os

import modal

from nonslop.runtime.access import configured_access
from nonslop.runtime.web_image import build_web_image

app = modal.App("nonslop-streets")
image = build_web_image()
access = configured_access()
web_secrets = [access]
if modal.is_local() and (nous_key := os.environ.get("NOUS_API_KEY", "").strip()):
    nous_settings = {"NOUS_API_KEY": nous_key}
    if nous_model := os.environ.get("NOUS_IMAGE_MODEL", "").strip():
        nous_settings["NOUS_IMAGE_MODEL"] = nous_model
    web_secrets.append(modal.Secret.from_dict(nous_settings))
data_volume = modal.Volume.from_name(
    "nonslop-scenes", create_if_missing=True, version=2
)
output_volume = modal.Volume.from_name(
    "nonslop-videos", create_if_missing=True, version=2
)
world_volume = modal.Volume.from_name(
    "nonslop-worlds", create_if_missing=True, version=2
)

with image.imports():
    from fastapi import FastAPI

    from nonslop.generation.schemas import GenerationResult, GenerationTask
    from nonslop.models.inference_schemas import ArchivistResult
    from nonslop.runtime.archivist import ModalArchivistGateway
    from nonslop.runtime.gateway import (
        ModalGenerationGateway,
        ModalReconstructionGateway,
    )
    from nonslop.web.app import create_app
    from nonslop.web.security import environment_settings


@app.function(image=image, cpu=2, memory=8192, timeout=3600, max_containers=4)
def prepare_visit(task_json: str, reference_images: list[bytes]) -> str:
    task = GenerationTask.model_validate_json(task_json)
    observations = ""
    if len(reference_images) > 1:
        try:
            observations = ArchivistResult.model_validate_json(
                modal.Cls.from_name(
                    "nonslop-inference", "MemoryArchivist"
                )().analyze_images.remote(
                    "Compare all of these reference photographs for a scene reconstruction. "
                    "Describe shared architecture, street layout, materials, and visible signs. "
                    "Mention conflicting viewpoints or details explicitly. Do not invent dates, "
                    "identities, hidden objects, or historical facts. Keep the observations concise.",
                    reference_images,
                )
            ).text
        except modal.exception.NotFoundError:
            logging.getLogger(__name__).info(
                "Omni is unavailable for reference analysis."
            )
        description, sound_marker, sound_fields = task.prompt.partition(
            "\n\noverall_soundscape:"
        )
        context = (
            f" AI observations of all {len(reference_images)} reference images (interpretations, not verified facts): "
            f"{observations}"
            if observations
            else ""
        )
        task.prompt = (
            f"{description} Begin on the first reference photograph and use the last as the ending frame."
            f"{context}{sound_marker}{sound_fields}"
        )
    result = GenerationResult.model_validate_json(
        modal.Cls.from_name("nonslop-video", "StreetGenerator")().generate.remote(
            task.model_dump_json(), reference_images
        )
    )
    result.reference_observations = observations
    return result.model_dump_json()


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    volumes={"/data": data_volume, "/outputs": output_volume, "/worlds": world_volume},
    secrets=web_secrets,
    max_containers=1,
    scaledown_window=300,
    timeout=1800,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web() -> FastAPI:
    return create_app(
        environment_settings(),
        ModalGenerationGateway(prepare_visit),
        ModalReconstructionGateway(
            modal.Cls.from_name("nonslop-world", "WorldReconstructor")
        ),
        ModalArchivistGateway(
            modal.Cls.from_name("nonslop-inference", "MemoryArchivist")
        ),
        data_volume.commit,
        output_volume.reload,
        world_volume.reload,
    )
