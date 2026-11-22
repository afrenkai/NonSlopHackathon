import os

import modal

from nonslop.runtime.access import configured_access
from nonslop.runtime.web_image import build_web_image

app = modal.App("nonslop-streets")
image = build_web_image()
access = configured_access()
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

    from nonslop.generation.prompt_writer import (
        PromptWriterSettings,
        compose_scene_prompt,
    )
    from nonslop.generation.schemas import GenerationResult, GenerationTask
    from nonslop.models.inference_schemas import ArchivistResult
    from nonslop.runtime.archivist import ModalArchivistGateway
    from nonslop.runtime.gateway import (
        ModalGenerationGateway,
        ModalReconstructionGateway,
    )
    from nonslop.web.app import create_app
    from nonslop.web.security import environment_settings
    from nonslop.web.uploads import normalize_image


@app.function(
    image=image, cpu=2, memory=8192, timeout=3600, max_containers=4, secrets=[access]
)
def prepare_visit(task_json: str, reference_images: list[bytes]) -> str:
    task = GenerationTask.model_validate_json(task_json)
    writer_packet = task.evidence.model_dump_json() if task.evidence else task.prompt
    observations = ""
    analysis_images = [normalize_image(content, 768) for content in reference_images]
    if analysis_images:
        observations = ArchivistResult.model_validate_json(
            modal.Cls.from_name(
                "nonslop-inference", "MemoryArchivist"
            )().analyze_images.remote(
                "Describe each photograph independently in input order. Start each "
                "short paragraph with its photo number, beginning with Photo 1 for "
                "the opening view. Name distinctive visible features with their "
                "materials or shapes and relative positions. Include lighting or "
                "legible text when useful. Mark blur, cropping, guesses, and "
                "uncertain overlap. Keep unobserved details out of the descriptions.",
                analysis_images,
            )
        ).text
        description, sound_marker, sound_fields = task.prompt.partition(
            "\n\noverall_soundscape:"
        )
        photo_context = (
            "Keep the opening photograph as the visual anchor. "
            "Additional photographs are supporting views, not an ordered camera trajectory. "
            f"AI observations of all {len(reference_images)} reference images (interpretations, not verified facts): "
            f"{observations}"
        )
        task.prompt = f"{description} {photo_context}{sound_marker}{sound_fields}"
        writer_packet = f"{photo_context}\n\n{writer_packet}"
    writer = PromptWriterSettings()
    composition = compose_scene_prompt(
        writer_packet, writer, reference_images=analysis_images
    )
    if composition is not None:
        task.prompt = composition.prompt
    result = GenerationResult.model_validate_json(
        modal.Cls.from_name("nonslop-video", "StreetGenerator")().generate.remote(
            task.model_dump_json(exclude={"evidence"}), reference_images
        )
    )
    result.reference_observations = observations
    result.used_prompt = task.prompt
    if composition is not None:
        result.prompt_model = os.environ.get(writer.model_env, writer.model)
        result.photo_readings = composition.photo_readings
    return result.model_dump_json()


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    volumes={"/data": data_volume, "/outputs": output_volume, "/worlds": world_volume},
    secrets=[access],
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
            modal.Function.from_name("nonslop-world", "prepare_reconstruction"),
            modal.Function.from_name(
                "nonslop-world-generation", "prepare_world_generation"
            ),
        ),
        ModalArchivistGateway(
            modal.Cls.from_name("nonslop-inference", "MemoryArchivist")
        ),
        data_volume.commit,
        output_volume.reload,
        world_volume.reload,
    )
