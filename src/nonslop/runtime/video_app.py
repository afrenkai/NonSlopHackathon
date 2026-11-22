from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import modal

from nonslop.models.settings import ModelSettings
from nonslop.runtime.modal_app import build_image
from nonslop.runtime.settings import CacheSettings

settings = ModelSettings()
cache_settings = CacheSettings()
cache_volume = modal.Volume.from_name(
    cache_settings.volume_name, create_if_missing=True, version=2
)
model_volume = modal.Volume.from_name(
    settings.model_volume, create_if_missing=True, version=2
)
output_volume = modal.Volume.from_name(
    "nonslop-videos", create_if_missing=True, version=2
)
gpu_image = (
    build_image()
    .uv_pip_install("beartype", "jaxtyping", "einops")
    .env(
        {
            "HF_HOME": settings.cache_root,
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .env(cache_settings.environment())
)
app = modal.App("nonslop-video")

with gpu_image.imports():
    import torch

    from nonslop.generation.pipeline import generate_video, load_pipeline
    from nonslop.generation.schemas import GenerationTask
    from nonslop.models.processor import load_processor


@app.cls(
    image=gpu_image,
    gpu="H200",
    cpu=8,
    memory=131072,
    volumes={
        "/models": model_volume,
        "/outputs": output_volume,
        "/cache": cache_volume,
    },
    timeout=3600,
    scaledown_window=120,
    max_containers=1,
)
class StreetGenerator:
    @modal.enter()
    def load(self) -> None:
        for path in cache_settings.environment().values():
            if path.startswith("/cache/"):
                Path(path).mkdir(parents=True, exist_ok=True)
        self.pipeline, self.color_encoder = load_pipeline()

    @modal.method()
    def generate(
        self,
        task_json: str,
        reference_images: list[bytes] | None = None,
        profile: bool = False,
    ) -> str:
        task = GenerationTask.model_validate_json(task_json)
        try:
            recording = (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                )
                if profile
                else nullcontext(None)
            )
            with recording as profiler:
                result = generate_video(
                    self.pipeline,
                    self.color_encoder,
                    task,
                    reference_images or [],
                    Path("/outputs"),
                )
            if profiler is not None:
                summary = profiler.key_averages().table(
                    sort_by="self_device_time_total", row_limit=30
                )
                reports = Path("/cache/profiling")
                reports.mkdir(exist_ok=True)
                (reports / f"{task.job_id}.txt").write_text(summary)
                print(summary, flush=True)
        finally:
            cache_volume.commit()
        output_volume.commit()
        return result.model_dump_json()

    @modal.exit()
    def retain_cache(self) -> None:
        cache_volume.commit()


@app.function(image=gpu_image, volumes={"/models": model_volume}, timeout=180)
def processor_check() -> None:
    processor = load_processor()
    print(type(processor).__name__, flush=True)


@app.local_entrypoint()
def smoke_generation(profile: bool = False) -> None:
    payload = {
        "job_id": str(uuid4()),
        "seed": 42,
        "prompt": "integrated_multimodal_description: [Shot 1] A realistic street-level "
        "view of a quiet historic street with brick shopfronts. The camera moves forward "
        "slowly at walking height in a continuous shot.\n\noverall_soundscape: Soft footsteps "
        "and distant street ambience.\n\nnon_diegetic_music: N/A",
    }
    print(StreetGenerator().generate.remote(json.dumps(payload), None, profile))
