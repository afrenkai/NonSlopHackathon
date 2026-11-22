from pathlib import Path

import modal

from nonslop.models.world_generation_schemas import WorldGenerationSettings
from nonslop.models.world_schemas import WorldSettings
from nonslop.runtime.world_app import build_world_image

settings = WorldGenerationSettings()
app = modal.App("nonslop-world-assets")
model_volume = modal.Volume.from_name(
    "nonslop-models", create_if_missing=True, version=2
)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "modelscope", "pydantic", "huggingface_hub"
)
sam3_image = build_world_image(WorldSettings()).uv_pip_install(
    "transformers==5.2.0", "accelerate"
)

with image.imports():
    from huggingface_hub import snapshot_download as download_huggingface_snapshot
    from modelscope.hub.snapshot_download import snapshot_download

with sam3_image.imports():
    from io import BytesIO

    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor


@app.function(
    image=image, cpu=4, memory=8192, volumes={"/models": model_volume}, timeout=3600
)
def download_sam3() -> str:
    model_volume.reload()
    try:
        directory = snapshot_download(
            settings.sam3_repository,
            local_dir=settings.sam3_directory,
            ignore_file_pattern=["sam3.pt"],
        )
        for filename in (
            "model.safetensors",
            "config.json",
            "processor_config.json",
            "tokenizer.json",
            "LICENSE",
        ):
            if not (Path(directory) / filename).is_file():
                raise FileNotFoundError(
                    f"The ModelScope SAM3 snapshot is missing {filename}"
                )
        return directory
    finally:
        model_volume.commit()


@app.function(
    image=image, cpu=4, memory=8192, volumes={"/models": model_volume}, timeout=7200
)
def download_world_models() -> dict[str, str]:
    model_volume.reload()
    snapshots: dict[str, str] = {}
    try:
        for snapshot in settings.model_snapshots():
            print(f"Prefetch native world model: {snapshot.repository}", flush=True)
            snapshots[snapshot.repository] = download_huggingface_snapshot(
                repo_id=snapshot.repository,
                cache_dir=settings.hub_cache_directory,
                allow_patterns=list(snapshot.allow_patterns),
                max_workers=settings.download_workers,
            )
        return snapshots
    finally:
        model_volume.commit()


@app.function(
    image=sam3_image,
    gpu="H100",
    cpu=4,
    memory=16384,
    volumes={"/models": model_volume},
    timeout=600,
)
def segment_image(content: bytes, prompt: str) -> dict[str, str | int]:
    model_volume.reload()
    model = (
        Sam3Model.from_pretrained(settings.sam3_directory, local_files_only=True)
        .eval()
        .to("cuda")
    )
    processor = Sam3Processor.from_pretrained(
        settings.sam3_directory, local_files_only=True
    )
    with Image.open(BytesIO(content)) as source:
        inputs = processor(
            images=source.convert("RGB"), text=prompt, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=0.4, mask_threshold=0.5, target_sizes=[source.size[::-1]]
        )
    return {"model": "ModelScope facebook/sam3", "instances": len(results[0]["scores"])}


@app.local_entrypoint()
def main(image_path: str = "", prompt: str = "mountain", native: bool = False) -> None:
    if native:
        print(download_world_models.remote())
    elif image_path:
        print(segment_image.remote(Path(image_path).read_bytes(), prompt))
    else:
        print(download_sam3.remote())
