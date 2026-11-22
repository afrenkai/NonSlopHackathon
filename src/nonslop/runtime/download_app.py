import json

import modal

from nonslop.models.settings import ModelSettings
from nonslop.runtime.modal_app import build_image

settings = ModelSettings()
model_volume = modal.Volume.from_name(
    settings.model_volume, create_if_missing=True, version=2
)
image = build_image().env(
    {"HF_HOME": settings.cache_root, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
)
app = modal.App("nonslop-model-download", image=image)

with image.imports():
    from nonslop.models.download import download_weights


@app.function(volumes={"/models": model_volume}, timeout=7200, memory=16384)
def download() -> dict[str, str]:
    snapshots = download_weights()
    model_volume.commit()
    return snapshots


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(download.remote(), indent=2))
