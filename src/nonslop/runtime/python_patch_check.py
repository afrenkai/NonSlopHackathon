import json
import sys
from importlib.metadata import version

import modal

from nonslop.models.settings import ModelSettings
from nonslop.runtime.modal_app import build_image


def build_patched_image() -> modal.Image:
    return build_image()


settings = ModelSettings()
image = build_patched_image().env(
    {"HF_HOME": settings.cache_root, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
)
model_volume = modal.Volume.from_name(
    settings.model_volume, create_if_missing=True, version=2
)
app = modal.App("nonslop-python-patch-check", image=image)

with image.imports():
    import mmcv
    import torch
    import triton
    from cutlass import cute
    from lakonlab.pipelines.pipeline_minimax_h3 import MiniMaxH3DWDBlocks

    from nonslop.models.processor import load_processor


@app.function(volumes={"/models": model_volume}, cpu=2, memory=8192, timeout=300)
def inspect_runtime() -> dict[str, str]:
    model_volume.reload()
    processor = load_processor()
    return {
        "python": sys.version,
        "prefix": sys.prefix,
        "executable": sys.executable,
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "mmcv": mmcv.__version__,
        "lakonlab": version("lakonlab"),
        "cutlass_dsl": version("nvidia-cutlass-dsl"),
        "cute_module": cute.__name__,
        "triton": triton.__version__,
        "pipeline": MiniMaxH3DWDBlocks.__name__,
        "processor": type(processor).__name__,
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(inspect_runtime.remote(), indent=2))
