from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class WorldSettings:
    repository_revision: str = "df9988efb87bfc0f4947eb3889411cf957478b06"
    model_repository: str = "tencent/HY-World-2.0"
    model_subfolder: str = "HY-WorldMirror-2.0"
    model_root: str = "/models/hyworld"
    cache_root: str = "/cache/world-py312-torch210-cu130"
    flash_attention_wheel: str = (
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3%2Bcu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    )

    def environment(self) -> dict[str, str]:
        return {
            "HF_HOME": "/models/hf",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TORCHINDUCTOR_CACHE_DIR": f"{self.cache_root}/torchinductor",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "TRITON_CACHE_DIR": f"{self.cache_root}/triton",
            "CUDA_CACHE_PATH": f"{self.cache_root}/cuda",
            "TORCH_EXTENSIONS_DIR": f"{self.cache_root}/torch_extensions",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "MPLBACKEND": "Agg",
        }


class WorldResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    ply_path: str
    camera_path: str
