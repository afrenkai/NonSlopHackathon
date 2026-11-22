from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from nonslop.reconstruction.review_schemas import ReconstructionReview

ReconstructionInput = Literal[
    "reference_photos", "generated_video", "photos_and_generated_video"
]


@dataclass(frozen=True)
class WorldSettings:
    repository_revision: str = "df9988efb87bfc0f4947eb3889411cf957478b06"
    model_repository: str = "tencent/HY-World-2.0"
    model_subfolder: str = "HY-WorldMirror-2.0"
    model_root: str = "/models/hyworld"
    cache_root: str = "/cache/world-py312-torch210-cu130"
    target_size: int = 756
    max_video_frames: int = 6
    max_splats: int = 1_500_000
    edge_normal_threshold: float = 5.0
    edge_depth_threshold: float = 0.05
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
    input_kind: ReconstructionInput = "generated_video"
    input_count: int = 0
    photo_count: int = 0
    generated_frame_count: int = 0
    context_applied: bool = False
    evidence_count: int = 0
    review: ReconstructionReview | None = None
    coverage_available: bool = False
    pipeline_kind: Literal["worldmirror_reconstruction", "hy_world_generation"] = (
        "worldmirror_reconstruction"
    )


class CameraMatrix(BaseModel):
    camera_id: int
    matrix: list[list[float]]


class CameraCalibration(BaseModel):
    num_cameras: int
    extrinsics: list[CameraMatrix]
    intrinsics: list[CameraMatrix]
    image_width: int | None = Field(default=None, gt=0)
    image_height: int | None = Field(default=None, gt=0)
    photo_count: int = 0
    generated_frame_count: int = 0
    coverage_available: bool = False
    scene_depth: float | None = Field(default=None, gt=0)
    background_rgb: list[float] | None = Field(default=None, min_length=3, max_length=3)
