from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MediaType = Literal[
    "",
    "image/jpeg",
    "image/png",
    "image/webp",
    "audio/wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/ogg",
    "audio/flac",
    "video/mp4",
    "video/webm",
    "video/quicktime",
]
ArchivistImage = Annotated[bytes, Field(min_length=1, max_length=10485760)]


class ArchivistRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=12000)
    media: bytes | None = Field(default=None, max_length=33554432)
    media_type: MediaType = ""

    @model_validator(mode="after")
    def validate_media(self) -> ArchivistRequest:
        if (self.media is None) != (self.media_type == ""):
            raise ValueError("Provide media bytes and their media_type together")
        if self.media is not None and not self.media:
            raise ValueError("Media must not be empty")
        return self


class ArchivistResult(BaseModel):
    text: str


class ArchivistImagesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=12000)
    images: list[ArchivistImage] = Field(min_length=1, max_length=8)


@dataclass(frozen=True)
class InferenceSettings:
    image_repository: str = "hongccc/sglang-omni:dev"
    source_revision: str = "a70a4c6339819067a51bffabcecde9574b684a97"
    model_repository: str = "marksverdhei/Qwen3-Omni-30B-A3B-FP8"
    model_volume: str = "nonslop-models"
    cache_volume: str = "nonslop-kernel-cache"
    hf_home: str = "/models/hf"
    cache_root: str = "/cache/omni"
    server_url: str = "http://127.0.0.1:8000"
    startup_timeout: int = 1800
    request_timeout: int = 600
    scaledown_window: int = 900
    context_length: int = 8192
    max_tokens: int = 1024

    def environment(self) -> dict[str, str]:
        return {
            "HF_HOME": self.hf_home,
            "HF_HUB_CACHE": f"{self.hf_home}/hub",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TORCHINDUCTOR_CACHE_DIR": f"{self.cache_root}/torchinductor",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "TRITON_CACHE_DIR": f"{self.cache_root}/triton",
            "CUDA_CACHE_PATH": f"{self.cache_root}/cuda",
            "TORCH_EXTENSIONS_DIR": f"{self.cache_root}/torch_extensions",
            "FLASHINFER_WORKSPACE_BASE": f"{self.cache_root}/flashinfer",
            "CUTE_DSL_CACHE_DIR": f"{self.cache_root}/cute",
            "DG_JIT_CACHE_DIR": f"{self.cache_root}/deep_gemm",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
            "SGLANG_OMNI_STARTUP_TIMEOUT": str(self.startup_timeout),
            "SGLANG_OMNI_AUTO_CLONE": "0",
        }

    def process_environment(self) -> dict[str, str]:
        return {
            **self.environment(),
            "XDG_CACHE_HOME": f"{self.cache_root}/xdg",
        }

    def pipeline_configuration(self, model_path: str) -> dict[str, object]:
        return {
            "config_cls": "Qwen3OmniPipelineConfig",
            "name": "memory-archivist",
            "model_path": model_path,
            "stages": {
                "preprocessing": {
                    "factory": {
                        "max_seq_len": self.context_length,
                        "video_fps": 1.0,
                        "video_max_frames": 24,
                        "video_max_pixels": 307200,
                        "video_total_pixels": 7372800,
                    }
                },
                "image_encoder": {"gpu_memory_fraction": 0.025},
                "audio_encoder": {"gpu_memory_fraction": 0.025},
                "thinker": {
                    "gpu_memory_fraction": 0.75,
                    "factory": {"max_seq_len": self.context_length},
                    "engine": {
                        "max_running_requests": 1,
                        "cuda_graph_max_bs": 1,
                    },
                },
            },
        }
