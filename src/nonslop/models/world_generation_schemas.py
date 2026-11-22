from dataclasses import dataclass


@dataclass(frozen=True)
class WorldModelSnapshot:
    repository: str
    allow_patterns: tuple[str, ...]


@dataclass(frozen=True)
class WorldGenerationSettings:
    sam3_repository: str = "facebook/sam3"
    sam3_directory: str = "/models/worldgen/sam3"
    source_revision: str = "df9988efb87bfc0f4947eb3889411cf957478b06"
    source_root: str = "/opt/HY-World-2.0"
    panorama_repository: str = "Qwen/Qwen-Image-Edit-2509"
    panorama_lora_repository: str = "tencent/HY-World-2.0"
    panorama_lora_subfolder: str = "HY-Pano-2.0"
    panorama_steps: int = 20
    panorama_height: int = 480
    panorama_width: int = 992
    trajectory_resolution: int = 480
    trajectory_frames: int = 21
    split_view_count: int = 3
    wonder_topk: int = 1
    reconstruction_topk: int = 1
    vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    stereo_model: str = "worldstereo-memory-dmd"
    training_steps: int = 8000
    hub_cache_directory: str = "/models/hf/hub"
    download_workers: int = 8

    def model_snapshots(self) -> tuple[WorldModelSnapshot, ...]:
        return (
            WorldModelSnapshot(
                self.panorama_repository,
                (
                    "model_index.json",
                    "transformer/*.json",
                    "transformer/*.safetensors",
                    "text_encoder/*.json",
                    "text_encoder/*.safetensors",
                    "tokenizer/*",
                    "processor/*",
                    "vae/*.json",
                    "vae/*.safetensors",
                    "scheduler/*",
                ),
            ),
            WorldModelSnapshot(
                self.panorama_lora_repository,
                (f"{self.panorama_lora_subfolder}/pytorch_lora_weights.safetensors",),
            ),
            WorldModelSnapshot("naver-iv/zim-anything-vitl", ("zim_vit_l_2092/*",)),
            WorldModelSnapshot(
                "IDEA-Research/grounding-dino-tiny",
                ("*.json", "*.txt", "model.safetensors"),
            ),
            WorldModelSnapshot("Ruicheng/moge-2-vitl-normal", ("model.pt",)),
            WorldModelSnapshot(
                "hanshanxue/WorldStereo",
                (
                    f"{self.stereo_model}/config.json",
                    f"{self.stereo_model}/model.safetensors",
                ),
            ),
            WorldModelSnapshot(
                "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                (
                    "model_index.json",
                    "transformer/*.json",
                    "transformer/*.safetensors",
                    "text_encoder/*.json",
                    "text_encoder/*.safetensors",
                    "tokenizer/*",
                    "image_encoder/*.json",
                    "image_encoder/*.safetensors",
                    "image_processor/*",
                    "vae/*.json",
                    "vae/*.safetensors",
                    "scheduler/*",
                ),
            ),
            WorldModelSnapshot("facebook/dinov2-base", ("*.json", "model.safetensors")),
        )
