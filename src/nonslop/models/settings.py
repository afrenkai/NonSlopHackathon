from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSettings:
    base_model: str = "multimodalart/MiniMax-H3-Pruned"
    original_model: str = "MiniMaxAI/MiniMax-H3"
    adapter_model: str = "Lakonik/pi-H3"
    adapter_folder: str = "pi_h3_4step_dwd_adapter"
    cache_root: str = "/models/hf"
    model_volume: str = "nonslop-models"
    width: int = 768
    height: int = 448
    frames: int = 124
    steps: int = 4
