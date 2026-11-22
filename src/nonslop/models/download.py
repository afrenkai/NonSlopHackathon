from huggingface_hub import snapshot_download

from nonslop.models.settings import ModelSettings


def download_weights() -> dict[str, str]:
    settings = ModelSettings()
    repositories = {
        settings.base_model: [
            "modular_model_index.json",
            "text_encoder/*",
            "transformer/*",
        ],
        settings.original_model: [
            "tokenizer/*",
            "processor/*",
            "vae/*",
            "audio_vae/*",
            "scheduler/*",
            "audio_scheduler/*",
        ],
        settings.adapter_model: [f"{settings.adapter_folder}/*"],
    }
    snapshots: dict[str, str] = {}
    for repository, patterns in repositories.items():
        print(f"Downloading {repository}: {patterns}", flush=True)
        snapshots[repository] = snapshot_download(
            repository, allow_patterns=patterns, max_workers=8
        )
    return snapshots
