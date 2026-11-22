from transformers import (
    AutoImageProcessor,
    AutoVideoProcessor,
    Qwen2TokenizerFast,
    Qwen3VLProcessor,
)

from nonslop.models.settings import ModelSettings


def load_processor() -> Qwen3VLProcessor:
    settings = ModelSettings()
    return Qwen3VLProcessor(
        image_processor=AutoImageProcessor.from_pretrained(
            settings.original_model,
            subfolder="processor",
            local_files_only=True,
        ),
        tokenizer=Qwen2TokenizerFast.from_pretrained(
            settings.original_model,
            subfolder="tokenizer",
            local_files_only=True,
        ),
        video_processor=AutoVideoProcessor.from_pretrained(
            settings.original_model,
            subfolder="processor",
            local_files_only=True,
        ),
    )
