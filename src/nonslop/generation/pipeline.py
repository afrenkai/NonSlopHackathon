from io import BytesIO
from math import sqrt
from pathlib import Path

import torch
from beartype import beartype
from diffusers import ModularPipeline
from diffusers.utils import encode_video
from einops import rearrange
from jaxtyping import Float, jaxtyped
from lakonlab.models.architectures import RGBColorEncoder
from lakonlab.models.architectures.diffusers.attention import (
    default_attention_backend,
    set_attention_backends,
)
from lakonlab.models.architectures.diffusers.minimax_h3 import MiniMaxH3SolAttnProcessor
from lakonlab.models.diffusions.schedulers import FlowEulerODEScheduler
from lakonlab.ops import quantize_frozen_linears
from lakonlab.ops.quant import FrozenFP8Linear
from lakonlab.pipelines.pipeline_minimax_h3 import MiniMaxH3DWDBlocks
from lakonlab.pipelines.utils import load_lakonlab_adapter
from PIL import Image, ImageOps

from nonslop.generation.schemas import GenerationResult, GenerationTask
from nonslop.models.normalization import align_refiner_norms
from nonslop.models.processor import load_processor
from nonslop.models.settings import ModelSettings


@torch.inference_mode()
def load_pipeline() -> tuple[ModularPipeline, RGBColorEncoder]:
    settings = ModelSettings()
    print("Loading pi-H3 components from the model volume", flush=True)
    pipeline = MiniMaxH3DWDBlocks().init_pipeline(settings.base_model)
    pipeline.load_components(
        names=[
            "text_encoder",
            "tokenizer",
            "vae",
            "audio_vae",
            "transformer",
        ],
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    pipeline.update_components(processor=load_processor())
    load_lakonlab_adapter(
        pipeline,
        settings.adapter_model,
        subfolder=settings.adapter_folder,
        local_files_only=True,
    )
    normalized_layers = align_refiner_norms(pipeline.transformer)
    print(f"Aligned {normalized_layers} refiner RMSNorm layers to FP32", flush=True)
    pipeline.transformer.requires_grad_(False)
    quantize_frozen_linears(
        pipeline.transformer,
        type="fp8",
        exclude_keys=["lora", "norm_out", "proj_out", "dwd_linear_head"],
    )
    pipeline.text_encoder.requires_grad_(False)
    quantize_frozen_linears(pipeline.text_encoder, type="fp8", exclude_keys=["lm_head"])
    for name, component in (
        ("transformer", pipeline.transformer),
        ("text_encoder", pipeline.text_encoder),
    ):
        layers = sum(
            isinstance(layer, FrozenFP8Linear) for layer in component.modules()
        )
        print(f"{name}: {layers} FP8 W8A8 linear layers", flush=True)
    torch.cuda.empty_cache()
    set_attention_backends(
        pipeline.transformer,
        backend=default_attention_backend("dense"),
        overrides={"self.dwd_pixel_refiner": default_attention_backend("varlen")},
    )
    for block in pipeline.transformer.transformer_blocks[2:]:
        block.attn.set_processor(MiniMaxH3SolAttnProcessor())
        block.attn.set_attention_backend(default_attention_backend("short"))
    pipeline.update_components(
        scheduler=FlowEulerODEScheduler(
            num_train_timesteps=1, shift=12.0, max_raw_t=0.96, final_step_size_scale=0.6
        ),
        audio_scheduler=FlowEulerODEScheduler(
            num_train_timesteps=1, shift=3.0, max_raw_t=0.96, final_step_size_scale=0.6
        ),
    )
    pipeline.to("cuda")
    for block in pipeline.transformer.transformer_blocks:
        block.compile(backend="inductor", fullgraph=True)
    print(
        f"Model GPU allocation: {torch.cuda.memory_allocated() / (1024 ** 3):.1f} GiB",
        flush=True,
    )
    color_encoder = RGBColorEncoder(
        use_affine_norm=True,
        mean=(-0.03, -0.088, -0.188),
        std=(0.458, 0.448, 0.45),
    ).cuda()
    print("pi-H3 is ready", flush=True)
    return pipeline, color_encoder


@jaxtyped(typechecker=beartype)
def video_frames(
    video: Float[torch.Tensor, "1 3 frames height width"],
) -> Float[torch.Tensor, "frames height width 3"]:
    return (
        rearrange(
            video, "1 channels frames height width -> frames height width channels"
        )
        .float()
        .cpu()
    )


@torch.inference_mode()
def reference_frame(content: bytes, width: int, height: int) -> Image.Image:
    with Image.open(BytesIO(content)) as source:
        return ImageOps.fit(
            source.convert("RGB"), (width, height), Image.Resampling.LANCZOS
        )


def reference_video_size(content: bytes, settings: ModelSettings) -> tuple[int, int]:
    with Image.open(BytesIO(content)) as source:
        scale = sqrt(settings.width * settings.height / (source.width * source.height))
        width = max(
            settings.spatial_alignment,
            round(source.width * scale / settings.spatial_alignment)
            * settings.spatial_alignment,
        )
        height = max(
            settings.spatial_alignment,
            round(source.height * scale / settings.spatial_alignment)
            * settings.spatial_alignment,
        )
        return width, height


@torch.inference_mode()
def generate_video(
    pipeline: ModularPipeline,
    color_encoder: RGBColorEncoder,
    task: GenerationTask,
    reference_images: list[bytes],
    output_root: Path,
) -> GenerationResult:
    settings = ModelSettings()
    width, height = (
        reference_video_size(reference_images[0], settings)
        if reference_images
        else (settings.width, settings.height)
    )
    image = (
        reference_frame(reference_images[0], width, height)
        if reference_images
        else None
    )
    output = pipeline(
        prompt=task.prompt,
        image=image,
        height=height,
        width=width,
        num_frames=settings.frames,
        num_inference_steps=settings.steps,
        generator=torch.Generator().manual_seed(task.seed),
        output=["dwd_pixels", "audio", "sampling_rate"],
    )
    video = (
        color_encoder.decode(output.pop("dwd_pixels")).add_(1).mul_(0.5).clamp_(0, 1)
    )
    frames = video_frames(video)
    frame_count = frames.shape[0]
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{task.job_id}.mp4"
    temporary = output_root / f"{task.job_id}.partial.mp4"
    encode_video(
        frames.numpy(),
        fps=24,
        output_path=str(temporary),
        audio=output["audio"][0],
        audio_sample_rate=output["sampling_rate"],
    )
    temporary.replace(path)
    del output, video, frames
    torch.cuda.empty_cache()
    return GenerationResult(
        video_path=str(path),
        width=width,
        height=height,
        frames=frame_count,
    )
