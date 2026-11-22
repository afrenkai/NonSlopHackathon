from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import modal

from nonslop.models.inference_schemas import ArchivistResult
from nonslop.models.world_schemas import CameraCalibration, WorldResult, WorldSettings
from nonslop.reconstruction.context_schemas import ReconstructionContext
from nonslop.reconstruction.review_schemas import ReconstructionReview


def build_world_image(settings: WorldSettings) -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04", add_python="3.12"
        )
        .entrypoint([])
        .apt_install("git", "build-essential", "ffmpeg")
        .uv_pip_install("setuptools", "wheel", "packaging", "uv")
        .uv_pip_install(
            "torch==2.10.0",
            "torchvision==0.25.0",
            index_url="https://download.pytorch.org/whl/cu130",
        )
        .uv_pip_install(
            "numpy==1.26.4",
            "scipy==1.14.1",
            "matplotlib==3.10.3",
            "opencv-python-headless==4.10.0.84",
            "Pillow",
            "omegaconf",
            "safetensors",
            "huggingface_hub",
            "einops",
            "plyfile",
            "trimesh",
            "requests",
            "tqdm",
            "pydantic",
            "beartype",
            "jaxtyping",
        )
        .uv_pip_install(settings.flash_attention_wheel)
        .run_commands(
            "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git "
            "/opt/HY-World-2.0",
            f"git -C /opt/HY-World-2.0 checkout {settings.repository_revision}",
        )
        .add_local_file(
            Path(__file__).with_name("worldmirror-fidelity.patch"),
            "/opt/worldmirror-fidelity.patch",
            copy=True,
        )
        .run_commands(
            "git -C /opt/HY-World-2.0 apply /opt/worldmirror-fidelity.patch",
            "BUILD_NO_CUDA=1 uv pip install --system --no-build-isolation "
            "/opt/HY-World-2.0/hyworld2/worldgen/third_party/gsplat_maskgaussian",
            "printf '%s\\n' '[build-system]' 'requires = [\"setuptools\"]' "
            "'build-backend = \"setuptools.build_meta\"' '[project]' "
            "'name = \"nonslop-hyworld-source\"' 'version = \"0.0.0\"' "
            "'[tool.setuptools.packages.find]' "
            '\'include = ["hyworld2", "hyworld2.worldrecon*"]\' '
            "> /opt/HY-World-2.0/pyproject.toml",
            "uv pip install --system --no-deps --no-build-isolation /opt/HY-World-2.0",
        )
        .env(settings.environment())
        .workdir("/opt/HY-World-2.0")
    )


settings = WorldSettings()
model_volume = modal.Volume.from_name(
    "nonslop-models", create_if_missing=True, version=2
)
video_volume = modal.Volume.from_name(
    "nonslop-videos", create_if_missing=True, version=2
)
world_volume = modal.Volume.from_name(
    "nonslop-worlds", create_if_missing=True, version=2
)
cache_volume = modal.Volume.from_name(
    "nonslop-kernel-cache", create_if_missing=True, version=2
)
world_image = build_world_image(settings)
download_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "huggingface_hub", "pydantic"
)
app = modal.App("nonslop-world")

with world_image.imports():
    from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import (
        compute_adaptive_target_size,
        compute_preprocessing_transform,
        prepare_input,
    )
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

    from nonslop.reconstruction.coverage import write_coverage_views
    from nonslop.reconstruction.images import (
        prepare_reference_images,
        prepare_review_image,
    )
    from nonslop.reconstruction.prompts import build_reconstruction_review_prompt

with download_image.imports():
    from huggingface_hub import snapshot_download


@app.function(
    image=download_image,
    cpu=2,
    memory=8192,
    volumes={"/models": model_volume},
    timeout=7200,
)
def download_models() -> str:
    model_volume.reload()
    try:
        return snapshot_download(
            repo_id=settings.model_repository,
            local_dir=settings.model_root,
            allow_patterns=[
                f"{settings.model_subfolder}/config.json",
                f"{settings.model_subfolder}/model.safetensors",
            ],
        )
    finally:
        model_volume.commit()


@app.function(
    image=world_image,
    cpu=2,
    memory=8192,
    volumes={"/outputs": video_volume},
    timeout=1800,
    max_containers=2,
)
def prepare_reconstruction(
    job_id: str, reference_images: list[bytes], context_json: str
) -> str:
    identifier = UUID(job_id)
    context = ReconstructionContext.model_validate_json(context_json)
    video_volume.reload()
    input_path = Path("/outputs") / f"{context.video_job_id or identifier}.mp4"
    with TemporaryDirectory(prefix=f"review-{identifier}-") as temporary:
        extracted = prepare_input(
            str(input_path),
            video_strategy="old",
            fps=1,
            max_frames=settings.max_video_frames,
            temp_dir=temporary,
        )
        frames = [Path(path).read_bytes() for path in extracted[0]]
    review_images = reference_images[:1] + frames
    response = ArchivistResult.model_validate_json(
        modal.Cls.from_name(
            "nonslop-inference", "MemoryArchivist"
        )().analyze_images.remote(
            build_reconstruction_review_prompt(
                context, len(frames), bool(reference_images)
            ),
            [prepare_review_image(content, 768) for content in review_images],
        )
    )
    review = ReconstructionReview.model_validate_json(
        response.text.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    if sorted(decision.frame_index for decision in review.frames) != list(
        range(len(frames))
    ):
        raise ValueError(
            "The reconstruction review must assess every candidate frame exactly once"
        )
    selected = [
        frames[decision.frame_index]
        for decision in sorted(review.frames, key=lambda decision: decision.frame_index)
        if decision.usable
    ]
    if not selected:
        raise ValueError(
            f"No generated views were consistent with the reconstruction context: {review.summary}"
        )
    print(
        f"Reconstruction context review: {len(reference_images)} photos, {len(selected)}/{len(frames)} H3 views, {len(context.sources)} sources; {review.environment}",
        flush=True,
    )
    return WorldReconstructor().reconstruct.remote(
        job_id,
        reference_images,
        selected,
        context.model_dump_json(),
        review.model_dump_json(),
    )


@app.cls(
    image=world_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    volumes={
        "/models": model_volume,
        "/worlds": world_volume,
        "/cache": cache_volume,
    },
    timeout=1800,
    scaledown_window=120,
    max_containers=1,
)
class WorldReconstructor:
    @modal.enter()
    def load(self) -> None:
        model_volume.reload()
        for path in settings.environment().values():
            if path.startswith("/cache/"):
                Path(path).mkdir(parents=True, exist_ok=True)
        model_directory = Path(settings.model_root) / settings.model_subfolder
        for filename in ("config.json", "model.safetensors"):
            if not (model_directory / filename).is_file():
                raise FileNotFoundError(
                    f"Missing WorldMirror artifact {model_directory / filename}; "
                    "run the download entrypoint first."
                )
        self.pipeline = WorldMirrorPipeline.from_pretrained(
            str(model_directory), enable_bf16=True
        )

    @modal.method()
    def reconstruct(
        self,
        job_id: str,
        reference_images: list[bytes],
        generated_images: list[bytes],
        context_json: str,
        review_json: str,
    ) -> str:
        identifier = UUID(job_id)
        context = ReconstructionContext.model_validate_json(context_json)
        review = ReconstructionReview.model_validate_json(review_json)
        assert generated_images and len(generated_images) == sum(
            decision.usable for decision in review.frames
        )
        world_volume.reload()
        output_path = Path("/worlds") / str(identifier)
        try:
            with TemporaryDirectory(prefix=f"memory-{identifier}-") as temporary:
                frame_directory = Path(temporary) / "frames"
                references = reference_images
                views = references + generated_images
                image_paths = [
                    str(path)
                    for path in prepare_reference_images(
                        views, frame_directory, settings.target_size
                    )
                ]
                effective_size = compute_adaptive_target_size(
                    image_paths, settings.target_size
                )
                transform = compute_preprocessing_transform(image_paths, effective_size)
                result_directory = self.pipeline(
                    str(frame_directory),
                    strict_output_path=str(output_path),
                    target_size=settings.target_size,
                    save_gs=True,
                    save_camera=True,
                    save_points=False,
                    save_depth=True,
                    save_sky_mask=True,
                    save_normal=False,
                    sky_mask_source="model",
                    apply_sky_mask=review.environment != "indoor",
                    edge_normal_threshold=settings.edge_normal_threshold,
                    edge_depth_threshold=settings.edge_depth_threshold,
                    save_rendered=False,
                    compress_gs_max_points=settings.max_splats,
                )
                if result_directory is not None:
                    calibration = CameraCalibration.model_validate_json(
                        (output_path / "camera_params.json").read_text()
                    )
                    calibration.image_width = transform["final_w"]
                    calibration.image_height = transform["final_h"]
                    write_coverage_views(
                        output_path,
                        image_paths,
                        calibration,
                        use_sky_mask=review.environment != "indoor",
                    )
            if result_directory is None:
                raise RuntimeError("WorldMirror skipped these input views")
            ply_path = output_path / "gaussians.ply"
            camera_path = output_path / "camera_params.json"
            for artifact in (ply_path, camera_path):
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise RuntimeError(f"WorldMirror did not produce {artifact}")
            calibration.photo_count = len(references)
            calibration.generated_frame_count = len(generated_images)
            camera_path.write_text(calibration.model_dump_json())
            (output_path / "reconstruction_context.json").write_text(
                context.model_dump_json()
            )
            (output_path / "reconstruction_review.json").write_text(
                review.model_dump_json()
            )
            return WorldResult(
                job_id=identifier,
                ply_path=str(ply_path),
                camera_path=str(camera_path),
                input_kind=(
                    "photos_and_generated_video" if references else "generated_video"
                ),
                input_count=calibration.num_cameras,
                photo_count=calibration.photo_count,
                generated_frame_count=calibration.generated_frame_count,
                context_applied=True,
                evidence_count=len(context.sources),
                review=review,
                coverage_available=calibration.coverage_available,
            ).model_dump_json()
        finally:
            world_volume.commit()
            cache_volume.commit()

    @modal.exit()
    def retain_cache(self) -> None:
        cache_volume.commit()


@app.local_entrypoint()
def main(
    job_id: str = "",
    download: bool = False,
    context_path: str = "",
    reference_directory: str = "",
) -> None:
    if download:
        print(download_models.remote())
    elif job_id and context_path:
        references = (
            [
                path.read_bytes()
                for path in sorted(Path(reference_directory).iterdir())
                if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            ]
            if reference_directory
            else []
        )
        print(
            prepare_reconstruction.remote(
                job_id, references, Path(context_path).read_text()
            )
        )
    else:
        raise ValueError(
            "Use --download or --job-id <uuid> --context-path <context.json>"
        )
