from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from uuid import UUID

import modal

from nonslop.models.world_schemas import CameraCalibration, WorldResult, WorldSettings


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

    from nonslop.reconstruction.images import prepare_reference_images

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


@app.cls(
    image=world_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    volumes={
        "/models": model_volume,
        "/outputs": video_volume,
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
        self, job_id: str, reference_images: list[bytes] | None = None
    ) -> str:
        identifier = UUID(job_id)
        world_volume.reload()
        input_path = Path("/outputs") / f"{identifier}.mp4"
        output_path = Path("/worlds") / str(identifier)
        try:
            with TemporaryDirectory(prefix=f"memory-{identifier}-") as temporary:
                frame_directory = Path(temporary) / "frames"
                if reference_images:
                    image_paths = [
                        str(path)
                        for path in prepare_reference_images(
                            reference_images, frame_directory
                        )
                    ]
                else:
                    video_volume.reload()
                    if not input_path.is_file():
                        raise FileNotFoundError(
                            f"No generated video exists at {input_path}"
                        )
                    extracted = prepare_input(
                        str(input_path),
                        video_strategy="old",
                        fps=1,
                        max_frames=settings.max_video_frames,
                        temp_dir=temporary,
                    )
                    frame_directory.mkdir()
                    image_paths = []
                    for index, frame_path in enumerate(extracted[0]):
                        destination = frame_directory / f"{index:04d}.png"
                        copyfile(frame_path, destination)
                        image_paths.append(str(destination))
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
                    save_depth=False,
                    save_normal=False,
                    sky_mask_source="model",
                    edge_normal_threshold=settings.edge_normal_threshold,
                    edge_depth_threshold=settings.edge_depth_threshold,
                    save_rendered=False,
                    compress_gs_max_points=settings.max_splats,
                )
            if result_directory is None:
                raise RuntimeError("WorldMirror skipped these input views")
            ply_path = output_path / "gaussians.ply"
            camera_path = output_path / "camera_params.json"
            for artifact in (ply_path, camera_path):
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise RuntimeError(f"WorldMirror did not produce {artifact}")
            calibration = CameraCalibration.model_validate_json(camera_path.read_text())
            calibration.image_width = transform["final_w"]
            calibration.image_height = transform["final_h"]
            camera_path.write_text(calibration.model_dump_json())
            return WorldResult(
                job_id=identifier,
                ply_path=str(ply_path),
                camera_path=str(camera_path),
                input_kind=(
                    "reference_photos" if reference_images else "generated_video"
                ),
                input_count=calibration.num_cameras,
            ).model_dump_json()
        finally:
            world_volume.commit()
            cache_volume.commit()

    @modal.exit()
    def retain_cache(self) -> None:
        cache_volume.commit()


@app.local_entrypoint()
def main(job_id: str = "", download: bool = False) -> None:
    if download:
        print(download_models.remote())
    elif job_id:
        print(WorldReconstructor().reconstruct.remote(job_id))
    else:
        raise ValueError("Use --download or --job-id <generated-video-uuid>")
