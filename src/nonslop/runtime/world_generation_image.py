from pathlib import Path
from shlex import quote

import modal

from nonslop.models.world_generation_schemas import WorldGenerationSettings


def build_world_generation_image(settings: WorldGenerationSettings) -> modal.Image:
    source_directory = "/opt/HY-World-2.0"
    worldgen_directory = f"{source_directory}/hyworld2/worldgen"
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11"
        )
        .entrypoint([])
        .apt_install(
            "git",
            "build-essential",
            "ffmpeg",
            "cmake",
            "ninja-build",
            "libgl1",
            "libegl1",
            "libglib2.0-0",
            "libglm-dev",
            "libgomp1",
        )
        .env(
            {
                "CUDA_HOME": "/usr/local/cuda",
                "FORCE_CUDA": "1",
                "TORCH_CUDA_ARCH_LIST": "9.0",
                "MAX_JOBS": "4",
                "SAM3_MODEL_PATH": settings.sam3_directory,
                "HF_HOME": "/models/hf",
                "HF_HUB_CACHE": "/models/hf/hub",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "XDG_CACHE_HOME": "/cache/worldgen",
                "TORCH_HOME": "/cache/worldgen/torch",
                "TORCH_EXTENSIONS_DIR": "/cache/worldgen/torch_extensions",
                "TORCHINDUCTOR_CACHE_DIR": "/cache/worldgen/torchinductor",
                "TRITON_CACHE_DIR": "/cache/worldgen/triton",
                "NUMBA_CACHE_DIR": "/cache/worldgen/numba",
                "UV_CACHE_DIR": "/cache/worldgen/uv",
                "TOKENIZERS_PARALLELISM": "false",
                "MPLBACKEND": "Agg",
            }
        )
        .uv_pip_install("setuptools<81", "wheel", "packaging", "uv", "ninja")
        .uv_pip_install(
            "torch==2.7.1",
            "torchvision==0.22.1",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .run_commands(
            "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git "
            f"{source_directory}",
            f"git -C {source_directory} checkout {quote(settings.source_revision)}",
            f"git -C {source_directory} submodule update --init --recursive",
        )
        .add_local_file(
            Path(__file__).with_name("worldgen-modelscope.patch"),
            "/opt/worldgen-modelscope.patch",
            copy=True,
        )
        .run_commands(
            f"git -C {source_directory} apply /opt/worldgen-modelscope.patch",
            "sed 's/^cupy==13.6.0$/cupy-cuda12x==13.6.0/' "
            f"{source_directory}/requirements.txt "
            "> /opt/worldgen-requirements.txt",
            "uv pip install --system -r /opt/worldgen-requirements.txt",
        )
        .uv_pip_install(
            "einops==0.8.1",
            "tiktoken",
            "sentencepiece",
            "protobuf",
            "pydantic",
            "beartype",
            "jaxtyping",
        )
        .env({"CC": "gcc", "CXX": "g++"})
        .apt_install("zlib1g-dev", "libzstd-dev")
        .run_commands(
            "uv pip install --system --no-build-isolation "
            f"-r {source_directory}/requirements_git.txt",
            "uv pip install --system --no-build-isolation "
            f"{worldgen_directory}/third_party/gsplat_maskgaussian",
            "uv pip install --system --no-build-isolation "
            f"{worldgen_directory}/third_party/navmesh",
            "uv pip install --system --no-build-isolation flash-attn",
        )
        .workdir(worldgen_directory)
        .run_commands("rm -rf /cache")
        .uv_pip_install("rtree")
        .add_local_file(
            Path(__file__).with_name("worldgen-navigation.patch"),
            "/opt/worldgen-navigation.patch",
            copy=True,
        )
        .run_commands(f"git -C {source_directory} apply /opt/worldgen-navigation.patch")
    )
