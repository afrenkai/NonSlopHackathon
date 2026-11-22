import json
from pathlib import Path

import modal


def build_image() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04", add_python="3.13"
        )
        .entrypoint([])
        .apt_install(
            "build-essential", "ffmpeg", "git", "libgl1", "libglib2.0-0", "ninja-build"
        )
        .uv_pip_install("setuptools<74", "wheel", "packaging", "uv")
        .uv_pip_install(
            "torch==2.10.0",
            "torchvision==0.25.0",
            index_url="https://download.pytorch.org/whl/cu130",
        )
        .run_commands(
            "git clone https://github.com/Lakonik/LakonLab.git /opt/LakonLab "
            "&& git -C /opt/LakonLab checkout 2ee2b334374ec91b7cee371a85a524bf4ba644d3",
        )
        .run_commands(
            "git clone --depth 1 --branch v1.6.1 "
            "https://github.com/open-mmlab/mmcv.git /opt/mmcv"
        )
        .add_local_file(
            Path(__file__).with_name("lakonlab-python313.patch"),
            "/opt/lakonlab-python313.patch",
            copy=True,
        )
        .add_local_file(
            Path(__file__).with_name("mmcv-python313.patch"),
            "/opt/mmcv-python313.patch",
            copy=True,
        )
        .run_commands(
            "git -C /opt/LakonLab apply /opt/lakonlab-python313.patch",
            "git -C /opt/mmcv apply /opt/mmcv-python313.patch",
            "uv pip install --system --no-build-isolation "
            "-e /opt/mmcv -e '/opt/LakonLab[sol]' 'nvidia-cutlass-dsl[cu13]'",
        )
        .workdir("/opt/LakonLab")
        .run_commands(
            "uv python install 3.13.12 --install-dir /opt/nonslop-python --no-bin",
            "cp -a --remove-destination "
            "/opt/nonslop-python/cpython-3.13.12-linux-x86_64-gnu/. /usr/local/",
            "rm -rf /opt/nonslop-python",
        )
    )


image = build_image()
app = modal.App("nonslop-pih3", image=image)

with image.imports():
    from nonslop.runtime.validation import validate_installation


@app.function(gpu="H100", timeout=600)
def smoke_test() -> dict[str, str]:
    return validate_installation()


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(smoke_test.remote(), indent=2))
