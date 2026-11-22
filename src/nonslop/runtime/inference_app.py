from __future__ import annotations

import base64
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

from nonslop.models.inference_schemas import (
    ArchivistImagesRequest,
    ArchivistRequest,
    ArchivistResult,
    InferenceSettings,
    MediaType,
)

settings = InferenceSettings()
model_volume = modal.Volume.from_name(
    settings.model_volume, create_if_missing=True, version=2
)
cache_volume = modal.Volume.from_name(
    settings.cache_volume, create_if_missing=True, version=2
)
inference_image = (
    modal.Image.from_registry(settings.image_repository)
    .entrypoint([])
    .run_commands(
        "uv pip install --python /opt/sglang/bin/python3 --no-deps "
        "--no-build-isolation "
        f"'git+https://github.com/sgl-project/sglang-omni.git@{settings.source_revision}'"
    )
    .env(settings.environment())
    .run_commands("if [ -d /cache ]; then mv /cache /opt/sglang-prebuilt-cache; fi")
    .add_local_python_source("nonslop")
)
download_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("huggingface-hub[hf_xet]", "pydantic", "httpx", "pyyaml")
    .env(settings.environment())
    .add_local_python_source("nonslop")
)
app = modal.App("nonslop-inference")

with inference_image.imports():
    import httpx
    import yaml
    from huggingface_hub import snapshot_download


@app.function(
    image=download_image,
    cpu=4,
    memory=8192,
    volumes={"/models": model_volume},
    timeout=3600,
)
def download_models() -> str:
    try:
        model_path = snapshot_download(settings.model_repository, max_workers=8)
    finally:
        model_volume.commit()
    return model_path


@app.function(image=inference_image, cpu=2, memory=4096, timeout=300)
def inspect_image() -> dict[str, str]:
    config_path = Path("/tmp/archivist-config-check.yaml")
    config_path.write_text(
        yaml.safe_dump(settings.pipeline_configuration(settings.model_repository)),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--no-sync",
            "--python",
            "/opt/sglang/bin/python3",
            "-m",
            "sglang_omni.cli",
            "config",
            "resolve",
            "--config",
            str(config_path),
            "--text-only",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    versions = {
        package_name: importlib.metadata.version(package_name)
        for package_name in (
            "torch",
            "torchvision",
            "torchaudio",
            "sglang",
            "sglang-omni",
            "transformers",
            "flashinfer-python",
            "nvidia-cutlass-dsl",
        )
    }
    shared_memory = os.statvfs("/dev/shm")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "cuda": os.environ.get("CUDA_VERSION", ""),
        "shared_memory_bytes": str(shared_memory.f_blocks * shared_memory.f_frsize),
        "configuration": "validated by sgl-omni config resolve",
        **versions,
    }


@app.cls(
    image=inference_image,
    gpu="H100",
    cpu=8,
    memory=98304,
    volumes={"/models": model_volume, "/cache": cache_volume},
    timeout=3600,
    scaledown_window=settings.scaledown_window,
    max_containers=1,
)
@modal.concurrent(max_inputs=1)
class MemoryArchivist:
    @modal.enter()
    def load(self) -> None:
        self.server_process: subprocess.Popen[bytes] | None = None
        os.environ.update(settings.process_environment())
        for cache_path in settings.process_environment().values():
            if cache_path.startswith("/cache/"):
                Path(cache_path).mkdir(parents=True, exist_ok=True)
        model_path = snapshot_download(settings.model_repository, local_files_only=True)
        config_path = Path("/tmp/memory-archivist.yaml")
        config_path.write_text(
            yaml.safe_dump(settings.pipeline_configuration(model_path)),
            encoding="utf-8",
        )
        self.server_process = subprocess.Popen(
            [
                "uv",
                "run",
                "--no-project",
                "--no-sync",
                "--python",
                "/opt/sglang/bin/python3",
                "-m",
                "sglang_omni.cli",
                "serve",
                "--config",
                str(config_path),
                "--text-only",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--model-name",
                "qwen3-omni",
            ]
        )
        try:
            self.wait_until_ready()
        finally:
            cache_volume.commit()

    def wait_until_ready(self) -> None:
        server_process = self.server_process
        assert server_process is not None
        deadline = time.monotonic() + settings.startup_timeout
        with httpx.Client(base_url=settings.server_url, timeout=5) as client:
            while time.monotonic() < deadline:
                if server_process.poll() is not None:
                    raise RuntimeError(
                        f"SGLang-Omni exited with code {server_process.returncode}"
                    )
                try:
                    if client.get("/health").status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                time.sleep(2)
        server_process.terminate()
        raise TimeoutError("SGLang-Omni did not become healthy before startup timeout")

    @modal.method()
    def analyze(
        self, prompt: str, media: bytes | None = None, media_type: MediaType = ""
    ) -> str:
        request = ArchivistRequest(prompt=prompt, media=media, media_type=media_type)
        payload: dict[str, object] = {
            "model": "qwen3-omni",
            "messages": [{"role": "user", "content": request.prompt}],
            "modalities": ["text"],
            "max_tokens": settings.max_tokens,
            "temperature": 0,
        }
        if request.media is not None:
            media_url = (
                f"data:{request.media_type};base64,"
                f"{base64.b64encode(request.media).decode('ascii')}"
            )
            media_family = request.media_type.partition("/")[0]
            payload[f"{media_family}s"] = [media_url]
        return self.complete_chat(payload)

    @modal.method()
    def analyze_images(self, prompt: str, images: list[bytes]) -> str:
        request = ArchivistImagesRequest(prompt=prompt, images=images)
        payload: dict[str, object] = {
            "model": "qwen3-omni",
            "messages": [{"role": "user", "content": request.prompt}],
            "images": [
                f"data:image/png;base64,{base64.b64encode(image).decode('ascii')}"
                for image in request.images
            ],
            "modalities": ["text"],
            "max_tokens": settings.max_tokens,
            "temperature": 0,
        }
        return self.complete_chat(payload)

    def complete_chat(self, payload: dict[str, object]) -> str:
        try:
            with httpx.Client(
                base_url=settings.server_url, timeout=settings.request_timeout
            ) as client:
                response = client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            return ArchivistResult(text=message["content"]).model_dump_json()
        finally:
            cache_volume.commit()

    @modal.method()
    def transcribe(self, audio: bytes, filename: str = "memory.wav") -> str:
        ArchivistRequest(prompt="Transcribe", media=audio, media_type="audio/wav")
        try:
            with httpx.Client(
                base_url=settings.server_url, timeout=settings.request_timeout
            ) as client:
                response = client.post(
                    "/v1/audio/transcriptions",
                    data={
                        "model": "qwen3-omni",
                        "response_format": "json",
                        "temperature": "0",
                    },
                    files={"file": (Path(filename).name, audio)},
                )
                response.raise_for_status()
            return ArchivistResult(text=response.json()["text"]).model_dump_json()
        finally:
            cache_volume.commit()

    @modal.exit()
    def shutdown(self) -> None:
        try:
            server_process = self.server_process
            if server_process is None:
                return
            server_process.terminate()
            try:
                server_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=5)
        finally:
            cache_volume.commit()


@app.local_entrypoint()
def main(
    action: str = "inspect",
    image_path: str = "/tmp/nonslop-street-preview.png",
    video_path: str = "/tmp/nonslop-street-preview.mp4",
    audio_path: str = "/tmp/nonslop-omni-investigation/tests/data/query_to_cars.wav",
) -> None:
    if action == "inspect":
        print(json.dumps(inspect_image.remote(), indent=2))
    elif action == "download":
        print(download_models.remote())
    elif action == "smoke":
        print(MemoryArchivist().analyze.remote("Reply with the word ready."))
    elif action == "images-smoke":
        reference_image = Path(image_path).read_bytes()
        print(
            MemoryArchivist().analyze_images.remote(
                "How many reference images were provided? Then describe their shared "
                "visible architecture in one sentence. Do not infer location or year.",
                [reference_image, reference_image],
            )
        )
    elif action == "media-smoke":
        archivist = MemoryArchivist()
        for media_name, source_path, media_type in (
            ("image", image_path, "image/png"),
            ("video", video_path, "video/mp4"),
        ):
            result = ArchivistResult.model_validate_json(
                archivist.analyze.remote(
                    "Describe only the visible buildings and street layout in two "
                    "concise sentences. Do not infer the location or year.",
                    Path(source_path).read_bytes(),
                    media_type,
                )
            )
            print(json.dumps({"kind": media_name, "text": result.text}), flush=True)
        transcript = ArchivistResult.model_validate_json(
            archivist.transcribe.remote(
                Path(audio_path).read_bytes(), Path(audio_path).name
            )
        )
        print(
            json.dumps({"kind": "transcription", "text": transcript.text}), flush=True
        )
    else:
        raise ValueError(
            "Choose inspect, download, smoke, images-smoke, or media-smoke"
        )
