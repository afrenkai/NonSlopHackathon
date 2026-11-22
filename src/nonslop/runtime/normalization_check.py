from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

from nonslop.runtime.python_patch_check import build_patched_image
from nonslop.runtime.settings import CacheSettings

cache_settings = CacheSettings(root="/cache/normalization-py31312")
cache_volume = modal.Volume.from_name(
    cache_settings.volume_name, create_if_missing=True, version=2
)
image = (
    build_patched_image()
    .uv_pip_install("beartype", "jaxtyping")
    .env(cache_settings.environment())
)
app = modal.App("nonslop-normalization-check", image=image)

with image.imports():
    import torch
    from beartype import beartype
    from jaxtyping import Float, Float32, jaxtyped
    from lakonlab.models.architectures.sol_h3.model import SolH3Block
    from torch.nn import functional
    from torch.profiler import ProfilerActivity, profile

    from nonslop.models.normalization import align_refiner_norms

    @jaxtyped(typechecker=beartype)
    def native_rms_norm(
        activations: Float32[torch.Tensor, "tokens features"],
        weight: Float[torch.Tensor, "features"],  # noqa: F821, UP037
    ) -> Float32[torch.Tensor, "tokens features"]:
        return functional.rms_norm(
            activations, (activations.shape[-1],), weight, eps=1e-5
        )

    @jaxtyped(typechecker=beartype)
    def time_rms_norm(
        activations: Float32[torch.Tensor, "tokens features"],
        weight: Float[torch.Tensor, "features"],  # noqa: F821, UP037
        iterations: int,
    ) -> float:
        for warmup_index in range(20):
            functional.rms_norm(activations, (activations.shape[-1],), weight, eps=1e-5)
        torch.cuda.synchronize()
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        for iteration_index in range(iterations):
            functional.rms_norm(activations, (activations.shape[-1],), weight, eps=1e-5)
        finished.record()
        finished.synchronize()
        return started.elapsed_time(finished) / iterations

    @jaxtyped(typechecker=beartype)
    def profile_rms_norm(
        activations: Float32[torch.Tensor, "tokens features"],
        weight: Float[torch.Tensor, "features"],  # noqa: F821, UP037
    ) -> tuple[list[str], list[str]]:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            functional.rms_norm(activations, (activations.shape[-1],), weight, eps=1e-5)
            torch.cuda.synchronize()
        operators = sorted({event.key for event in trace.key_averages()})
        kernels = sorted(
            {
                event.name
                for event in trace.events()
                if str(event.device_type) == "DeviceType.CUDA"
            }
        )
        return operators, kernels


def inspect_block_dtypes() -> dict[str, str]:
    block = SolH3Block().bfloat16()
    aligned_norms = align_refiner_norms(block)
    assert aligned_norms == 2
    assert block.norm1.weight.dtype == torch.float32
    assert block.norm2.weight.dtype == torch.float32
    assert block.scale1.dtype == torch.bfloat16
    assert block.scale2.dtype == torch.bfloat16
    assert all(
        parameter.dtype == torch.bfloat16
        for module in block.modules()
        if isinstance(module, torch.nn.Linear)
        for parameter in module.parameters(recurse=False)
    )
    return {
        "aligned_norms": str(aligned_norms),
        "norm1_dtype": str(block.norm1.weight.dtype),
        "norm2_dtype": str(block.norm2.weight.dtype),
        "linear_dtype": str(block.attn.to_qkv.weight.dtype),
    }


@app.function(
    gpu="H100", cpu=4, memory=16384, volumes={"/cache": cache_volume}, timeout=600
)
def check_normalization() -> dict[str, str]:
    try:
        for path in cache_settings.environment().values():
            if path.startswith("/cache/"):
                Path(path).mkdir(parents=True, exist_ok=True)
        block_report = inspect_block_dtypes()
        torch.manual_seed(42)
        activations = torch.randn(8192, 2048, device="cuda", dtype=torch.float32)
        mixed_weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        aligned_weight = mixed_weight.float()
        mixed_output = native_rms_norm(activations, mixed_weight)
        aligned_output = native_rms_norm(activations, aligned_weight)
        torch.testing.assert_close(aligned_output, mixed_output, rtol=1e-5, atol=1e-6)
        mixed_downstream = mixed_output.bfloat16()
        aligned_downstream = aligned_output.bfloat16()
        torch.testing.assert_close(
            aligned_downstream, mixed_downstream, rtol=0.01, atol=1e-5
        )
        compiled_norm = torch.compile(functional.rms_norm, fullgraph=True)
        compiled_output = compiled_norm(
            activations, (activations.shape[-1],), aligned_weight, eps=1e-5
        )
        torch.testing.assert_close(
            compiled_output, aligned_output, rtol=1e-5, atol=1e-6
        )
        mixed_operators, mixed_kernels = profile_rms_norm(activations, mixed_weight)
        aligned_operators, aligned_kernels = profile_rms_norm(
            activations, aligned_weight
        )
        mixed_ms = time_rms_norm(activations, mixed_weight, 100)
        aligned_ms = time_rms_norm(activations, aligned_weight, 100)
        return block_report | {
            "python": sys.version,
            "gpu": torch.cuda.get_device_name(),
            "shape": str(tuple(activations.shape)),
            "activation_dtype": str(activations.dtype),
            "mixed_ms": str(mixed_ms),
            "aligned_ms": str(aligned_ms),
            "speedup": str(mixed_ms / aligned_ms),
            "fp32_max_absolute_error": str(
                (aligned_output - mixed_output).abs().max().item()
            ),
            "bf16_max_absolute_error": str(
                (aligned_downstream - mixed_downstream).abs().max().item()
            ),
            "compiled_max_absolute_error": str(
                (compiled_output - aligned_output).abs().max().item()
            ),
            "mixed_operators": json.dumps(mixed_operators),
            "aligned_operators": json.dumps(aligned_operators),
            "mixed_cuda_kernels": json.dumps(mixed_kernels),
            "aligned_cuda_kernels": json.dumps(aligned_kernels),
            "compiler_cache": cache_settings.root,
        }
    finally:
        cache_volume.commit()


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(check_normalization.remote(), indent=2))
