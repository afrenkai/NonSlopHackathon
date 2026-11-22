import subprocess
from importlib.metadata import version
from platform import python_version

import torch
from lakonlab.models.architectures.diffusers.attention import default_attention_backend
from lakonlab.ops.sol_attn import get_sol_attn_backend, sol_attn
from lakonlab.pipelines.pipeline_minimax_h3 import MiniMaxH3DWDBlocks


def validate_installation() -> dict[str, str]:
    subprocess.run(["uv", "pip", "check", "--system"], check=True)
    compiler = subprocess.run(
        ["nvcc", "--version"], check=True, capture_output=True, text=True
    )
    assert torch.cuda.is_available()
    assert torch.version.cuda == "13.0"

    query = torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    backend = get_sol_attn_backend()
    assert backend == "cute_sm90", backend
    output = sol_attn(query, key, value, thresh_type="exact")
    torch.cuda.synchronize()
    assert output.shape == query.shape
    assert torch.isfinite(output).all().item()

    return {
        "gpu": torch.cuda.get_device_name(),
        "python": python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()),
        "nvcc": compiler.stdout.strip(),
        "lakonlab": version("lakonlab"),
        "diffusers": version("diffusers"),
        "transformers": version("transformers"),
        "cutlass_dsl": version("nvidia-cutlass-dsl"),
        "pipeline": type(MiniMaxH3DWDBlocks()).__name__,
        "dense_attention": default_attention_backend("dense"),
        "sol_attention": backend,
        "sol_output_shape": str(tuple(output.shape)),
    }
