from dataclasses import dataclass


@dataclass(frozen=True)
class CacheSettings:
    root: str = "/cache"
    volume_name: str = "nonslop-kernel-cache"

    def environment(self) -> dict[str, str]:
        return {
            "TORCHINDUCTOR_CACHE_DIR": f"{self.root}/torchinductor",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "TORCHINDUCTOR_COMPREHENSIVE_PADDING": "0",
            "TORCHINDUCTOR_MIX_ORDER_REDUCTION": "0",
            "TRITON_CACHE_DIR": f"{self.root}/triton",
            "CUDA_CACHE_PATH": f"{self.root}/cuda",
            "TORCH_EXTENSIONS_DIR": f"{self.root}/torch_extensions",
            "CUTE_DSL_CACHE_DIR": f"{self.root}/cute",
            "LAKONLAB_CACHE": f"{self.root}/lakonlab",
        }
