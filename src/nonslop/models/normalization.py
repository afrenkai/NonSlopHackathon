import torch
from lakonlab.models.architectures.sol_h3.model import SolH3Block


def align_refiner_norms(model: torch.nn.Module) -> int:
    normalized_layers = 0
    for module in model.modules():
        if isinstance(module, SolH3Block):
            module.norm1.float()
            module.norm2.float()
            normalized_layers += 2
    return normalized_layers
