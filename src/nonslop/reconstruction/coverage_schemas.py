from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from jaxtyping import Float


@dataclass(frozen=True)
class CoverageSettings:
    pixel_stride: int = 4
    surface_depth_offset: float = 1.05
    sky_distance_multiplier: float = 40.0
    pixel_scale: float = 0.8
    thickness_ratio: float = 0.05
    opacity: float = 0.85
    spherical_harmonic_dc_basis: float = 0.28209479177387814


class CoverageGaussians(NamedTuple):
    positions: Float[np.ndarray, "points 3"]
    scales: Float[np.ndarray, "points 3"]
    rotations: Float[np.ndarray, "points 4"]
    colors: Float[np.ndarray, "points 3"]
