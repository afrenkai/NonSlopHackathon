from pathlib import Path

import numpy as np
from beartype import beartype
from jaxtyping import Bool, Float, UInt8, jaxtyped
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from nonslop.models.world_schemas import CameraCalibration
from nonslop.reconstruction.coverage_schemas import CoverageGaussians, CoverageSettings


@jaxtyped(typechecker=beartype)
def coverage_gaussians(
    depth: Float[np.ndarray, "height width"],
    colors: UInt8[np.ndarray, "height width 3"],
    geometry_mask: Bool[np.ndarray, "height width"],
    intrinsics: Float[np.ndarray, "3 3"],
    pose: Float[np.ndarray, "4 4"],
    scene_depth: float,
    settings: CoverageSettings,
) -> CoverageGaussians:
    rows, columns = np.mgrid[
        0 : depth.shape[0] : settings.pixel_stride,
        0 : depth.shape[1] : settings.pixel_stride,
    ]
    samples = depth[rows, columns].copy()
    sky = ~geometry_mask[rows, columns]
    samples[sky] = scene_depth * settings.sky_distance_multiplier
    samples[~sky] *= settings.surface_depth_offset
    valid = np.isfinite(samples) & (samples > 0)
    pixel_columns = columns[valid].astype(np.float32) + 0.5
    pixel_rows = rows[valid].astype(np.float32) + 0.5
    pixels = np.stack((pixel_columns, pixel_rows, np.ones_like(pixel_rows)), axis=1)
    distances = samples[valid]
    rays = np.linalg.solve(intrinsics, pixels.T).T
    positions = (rays * distances[:, None]) @ pose[:3, :3].T + pose[:3, 3]
    horizontal_scale = (
        distances / intrinsics[0, 0] * settings.pixel_stride * settings.pixel_scale
    )
    vertical_scale = (
        distances / intrinsics[1, 1] * settings.pixel_stride * settings.pixel_scale
    )
    scales = np.stack(
        (
            horizontal_scale,
            vertical_scale,
            np.minimum(horizontal_scale, vertical_scale) * settings.thickness_ratio,
        ),
        axis=1,
    )
    quaternion = np.roll(Rotation.from_matrix(pose[:3, :3]).as_quat(), 1).astype(
        np.float32
    )
    rotations = np.broadcast_to(quaternion, (len(positions), 4)).copy()
    coefficients = (
        colors[rows, columns][valid].astype(np.float32) / 255.0 - 0.5
    ) / settings.spherical_harmonic_dc_basis
    return CoverageGaussians(positions, scales, rotations, coefficients)


def save_coverage_ply(
    path: Path, gaussians: CoverageGaussians, settings: CoverageSettings
) -> None:
    names = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    vertices = np.zeros(
        len(gaussians.positions), dtype=[(name, "f4") for name in names]
    )
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = gaussians.positions[:, axis]
    for axis in range(3):
        vertices[f"f_dc_{axis}"] = gaussians.colors[:, axis]
        vertices[f"scale_{axis}"] = np.log(gaussians.scales[:, axis])
    for axis in range(4):
        vertices[f"rot_{axis}"] = gaussians.rotations[:, axis]
    vertices["opacity"] = np.log(settings.opacity / (1 - settings.opacity))
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def write_coverage_views(
    directory: Path,
    image_paths: list[str],
    calibration: CameraCalibration,
    use_sky_mask: bool,
) -> None:
    settings = CoverageSettings()
    assert calibration.image_width is not None and calibration.image_height is not None
    width, height = calibration.image_width, calibration.image_height
    intrinsics_by_id = {
        camera.camera_id: camera.matrix for camera in calibration.intrinsics
    }
    coverage_directory = directory / "coverage"
    coverage_directory.mkdir(exist_ok=True)
    for camera_index, camera in enumerate(calibration.extrinsics):
        depth = np.load(
            directory / "depth" / f"depth_{camera_index:04d}.npy", allow_pickle=False
        ).astype(np.float32)
        assert depth.shape == (height, width)
        with Image.open(image_paths[camera_index]) as source:
            colors = np.asarray(
                source.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
            )
        mask_path = directory / "sky_mask" / f"sky_mask_{camera_index:04d}.png"
        if use_sky_mask and mask_path.is_file():
            with Image.open(mask_path) as mask_image:
                geometry_mask = np.asarray(mask_image.convert("L")) == 0
        else:
            geometry_mask = np.ones(depth.shape, dtype=bool)
        valid_depth = np.isfinite(depth) & (depth > 0)
        visible_depth = depth[valid_depth & geometry_mask]
        scene_depth = float(
            np.median(visible_depth if visible_depth.size else depth[valid_depth])
        )
        if camera_index == 0:
            calibration.scene_depth = scene_depth
            sky_colors = colors[~geometry_mask]
            background = (
                sky_colors
                if sky_colors.size
                else np.concatenate(
                    (colors[0], colors[-1], colors[:, 0], colors[:, -1])
                )
            )
            calibration.background_rgb = (
                np.median(background, axis=0) / 255.0
            ).tolist()
        gaussians = coverage_gaussians(
            depth,
            colors,
            geometry_mask,
            np.asarray(intrinsics_by_id[camera.camera_id], dtype=np.float32),
            np.asarray(camera.matrix, dtype=np.float32),
            scene_depth,
            settings,
        )
        save_coverage_ply(
            coverage_directory / f"{camera.camera_id:04d}.ply", gaussians, settings
        )
    calibration.coverage_available = True
