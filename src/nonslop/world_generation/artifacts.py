import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from uuid import UUID

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from PIL import Image

from nonslop.models.world_schemas import CameraCalibration, CameraMatrix, WorldResult


def checkpoint_step(path: Path) -> int:
    return int(path.stem.split("_")[1])


@jaxtyped(typechecker=beartype)
def normalized_cameras(
    transform: Float[np.ndarray, "4 4"],
    cameras: Float[np.ndarray, "cameras 4 4"],
) -> Float[np.ndarray, "cameras 4 4"]:
    transformed = transform @ cameras
    scales = np.linalg.norm(transformed[:, 0, :3], axis=1)
    assert np.all(scales > 0)
    transformed[:, :3, :3] /= scales[:, None, None]
    return transformed


def export_generated_world(
    scene_directory: Path, result_directory: Path, job_id: str
) -> WorldResult:
    identifier = UUID(job_id)
    checkpoints = list(result_directory.glob("ckpts/ckpt_*_rank0.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No trained checkpoint in {result_directory / 'ckpts'}"
        )
    checkpoint_path = max(checkpoints, key=checkpoint_step)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    transform = np.asarray(checkpoint["transform"], dtype=np.float64).copy()
    del checkpoint

    data_directory = scene_directory / "gs_data"
    manifest = json.loads((data_directory / "cameras.json").read_text())
    image_paths = sorted((data_directory / "images").glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No training images in {data_directory / 'images'}")
    with Image.open(image_paths[0]) as image:
        width, height = image.size

    world_to_camera = np.stack(
        [
            np.asarray(manifest[image_path.stem]["extrinsic"], dtype=np.float64)
            for image_path in image_paths
        ]
    )
    cameras = normalized_cameras(transform, np.linalg.inv(world_to_camera))
    extrinsics: list[CameraMatrix] = []
    intrinsics: list[CameraMatrix] = []
    for camera_id, image_path in enumerate(image_paths):
        intrinsic = np.asarray(manifest[image_path.stem]["intrinsic"], dtype=np.float64)
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        intrinsic[0, :] *= width / image_width
        intrinsic[1, :] *= height / image_height
        extrinsics.append(
            CameraMatrix(camera_id=camera_id, matrix=cameras[camera_id].tolist())
        )
        intrinsics.append(CameraMatrix(camera_id=camera_id, matrix=intrinsic.tolist()))

    calibration = CameraCalibration(
        num_cameras=len(image_paths),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_width=width,
        image_height=height,
        generated_frame_count=len(image_paths),
    )
    output_directory = scene_directory.parent
    output_directory.mkdir(parents=True, exist_ok=True)
    ply_path = output_directory / "gaussians.ply"
    camera_path = output_directory / "camera_params.json"
    source_ply = (
        result_directory / "ply" / f"point_cloud_{checkpoint_step(checkpoint_path)}.ply"
    )
    with TemporaryDirectory(
        prefix=".worldgen-export-", dir=output_directory
    ) as temporary:
        temporary_directory = Path(temporary)
        temporary_ply = temporary_directory / ply_path.name
        temporary_camera = temporary_directory / camera_path.name
        copyfile(source_ply, temporary_ply)
        temporary_camera.write_text(calibration.model_dump_json())
        temporary_ply.replace(ply_path)
        temporary_camera.replace(camera_path)

    return WorldResult(
        job_id=identifier,
        ply_path=str(ply_path),
        camera_path=str(camera_path),
        input_kind="generated_video",
        input_count=len(image_paths),
        photo_count=0,
        generated_frame_count=len(image_paths),
        pipeline_kind="hy_world_generation",
    )
