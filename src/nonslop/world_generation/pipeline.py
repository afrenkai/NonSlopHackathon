from pathlib import Path
from subprocess import run
from sys import executable

from nonslop.models.world_generation_schemas import WorldGenerationSettings
from nonslop.models.world_schemas import WorldResult
from nonslop.world_generation.artifacts import export_generated_world


def run_native_stage(
    name: str,
    arguments: list[str],
    working_directory: Path,
    required_artifacts: list[Path],
) -> None:
    print(f"HY-World: {name}", flush=True)
    run(
        ["uv", "run", "--no-project", "--no-sync", "--python", executable, *arguments],
        cwd=working_directory,
        check=True,
    )
    for artifact in required_artifacts:
        artifact.stat()


def run_world_generation(
    job_directory: Path,
    anchor_image: Path,
    prompt: str,
    llm_port: int,
    settings: WorldGenerationSettings,
) -> WorldResult:
    job_directory = job_directory.resolve()
    scene_directory = job_directory / "worldgen"
    scene_directory.mkdir(parents=True, exist_ok=True)
    source_directory = Path(settings.source_root)
    worldgen_directory = source_directory / "hyworld2" / "worldgen"
    render_directory = scene_directory / "render_results"
    stereo_bank = render_directory / f"generation_bank_{settings.stereo_model}"
    panorama = scene_directory / "panorama.png"
    distributed = [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=1",
    ]
    vlm_arguments = [
        "--llm_addr",
        "127.0.0.1",
        "--llm_port",
        str(llm_port),
        "--llm_name",
        settings.vlm_model,
    ]

    run_native_stage(
        "generate the 360 panorama with HY-Pano",
        [
            "pipeline_with_qwen_image.py",
            "--image",
            str(anchor_image.resolve()),
            "--prompt",
            prompt,
            "--pretrained-model-name-or-path",
            settings.panorama_repository,
            "--lora-path",
            settings.panorama_lora_repository,
            "--lora-subfolder",
            settings.panorama_lora_subfolder,
            "--num-inference-steps",
            str(settings.panorama_steps),
            "--height",
            str(settings.panorama_height),
            "--width",
            str(settings.panorama_width),
            "--save",
            str(panorama),
        ],
        source_directory / "hyworld2" / "panogen",
        [panorama],
    )
    run_native_stage(
        "plan camera trajectories with WorldNav",
        [
            "traj_generate.py",
            "--target_path",
            str(scene_directory),
            "--apply_nav_traj",
            "--force_vlm",
            "--nframe",
            str(settings.trajectory_frames),
            "--split_view_num",
            str(settings.split_view_count),
            "--splitted_resolution",
            str(settings.trajectory_resolution),
            "--wonder_topk",
            str(settings.wonder_topk),
            "--recon_topk",
            str(settings.reconstruction_topk),
            *vlm_arguments,
        ],
        worldgen_directory,
        [scene_directory / "meta_info.json", render_directory / "global_pcd.ply"],
    )
    trajectory_cameras = sorted(render_directory.glob("*/traj*/camera.json"))
    if not trajectory_cameras:
        raise FileNotFoundError(
            f"WorldNav produced no camera trajectories in {scene_directory}"
        )
    first_trajectory = trajectory_cameras[0].parent
    run_native_stage(
        "render and caption the camera trajectories",
        [
            *distributed,
            "traj_render.py",
            "--target_path",
            str(scene_directory),
            *vlm_arguments,
        ],
        worldgen_directory,
        [first_trajectory / "render.mp4", first_trajectory / "traj_caption.json"],
    )
    run_native_stage(
        "expand the scene with WorldStereo DMD",
        [
            *distributed,
            "video_gen.py",
            "--target_path",
            str(scene_directory),
            "--model_type",
            settings.stereo_model,
        ],
        worldgen_directory,
        [stereo_bank / "global_pcd.ply", stereo_bank / "aligned_pcd.ply"],
    )
    run_native_stage(
        "prepare depth, normals, and cameras for 3D Gaussian training",
        [
            *distributed,
            "gen_gs_data.py",
            "--root_path",
            str(scene_directory),
            "--result_name",
            settings.stereo_model,
            "--save_normal",
            "--split_sky",
        ],
        worldgen_directory,
        [
            scene_directory / "gs_data" / "cameras.json",
            scene_directory / "gs_data" / "points.ply",
        ],
    )
    return train_generated_world(job_directory, settings)


def train_generated_world(
    job_directory: Path, settings: WorldGenerationSettings
) -> WorldResult:
    job_directory = job_directory.resolve()
    scene_directory = job_directory / "worldgen"
    result_directory = scene_directory / "trained"
    worldgen_directory = Path(settings.source_root) / "hyworld2" / "worldgen"
    steps = settings.training_steps
    refinement_stop = max(1, steps // 2)
    run_native_stage(
        "train the navigable 3D Gaussian world",
        [
            "-m",
            "world_gs_trainer",
            "default",
            "--data_dir",
            str(scene_directory / "gs_data"),
            "--result_dir",
            str(result_directory),
            "--max_steps",
            str(steps),
            "--save_steps",
            str(steps),
            "--eval_steps",
            str(steps),
            "--ply_steps",
            str(steps),
            "--save_ply",
            "--disable_viewer",
            "--disable_video",
            "--use_scale_regularization",
            "--depth_loss",
            "--normal_loss",
            "--sky_depth_from_pcd",
            "--use_mask_gaussian",
            "--mask_export_stochastic",
            "--use_anchor_protection",
            "--strategy.refine-start-iter",
            str(max(1, steps // 10)),
            "--strategy.refine-stop-iter",
            str(refinement_stop),
            "--strategy.refine-every",
            str(max(1, steps // 15)),
            "--strategy.refine-scale2d-stop-iter",
            str(refinement_stop),
            "--strategy.reset-every",
            "99990",
            "--strategy.grow-grad2d",
            "0.0001",
            "--strategy.prune-scale3d",
            "0.1",
        ],
        worldgen_directory,
        [
            result_directory / "ckpts" / f"ckpt_{steps - 1}_rank0.pt",
            result_directory / "ply" / f"point_cloud_{steps - 1}.ply",
        ],
    )
    print("HY-World: export the trained world and aligned cameras", flush=True)
    return export_generated_world(scene_directory, result_directory, job_directory.name)
