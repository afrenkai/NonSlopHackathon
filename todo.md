# Project TODO

Updated: 2026-11-22. Maintain this file as implementation progresses; mark a
runtime milestone complete only after exercising it on Modal.

Goal: source material → grounded video → explorable 3D space → time capsules.

Live website: https://afrenkai--nonslop-streets-web.us-east.modal.run
The site opens without login. Browser sessions isolate visits and media; API
requests require server-side credentials configured through CLI flags or `.env`.

## Working baseline

- [x] Python 3.13 pi-H3 environment on Modal with the CUDA 13.0.3 cuDNN development image.
- [x] Persistent model downloads, FP8 transformer/text encoder, Sol/CuTe attention.
- [x] TorchInductor, Triton, CuTe, CUDA JIT, extensions, and LakonLab caches under `/cache`; graph artifacts survived a worker restart.
- [x] FastAPI + DuckDB + HTMX website with protected uploads/media, CSRF and upload limits.
- [x] Real text-to-video and image-conditioned video generation, archive grounding, and authenticated HTTP 206 playback.
- [x] Focused web tests, Ruff, and Black.
- [x] Enable the redesigned local website's real Modal workers, on-demand private video/3D downloads, web context, and capsule AI. Real two-reference generation, video playback, six-camera reconstruction, movement/Reset, AI observations, browser isolation, and 320–1440 px layouts passed; 11 focused tests, Ruff, Black, and the viewer build passed.
- [x] Deploy the Memory Worlds redesign to the existing public Modal URL. Public browser checks passed for assets, HTMX validation, capsule creation/reopening, timeline/whole views, CSRF, browser isolation, and 320–1440 px layouts with no unexpected browser or network errors.
- [x] Remove decorative editorial headers and the global footer from the public design; nine web tests and public desktop/mobile layout and control checks passed.
- [x] Apply locally served Outfit variable fonts to body text and controls; public font loading, weights, extended Latin, and desktop/mobile layouts passed browser checks.
- [x] Deploy and verify public access without login, per-browser history/media isolation, and CLI credential flags with `.env` defaults.
- [x] Separate CPU (`nonslop-streets`), video (`nonslop-video`), 3D (`nonslop-world`), and inference (`nonslop-inference`) into independent deployed Modal apps.
- [x] Add HTMX folder and multi-file pickers, retain/order all references, and bound uploads to 8 images / 40 MB total. Real browser selection of two images, Omni comparison, first/last-frame video generation, and reconstruction passed end to end.

## Current diagnosis and runtime decisions

| Symptom | Evidence and status |
| --- | --- |
| `c10::BFloat16` / cannot dispatch fused RMSNorm | A performance warning, not an exception. DWD code passes FP32 activations into RMSNorm with BF16 weights; PyTorch selects its composite implementation. Both generated clips completed. |
| Python 3.13.0 guard warning | Separate interpreter performance issue. PyTorch recommends 3.13.1 or newer within the same series. |
| `KeyError: __version__` during installation | Actual earlier Python 3.13 packaging incompatibility. Fixed by explicit `exec` namespaces in LakonLab/MMCV patches. |
| Missing processor / `NoneType.create_mm_token_type_ids` | Actual earlier component-loading failure. Fixed by explicitly assembling the cached processor components. |
| No CUDA runtime in processor check | Expected from a CPU-only Modal check using the CUDA image. GPU generation completed on H200. |

Keep the working pi-H3 deployment on Python 3.13. A move to 3.14 does not solve
mixed tensor dtypes. Do not globally recast the FP8 model to silence warnings.

## Next: CUTLASS DSL and runtime cleanup

- [x] Trace the reported c10 warning to PyTorch RMSNorm dispatch and the DWD normalization code.
- [x] Profile DWD normalization and FP8 kernels. Next CuTe candidate: FP8 linear/GEMM and rowwise-quantization fusion. Cold profile: FP8 GEMM 1.536 s / 1,605 calls, rowwise quantization 109 ms / 1,450 calls; warm measurements must precede a custom implementation. Ignore profiler callback rows and cold-autotuning reducer counts as steady-state evidence.
- [x] Resolve mixed-dtype RMSNorm dispatch at the responsible normalization boundary while retaining FP32 accumulation. Sixteen loaded refiner norms aligned; isolated H100 dispatch improved from 0.203 ms to 0.056 ms. Numerical assertions and a real 124-frame H200 clip passed; FP8 linear counts remain 348/473. Full profile retained under `/cache/profiling/3789e363-3599-40a7-81af-41dbc2a70f07.txt`.
- [x] Replace the Modal-injected Python 3.13.0 runtime with Python 3.13.12. CPU imports, cached processor loading, and TorchInductor GPU compilation passed; deployed video image uses the patch runtime.

## Main milestone: video → 3D with HY-World

- [x] Locate official weights: [`tencent/HY-World-2.0/HY-WorldMirror-2.0`](https://huggingface.co/tencent/HY-World-2.0/tree/main/HY-WorldMirror-2.0), approximately 5.05 GB.
- [x] Confirm WorldMirror accepts video and exports `gaussians.ply`, `points.ply`, depth, and camera parameters.
- [x] Create a separate Modal reconstruction image: Python 3.12, CUDA 13.0.3, Torch 2.10, official FA2 binary, and inference-only gsplat installation.
- [x] Cache only the WorldMirror component for the first reconstruction trial.
- [x] Run an existing pi-H3 MP4 through WorldMirror; retain 3D assets, cameras, and source job ID on a Modal volume. Six frames, 500k splat cap, approximately 5.3 seconds after model load on H100.
- [x] Inspect geometry and camera coverage: six camera-to-world poses, finite 490k-splat initial output, and a rendered street view. Black gaps remain unfilled and the viewer labels all geometry as inferred and unscaled.
- [x] Serve protected 3D assets and integrate the three.js/Spark viewer. The live folder workflow passed with no page, network, console, or CSP errors; other browser workspaces received 404 for video/PLY assets.
- [x] Acceptance: open a generated street in the browser, then verify keyboard movement and Reset using its retained geometry and the deployed viewer. Long frame times are clamped; accessible status confirms actual control updates. Rendered geometry was visually inspected.

## Supporting service: SGLang-Omni on H100

- [x] Confirm upstream provides a single-H100 Qwen3-Omni FP8 configuration.
- [x] Build an isolated Python 3.12 worker with pinned SGLang-Omni source, persistent FP8 model storage, and isolated caches under `/cache/omni`. Defer XDG cache setup until volume mounting to avoid Modal bootstrap collisions.
- [x] Validate image, video, multiple-image understanding, and known-speech transcription on H100. Responses are labeled AI observations because descriptions can infer unsupported details.
- [x] Connect structured scene observations and user corrections to capsule entries and linked visit/3D evidence panels. Omni supplies understanding; WorldMirror supplies 3D reconstruction.

Compatibility: current upstream SGLang-Omni requires Python `>=3.11,<3.13`,
Torch 2.14.1 and Transformers 5.19.0; its stack uses a different CuTe DSL pin.
It cannot share pi-H3's present environment, and Python 3.14 is outside that
supported range.

## After 3D: time capsules

- [x] Add an object/place anchor and dated user-authored text, image, voice, and video entries with HTMX.
- [x] Preserve original entries; label transcription, AI observations, and generated reconstructions separately. Focused tests verify original image/audio retrieval and reopening.
- [x] Link entries to their videos, 3D spaces, sources, and corrections, with owner-scoped evidence panels.
- [x] Add HTMX whole-capsule and timeline views; leave undocumented periods empty.
- [x] Add capsule ownership/access controls using persistent signed browser workspaces. Other browsers receive 404 for capsule metadata and media.
- [x] Acceptance: add/reopen dated entries with original photos and audio, complete real Omni transcription through the live app, and validate linked visit/3D correction panels in focused integration tests.

## References

- [PyTorch RMSNorm dispatch](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/layer_norm.cpp)
- [HY-World reconstruction API and outputs](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/DOCUMENTATION.md)
- [HY-World dependency requirements](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/requirements.txt)
- [SGLang-Omni Python/dependency contract](https://github.com/sgl-project/sglang-omni/blob/main/pyproject.toml)
- [SGLang-Omni single-H100 FP8 configuration](https://github.com/sgl-project/sglang-omni/blob/main/examples/configs/qwen3_omni_fp8_colocated.yaml)
