# Memory Worlds

A prototype for revisiting remembered scenes, indoors or outdoors, from one or
a few photographs, descriptions, scans, and archive material. FastAPI serves a
small HTMX interface. Kimi K3.1 Pro composes a scene prompt from the evidence,
pi-H3 generates a 5.2-second visit with audio, and WorldMirror reconstructs a
Gaussian-splat space from frames of that video. Field notes retain photographs,
optional remembered details, source context, and generated visits.

The implementation roadmap is maintained in [todo.md](todo.md): video → 3D
space with HY-WorldMirror, with an isolated SGLang-Omni service.

Live website: https://afrenkai--nonslop-streets-web.us-east.modal.run

The Field Notes guide opens at `/`; `/dashboard` holds the browser's field notes,
and `/visits/new` opens the photograph intake sheet. The upstream
guide, paper palette, typography, and navigation are integrated with the protected
browser workspaces.

## Run locally with model workers

From the repo root with Python 3.13, install once with `uv sync --locked --group checks`. Authenticate with `uv run -m modal setup` using the workspace containing the deployed apps below, then start the website:

```bash
uv run scripts/local_preview.py
```

Open **http://127.0.0.1:8742/**. Use one server at a time; stop it with Ctrl+C. Field notes, photographs, and generated media use the local store. Keep the browser cookie to return after restarting. On Windows, the database, uploads and session key live under `%LOCALAPPDATA%\NonSlop\cpu-preview`; on other systems they live under `$XDG_DATA_HOME/NonSlop/cpu-preview` (default `~/.local/share/NonSlop/cpu-preview`). Override this with `--data-root /absolute/private/path`, outside the checkout and its parent. Keep this directory private; never serve it with a static file server.

Video generation, 3D reconstruction, and reference-image observations run on the existing private Modal workers using your Modal credentials. Generated videos and 3D files download on demand into the private local cache and are served through the same owner-protected routes. Web context lookup is enabled. Uploads selected for generation or AI processing are sent to Modal, and GPU calls incur normal Modal charges; model weights stay on Modal. `.env` is loaded without overriding existing environment variables. Image scans need local English Tesseract.

## Run on Modal

```bash
uv run modal setup
uv run modal run -m nonslop.runtime.download_app
uv run modal run -m nonslop.runtime.world_app --download
uv run modal run -m nonslop.runtime.inference_app --action download
uv run nonslop deploy
```

Use Python 3.13 for the launcher, web, and pi-H3 containers. ML packages and model
weights stay on Modal. The optional `checks` dependency group contains CPU web
and test packages for local validation.

The video image uses Python 3.13.12 and `nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04` with PyTorch
2.10.0+cu130, torchvision 0.25.0, LakonLab 0.3.0, and Sol-Attn/CuTe DSL.
LakonLab is pinned to `2ee2b334374ec91b7cee371a85a524bf4ba644d3`. Two small
packaging patches fix Python 3.13 version extraction in LakonLab and MMCV 1.6.1.
The processor is assembled explicitly from the original cached component files.

The H200 worker quantizes frozen transformer and text-encoder linear layers to
FP8 W8A8. Unsupported dimensions and excluded output/adapter layers retain their
original precision. Transformer blocks use TorchInductor; sparse attention uses
Sol and dense attention uses PyTorch SDPA. Output is 768 × 448, 124 frames at
24 fps, with the four-step DWD adapter. First use includes model loading and
kernel compilation.

| Modal app | Responsibility | Runtime |
| --- | --- | --- |
| `nonslop-streets` | FastAPI, HTMX, DuckDB, OCR, CPU preparation | Python 3.13, CPU |
| `nonslop-video` | pi-H3 FP8 video and audio generation | Python 3.13.12, CUDA 13, H200 |
| `nonslop-world` | HY-WorldMirror video reconstruction | Python 3.12, CUDA 13, H100 |
| `nonslop-inference` | SGLang-Omni FP8 image/video understanding and transcription | Python 3.12, CUDA 13, H100 |

Model workers are private Modal classes. Omni's HTTP server listens on localhost
inside its worker. Its current Python support excludes 3.13; WorldMirror uses
3.12 to match the official CUDA 13 FlashAttention wheel. Each stack has isolated
compiler cache paths within the retained `/cache` volume.

## Access

Create the Modal secret `nonslop-access` with three different values:

- `APP_API_KEY`: at least 16 characters.
- `APP_PASSWORD`: at least 8 characters.
- `SESSION_SECRET`: at least 32 random characters.

The website opens directly, without an API key or password form. Each browser
has its own signed, Secure, HttpOnly, SameSite=Strict session, with CSRF-protected
forms. Visits, uploaded images, and videos are restricted to that browser's
workspace. Keep the browser cookies to return to its visits. Existing visits
from the former shared workspace remain accessible through the authenticated API.

API requests still require both `X-API-Key` and `X-Password` headers. These are
server-side credentials, configured by the deployment CLI rather than entered
on the website.

`.env` is ignored; use `.env.example` as its template. Fill it
yourself; credentials are never pulled back from Modal. The CLI reads `.env`
automatically. Existing environment variables override `.env`, and explicit CLI
flags override both:

```bash
uv run nonslop deploy --app cpu
uv run nonslop deploy --app cpu --env-file .env.production
uv run nonslop deploy --app cpu --api-key "$APP_API_KEY" --password "$APP_PASSWORD" --session-secret "$SESSION_SECRET"
```

Configure all three values together. When all three are absent or blank, the CLI
uses the existing Modal secret `nonslop-access`. Local values are injected as a
deployment secret; the named secret is never overwritten. Credentials are never
printed by the CLI. The CPU deployment keeps the website URL above; the video
worker is a separate private Modal app, `nonslop-video`.

## Evidence and uploads

Start a browser field note with at least one PNG, JPEG, or WebP photograph and
add a place or scene and year. Written notes are optional; the photographs anchor
the first frame and the reconstructed space. Optional UTF-8 text/Markdown or an image scan
supplies historical context; scans use English Tesseract OCR. Files are limited
to 10 MB each and images to 20 megapixels. An optional JSON research packet
(`research_file`, a scene_scout manifest or a list of evidence items with
`title`/`record_url`/`excerpt`/`kind`/`capture_from`/`capture_to` fields)
joins the visit's saved research evidence. An uploaded packet contributes up to
24 usable items with up to 1,000 characters per excerpt.

New demo visits load the matching [Flatbush](research/25-flatbush/README.md)
or [Hakodate Bay](research/hakodate-bay/README.md) evidence packet automatically.
Each packet can also be attached to a custom visit.

The HTMX form also accepts an image folder or multiple selected photographs.
Use up to eight references, 10 MB per file, and 40 MB total. References are
ordered by filename; an explicit opening photograph takes precedence. All
references are retained at up to 2048 pixels on their longest edge and inspected
by Omni to inform the prompt, including single-photo submissions. pi-H3 uses
the opening photograph as its native frame condition. Other photographs provide
supporting evidence; filename order does not force an ending view. Copies sent to
Omni are resized to at most 768 pixels on their longest edge.
AI observations are shown separately from the supplied historical evidence.

Available EXIF capture time, GPS coordinates, camera model, and focal length are
retained separately and included in the generation context and source ledger.
Normalized images carry no embedded metadata. Metadata does not override the
requested memory year or establish a camera pose; scan dates may describe
digitization. GPS is a prompt clue, not a reverse-geocoded or verified address.
Focal length alone is not treated as a calibrated intrinsic matrix.

Web grounding queries Internet Archive using all comma-separated location terms
and the exact year, then a
ten-year window if needed. Source links, excerpts, dates, and retrieval notices
are retained with each scene. Search results require relevance review; they do
not certify an exact reconstruction. H3's hosted Context-IR is not part of its
open-source release. The prompt uses H3's structured visual/sound fields and
preserves the supplied opening image.

When configured, a Modal-hosted endpoint for Kimi K3.1 Pro receives the full
accepted visit description and source text, photo metadata, retrieved and saved
research, the resized reference photographs, and Omni's photo observations.
The demo packets also include all 12 successful Flatbush DeerFlow inspections
or all 42 successful Hakodate inspections, with receipt IDs and source URLs.
The research folders retain the original archival photos and documents.
Kimi returns H3's visual and sound prompt. Create a Shared or Dedicated Endpoint for
`moonshotai/Kimi-K3.1-Pro` and a workspace proxy token, then set
`NONSLOP_PROMPT_TOKEN` (`wk-<id>.ws-<secret>`), `NONSLOP_PROMPT_BASE_URL` (your
endpoint URL ending in `/v1`), and `NONSLOP_PROMPT_MODEL` (the model ID returned
by the endpoint's `/v1/models` route). Keep these with the web credentials in
`.env` or the `nonslop-access` secret, then redeploy the CPU app. Per-token
billing applies on Shared Endpoints. Without a token, or if the endpoint fails
or returns an incomplete prompt, generation falls back to the deterministic
template. The visit record shows the prompt H3 used and identifies Kimi or the
template. For photo visits, Kimi checks Omni's descriptions against the
numbered reference images and records concrete visible details and
uncertainties for each. Supporting views contribute features when the opening
photo or established overlap supports their placement.

`POST /api/jobs` accepts the same multipart fields as the UI: `location`, `year`,
`description`, `source_text`, `source_file`, `research_file`, `reference_image`,
`use_web`, and optional `seed`. Repeat `reference_images` multipart fields for multiple
references. Poll `GET /api/jobs/{job_id}`; retrieve completed media from
`GET /api/jobs/{job_id}/video`. There is room for four outstanding jobs and one
GPU worker processes them in sequence.

After the video is ready, use **Build a 3D space**, then **Step into this space**.
WorldMirror reconstructs up to six sampled frames from the H3 video, so Kimi's
evidence-based prompt guides the visuals used for the splat. Reconstruction
uses up to a 756-pixel input edge and a 1.5-million-splat export cap. The source
ledger retains the full normalized reference photos. Previously built spaces
remain viewable with their original input type. **Rebuild this space** uses
the visit's saved video and invalidates the cached 3D artifacts.

The pinned WorldMirror export patch preserves predicted opacity during voxel
merging and writes standard PLY opacity logits. Scale-outlier filtering remains
enabled to remove excessively large splats. Edge filtering remains
enabled with less aggressive normal and depth thresholds. These changes retain
more observed surfaces; larger assets use more download bandwidth and GPU memory.

The Spark/three.js viewer supports keyboard, pointer, and touch movement and
shows input camera positions. Projection uses WorldMirror's estimated focal lengths,
principal point, and processed image dimensions rather than a fixed field of view.
Older assets without image dimensions assume a centered principal point. Geometry
is inferred and unscaled; single-photo depth and unseen surfaces remain uncertain.
Historical sources and remembered details inform generation, not measured geometry.
Spark uses its native platform-specific detail budget rather than a fixed 150k
display cap. Its blur settings match WorldMirror's classic Gaussian rasterization
rather than applying the opacity compensation intended for antialiased training.

Earlier photo-only validation on November 22, 2026 completed with both one and two
indoor example photographs. The updated two-photo rebuild produced 605,046 exported
splats (41.1 MB), two camera poses, and 756 × 504 calibration dimensions. The owner’s
3D page, camera JSON, and PLY download returned HTTP 200. This verifies the deployed reconstruction
path, not metric geometric accuracy or interactive browser rendering. The
companion 124-frame video with audio completed. Native renders of the exported
splats show improved bed and window coverage at both input views and an
interpolated viewpoint. Soft boundaries, grainy walls, floating artifacts, and
unobserved regions remain; memory-quality fidelity remains unfinished.

HTMX handles forms, uploads, polling, and view changes. The bundled JavaScript
for the 3D renderer is built with `npm ci` then `npm run build` when its source
changes; the served bundle is included in the repository.

## Persistent storage

| Modal volume | Mount | Contents |
| --- | --- | --- |
| `nonslop-models` | `/models` | pi-H3 snapshots, WorldMirror, and Omni FP8 weights |
| `nonslop-kernel-cache` | `/cache` | Inductor, Triton, CUDA JIT, CuTe, extension and LakonLab caches |
| `nonslop-scenes` | `/data` | DuckDB scene history and opening photographs |
| `nonslop-videos` | `/outputs` | Generated MP4 files |
| `nonslop-worlds` | `/worlds` | Gaussian PLY assets and camera calibration |

`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`,
`CUTE_DSL_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, and `LAKONLAB_CACHE` all point into
`/cache`. FX graph caching is enabled. The worker commits this volume after
every attempt and on normal shutdown. New shapes or software/GPU changes can
still require compilation. The upstream compiler safeguards for PyTorch 2.10
are enabled.

Only the web container writes DuckDB. Keep its maximum container count at one
and use the `recreate` deployment strategy to avoid overlapping writers.

## Checks

```bash
uv run --group checks ruff check src tests
uv run --group checks black --check src tests
uv run --group checks pytest -q tests
uv run modal run -m nonslop.runtime.modal_app
```

Verified on November 22, 2026: nine focused web/CLI tests, Ruff, and Black passed.
The live website opens without login (HTTP 200), enforces CSRF (403), and rejects
API requests without credentials (401). Browser workspace isolation, media
access, cookie restoration, and CLI/`.env` precedence are covered by tests.
Live text-to-video and image-conditioned generations completed on an H200 with
348 transformer and 473 text-encoder FP8 layers (55.2 GiB model allocation).
Both results returned authenticated HTTP 206 video ranges; unauthenticated media
access was rejected. Inductor graph artifacts remained on the Modal volume
after a worker restart. Python 3.13.12 passes GPU compilation and generation;
aligning only the DWD RMSNorm weights to FP32 restores fused dispatch. Its
isolated benchmark improved from 0.203 ms to 0.056 ms; this is not a whole-video
speedup measurement. Full generation profiles are retained under
`/cache/profiling`. A real HTMX folder workflow selected two images, generated
video, reconstructed six views, and opened the WebGL2 viewer without browser or
CSP errors. A separate saved-geometry browser fixture verified movement and
Reset against the deployed viewer. Original photo/audio retention, live voice
transcription, and workspace isolation passed through the public application.

Sources: [pi-H3](https://github.com/Lakonik/LakonLab/blob/main/docs/piH3.md),
[original H3](https://huggingface.co/MiniMaxAI/MiniMax-H3),
[NVIDIA image](https://gitlab.com/nvidia/container-images/cuda/blob/master/dist/13.0.3/ubuntu2204/devel/cudnn/Dockerfile),
[prompt enrichment reference](https://gist.github.com/Naxdy/43b7422a1e4a79fb8b0489c6c39eaace).
