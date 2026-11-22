# Memory Worlds

A prototype for revisiting a lost street from photographs, descriptions,
newspaper scans, and archive material. FastAPI serves a small HTMX interface;
pi-H3 generates a 5.2-second street visit with audio on Modal. WorldMirror turns
the video into a Gaussian-splat space; dated time capsules hold original notes,
photos, audio, video, and linked visits.

The implementation roadmap is maintained in [todo.md](todo.md): video → 3D
space with HY-WorldMirror → time capsules, with an isolated SGLang-Omni service.

Existing deployment (not this redesign): https://afrenkai--nonslop-streets-web.us-east.modal.run

## Local CPU preview

From the repo root with Python 3.13, install once with `uv sync --locked --group checks`. Then start the browser app without network access, model downloads, or Modal credentials:

```bash
uv run --offline --no-sync python scripts/local_preview.py
```

Open **http://127.0.0.1:8742/**. Use one server at a time; stop it with Ctrl+C. Capsules, dated entries, and original media use the real local store. Keep the browser cookie to return after restarting. On Windows, the database, uploads and session key live under `%LOCALAPPDATA%\NonSlop\cpu-preview`; on other systems they live under `~/.local/share/NonSlop/cpu-preview`. Keep this directory private; never serve it with a static file server.

Visit submissions save supplied text and reference images, even if web search is checked. No archive lookup, video, 3D space, transcript or AI observations are produced. Worker requests show an offline result instead of polling. Image scans need local English Tesseract; audio/video uploads need `ffprobe` on `PATH`. Text/Markdown and photographs need neither.

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

Enter a street/place, year, and description. An optional PNG, JPEG, or WebP
photograph anchors the first frame. Optional UTF-8 text/Markdown or an image scan
supplies historical context; scans use English Tesseract OCR. Files are limited
to 10 MB each and images to 20 megapixels.

The HTMX form also accepts an image folder or multiple selected photographs.
Use up to eight references, 10 MB per file, and 40 MB total. References are
ordered by filename; an explicit opening photograph takes precedence. All
references are retained and compared by Omni to inform the prompt. pi-H3 uses
the first and last as native frame conditions. Multi-image references are
resized to at most 768 pixels on their longest edge for the vision context.
AI observations are shown separately from the supplied historical evidence.

Web grounding queries Internet Archive for the street and exact year, then a
ten-year window if needed. Source links, excerpts, dates, and retrieval notices
are retained with each scene. Search results require relevance review; they do
not certify an exact reconstruction. H3's hosted Context-IR is not part of its
open-source release. The prompt uses H3's structured visual/sound fields and
preserves the supplied opening image.

`POST /api/jobs` accepts the same multipart fields as the UI: `location`, `year`,
`description`, `source_text`, `source_file`, `reference_image`, `use_web`, and
optional `seed`. Repeat `reference_images` multipart fields for multiple
references. Poll `GET /api/jobs/{job_id}`; retrieve completed media from
`GET /api/jobs/{job_id}/video`. There is room for four outstanding jobs and one
GPU worker processes them in sequence.

After a video finishes, use **Build a 3D space**, then **Step into this space**.
The Spark/three.js viewer supports keyboard, pointer, and touch movement and
shows input camera positions. Reconstruction uses six sampled frames and a
500k input-splat cap. Geometry is inferred and unscaled; gaps and distorted
regions do not establish what was historically present.

Open **Time capsules** to anchor an object or place with a photograph. Add dated
entries in your own words, upload original image/audio/video files, and link
visits or corrections. Whole-capsule and timeline views use HTMX; undocumented
periods remain empty. Optional transcripts and observations are labeled AI
output and never replace original entries. Original media stays within its
browser workspace.

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
