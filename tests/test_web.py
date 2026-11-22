import json
import re
import wave
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr

from nonslop.data.jobs import JobStore
from nonslop.generation.schemas import (
    GenerationResult,
    GenerationTask,
    JobRecord,
    SceneRequest,
)
from nonslop.models.inference_schemas import ArchivistResult, MediaType
from nonslop.models.world_schemas import WorldResult
from nonslop.web import image_search
from nonslop.web.app import create_app
from nonslop.web.image_relevance import image_subject, nous_title_scores, title_score
from nonslop.web.schemas import WebSettings


@dataclass
class RecordingGenerator:
    submitted: list[GenerationTask] = field(default_factory=list)
    references: list[list[bytes]] = field(default_factory=list)
    completed: GenerationResult | None = None

    def submit(self, task: GenerationTask, reference_images: list[bytes]) -> str:
        self.submitted.append(task)
        self.references.append(reference_images)
        for reference_image in reference_images:
            with Image.open(BytesIO(reference_image)) as image:
                assert image.mode == "RGB"
        return task.job_id

    def result(self, call_id: str) -> GenerationResult | None:
        return self.completed


@dataclass
class RecordingReconstructor:
    submitted: list[str] = field(default_factory=list)
    completed: WorldResult | None = None

    def submit(self, job_id: str) -> str:
        self.submitted.append(job_id)
        return job_id

    def result(self, call_id: str) -> WorldResult | None:
        return self.completed


@dataclass
class RecordingArchivist:
    submitted: list[bytes] = field(default_factory=list)
    completed: ArchivistResult | None = None

    def submit(self, media: bytes, media_type: MediaType, filename: str) -> str:
        self.submitted.append(media)
        return str(uuid4())

    def result(self, call_id: str) -> ArchivistResult | None:
        return self.completed


def no_operation() -> None:
    pass


def client_settings(root: Path) -> WebSettings:
    return WebSettings(
        api_key=SecretStr("test-api-key-with-enough-entropy"),
        password=SecretStr("test-password-only"),
        session_secret=SecretStr("test-session-secret-with-enough-entropy"),
        data_root=root / "data",
        output_root=root / "outputs",
        serving_root=root / "serving",
        world_root=root / "worlds",
        world_serving_root=root / "world-serving",
        assets_root=Path(__file__).parents[1] / "src/nonslop/web",
    )


def browser_client(
    settings: WebSettings,
    worker: RecordingGenerator,
    world: RecordingReconstructor | None = None,
    archivist: RecordingArchivist | None = None,
) -> TestClient:
    return TestClient(
        create_app(
            settings,
            worker,
            world or RecordingReconstructor(),
            archivist or RecordingArchivist(),
            no_operation,
            no_operation,
            no_operation,
        ),
        base_url="https://testserver",
    )


def csrf_from(html: str) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', html)
    assert match is not None
    return match.group(1)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Empire State Building (aerial view).jpg", True),
        ("View of Empire State Building from Rockefeller Center", True),
        ("View of Manhattan from the Empire State Building", False),
        ("Manhattan from Weehawken, NJ.jpg", False),
        ("Empire State Building blueprint", False),
    ],
)
def test_image_titles_must_indicate_the_requested_subject(
    title: str, expected: bool
) -> None:
    assert (
        bool(title_score(title, image_subject("empire state building", ""))) == expected
    )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_nous_scores_titles_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    titles = ["Empire State Building at night", "City panorama"]

    def response(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "inference-api.nousresearch.com"
        assert request.headers["Authorization"] == "Bearer test-key"
        prompt = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert prompt["titles"] == titles
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"scores": [4, 1]}'}}]}
        )

    async_client = httpx.AsyncClient
    monkeypatch.setenv("NOUS_API_KEY", "test-key")
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(
            transport=httpx.MockTransport(response), **kwargs
        ),
    )
    assert await nous_title_scores("Empire State Building", titles) == [4, 1]


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_commons_preview_and_generation_use_the_same_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    photo = BytesIO()
    Image.new("RGB", (12, 10), "blue").save(photo, format="PNG")
    requests: list[httpx.URL] = []

    def commons_images(request: httpx.Request) -> httpx.Response:
        requests.append(request.url)
        if request.url.host == "commons.wikimedia.org":
            assert request.url.params["generator"] == "search"
            assert request.url.params["gsrnamespace"] == "6"
            assert (
                request.url.params["gsrsearch"]
                == "Hester Street 1905 Brick storefronts filetype:bitmap"
            )
            assert "api_key" not in request.url.params
            return httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {
                                "index": 1,
                                "title": "File:Hester Street photograph",
                                "imageinfo": [
                                    {"thumburl": "https://other.test/untrusted"}
                                ],
                            },
                            {
                                "index": 2,
                                "title": "File:Hester Street storefronts",
                                "imageinfo": [
                                    {
                                        "thumburl": "https://thumb.wikimedia.org/wikipedia/commons/thumb/a/ab/street.jpg/500px-street.jpg"
                                    },
                                ],
                            },
                            {
                                "index": 3,
                                "title": "File:Different street",
                                "imageinfo": [
                                    {
                                        "thumburl": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/unrelated.jpg/500px-unrelated.jpg"
                                    },
                                ],
                            },
                        ]
                    }
                },
            )
        assert request.url.host == "upload.wikimedia.org"
        return httpx.Response(
            200, content=photo.getvalue(), headers={"Content-Type": "image/png"}
        )

    async_client = httpx.AsyncClient
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    monkeypatch.setattr(image_search, "library_results", lambda query, limit: [])
    monkeypatch.setattr(image_search, "archive_results", lambda query, limit: [])
    monkeypatch.setattr(image_search, "museum_results", lambda query, limit: [])
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(
            transport=httpx.MockTransport(commons_images), **kwargs
        ),
    )
    settings = client_settings(tmp_path)
    worker = RecordingGenerator()
    inputs = {
        "location": "Hester Street",
        "year": "1905",
        "description": "Brick storefronts",
    }

    def transport(web_settings: WebSettings) -> httpx.ASGITransport:
        return httpx.ASGITransport(
            app=create_app(
                web_settings,
                worker,
                RecordingReconstructor(),
                RecordingArchivist(),
                no_operation,
                no_operation,
                no_operation,
            )
        )

    async with async_client(
        transport=transport(settings.model_copy(update={"references_only": True})),
        base_url="https://testserver",
    ) as preview:
        token = csrf_from((await preview.get("/")).text)
        gallery = await preview.post("/", data={**inputs, "csrf": token})
        assert gallery.status_code == 200
        image = re.search(r'data:image/png;base64,([^"]+)', gallery.text)
        assert image is not None
        assert 'alt="Hester Street storefronts"' in gallery.text
        assert (await preview.post("/jobs")).status_code == 404
        assert not worker.submitted

    async with async_client(
        transport=transport(settings), base_url="https://testserver"
    ) as website:
        token = csrf_from((await website.get("/")).text)
        submitted = await website.post(
            "/jobs", data={**inputs, "csrf": token, "use_web": "true"}
        )
        assert submitted.status_code == 200
        assert worker.references == [[b64decode(image.group(1))]]
        assert len(requests) == 4
        assert requests[0].params["gsrlimit"] == "32"
        assert requests[2].params["gsrlimit"] == "8"
        assert "Begin on the supplied first-frame image." in worker.submitted[0].prompt


def api_headers(settings: WebSettings) -> dict[str, str]:
    return {
        "X-API-Key": settings.api_key.get_secret_value(),
        "X-Password": settings.password.get_secret_value(),
    }


def test_public_site_keeps_api_credentials_and_browser_csrf(tmp_path: Path) -> None:
    settings = client_settings(tmp_path)
    client = browser_client(settings, RecordingGenerator())
    home = client.get("/", follow_redirects=False)
    assert home.status_code == 200
    assert "Build your street" in home.text
    assert 'name="api_key"' not in home.text and 'name="password"' not in home.text
    cookie = home.headers["set-cookie"].lower()
    assert "httponly" in cookie and "secure" in cookie and "samesite=strict" in cookie
    assert client.get("/login", follow_redirects=False).headers["location"] == "/"
    assert client.post("/api/jobs").status_code == 401
    assert (
        client.post(
            "/api/jobs", headers={"X-API-Key": settings.api_key.get_secret_value()}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/jobs", headers={"X-Password": settings.password.get_secret_value()}
        ).status_code
        == 401
    )
    assert client.post("/jobs").status_code == 403
    assert client.post("/jobs", data={"csrf": "wrong"}).status_code == 403


def test_image_and_text_job_survives_restart_and_protects_video(tmp_path: Path) -> None:
    settings = client_settings(tmp_path)
    worker = RecordingGenerator()
    client = browser_client(settings, worker)
    image_data = BytesIO()
    Image.new("RGBA", (64, 64), "red").save(image_data, format="PNG")
    response = client.post(
        "/api/jobs",
        headers=api_headers(settings),
        data={
            "location": "Hester Street, New York",
            "year": "1905",
            "description": "A slow walk along the brick storefronts.",
            "use_web": "false",
        },
        files={
            "reference_image": ("street.png", image_data.getvalue(), "image/png"),
            "source_file": ("news.txt", b"A market occupied the street.", "text/plain"),
        },
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    assert response.json()["has_reference"] is True
    assert "A market occupied" in worker.submitted[0].prompt
    settings.output_root.mkdir()
    video = settings.output_root / f"{job_id}.mp4"
    video.write_bytes(b"private-video")
    worker.completed = GenerationResult(
        video_path=str(video), width=768, height=448, frames=124
    )
    assert (
        client.get(f"/api/jobs/{job_id}", headers=api_headers(settings)).json()[
            "status"
        ]
        == "complete"
    )
    restarted = browser_client(settings, RecordingGenerator())
    assert (
        restarted.get(f"/api/jobs/{job_id}", headers=api_headers(settings)).json()[
            "status"
        ]
        == "complete"
    )
    assert restarted.get(f"/jobs/{job_id}/video").status_code == 404
    assert (
        restarted.get(
            f"/api/jobs/{job_id}/video", headers=api_headers(settings)
        ).content
        == b"private-video"
    )


def test_browser_jobs_are_isolated_and_survive_session_restore(tmp_path: Path) -> None:
    settings = client_settings(tmp_path)
    worker = RecordingGenerator()
    world = RecordingReconstructor()
    owner = browser_client(settings, worker, world)
    visitor = browser_client(settings, worker)
    token = csrf_from(owner.get("/").text)
    image_data = BytesIO()
    Image.new("RGB", (16, 16), "red").save(image_data, format="PNG")
    submitted = owner.post(
        "/jobs",
        data={
            "csrf": token,
            "location": "Private remembered street",
            "year": "1905",
            "description": "A street remembered by its owner.",
            "use_web": "false",
        },
        files={"reference_image": ("street.png", image_data.getvalue(), "image/png")},
    )
    assert submitted.status_code == 200, submitted.text
    job_id = worker.submitted[0].job_id
    assert owner.post(f"/jobs/{job_id}/world", data={"csrf": token}).status_code == 409
    assert job_id in owner.get("/").text
    assert job_id not in visitor.get("/").text
    for suffix in ("", "/reference", "/video"):
        assert visitor.get(f"/jobs/{job_id}{suffix}").status_code == 404
    assert owner.get(f"/jobs/{job_id}/reference").status_code == 200
    settings.output_root.mkdir()
    video = settings.output_root / f"{job_id}.mp4"
    video.write_bytes(b"private-video")
    worker.completed = GenerationResult(
        video_path=str(video), width=768, height=448, frames=124
    )
    assert (
        owner.get(f"/jobs/{job_id}/video", headers={"Range": "bytes=0-3"}).status_code
        == 206
    )
    assert owner.post(f"/jobs/{job_id}/world").status_code == 403
    visitor_csrf = csrf_from(visitor.get("/").text)
    assert (
        visitor.post(f"/jobs/{job_id}/world", data={"csrf": visitor_csrf}).status_code
        == 404
    )
    for attempt in range(2):
        assert (
            owner.post(f"/jobs/{job_id}/world", data={"csrf": token}).status_code == 200
        )
    assert world.submitted == [job_id]
    world_directory = settings.world_root / job_id
    world_directory.mkdir(parents=True)
    (world_directory / "gaussians.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nend_header\n"
    )
    (world_directory / "camera_params.json").write_text('{"num_cameras": 1}')
    world.completed = WorldResult(
        job_id=job_id,
        ply_path=str(world_directory / "gaussians.ply"),
        camera_path=str(world_directory / "camera_params.json"),
    )
    assert "Step into this space" in owner.get(f"/jobs/{job_id}").text
    assert owner.get(f"/jobs/{job_id}/world").status_code == 200
    assert owner.get(f"/jobs/{job_id}/world/gaussians.ply").content.startswith(b"ply\n")
    for suffix in ("", "/gaussians.ply", "/cameras"):
        assert visitor.get(f"/jobs/{job_id}/world{suffix}").status_code == 404
    restored = browser_client(settings, RecordingGenerator())
    restored.cookies.update(owner.cookies)
    assert job_id in restored.get("/").text
    assert restored.get(f"/jobs/{job_id}/video").content == b"private-video"


def test_invalid_upload_does_not_start_generation(tmp_path: Path) -> None:
    settings = client_settings(tmp_path)
    worker = RecordingGenerator()
    client = browser_client(settings, worker)
    response = client.post(
        "/api/jobs",
        headers=api_headers(settings),
        data={
            "location": "Hester Street",
            "year": "1905",
            "description": "A short walk.",
            "use_web": "false",
        },
        files={"reference_image": ("image.png", b"not an image", "image/png")},
    )
    assert response.status_code == 400
    assert worker.submitted == []


def test_capsule_timeline_preserves_originals_and_browser_ownership(
    tmp_path: Path,
) -> None:
    settings = client_settings(tmp_path)
    archivist = RecordingArchivist()
    worker = RecordingGenerator()
    world = RecordingReconstructor()
    owner = browser_client(settings, worker, world, archivist)
    visitor = browser_client(settings, RecordingGenerator())
    token = csrf_from(owner.get("/capsules").text)
    image_data = BytesIO()
    Image.new("RGB", (16, 16), "red").save(image_data, format="PNG")
    response = owner.post(
        "/capsules",
        data={
            "csrf": token,
            "title": "My old bear",
            "description": "The one from home.",
        },
        files={"anchor": ("bear.png", image_data.getvalue(), "image/png")},
    )
    assert response.status_code == 200, response.text
    capsule_match = re.search(r'data-capsule-id="([^"]+)"', response.text)
    assert capsule_match is not None
    capsule_id = capsule_match.group(1)
    assert owner.get(f"/capsules/{capsule_id}/anchor").content == image_data.getvalue()
    assert "My old bear" not in visitor.get("/capsules").text
    assert visitor.get(f"/capsules/{capsule_id}").status_code == 404
    assert visitor.get(f"/capsules/{capsule_id}/anchor").status_code == 404
    assert (
        owner.post(
            f"/capsules/{capsule_id}/entries", data={"text": "Missing CSRF"}
        ).status_code
        == 403
    )
    added = owner.post(
        f"/capsules/{capsule_id}/entries",
        data={
            "csrf": token,
            "occurred_on": "2020-05-02",
            "text": "These are my original words.",
        },
    )
    assert added.status_code == 200, added.text
    audio_data = BytesIO()
    with wave.open(audio_data, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16000)
        recording.writeframes(bytes(320000))
    added = owner.post(
        f"/capsules/{capsule_id}/entries",
        data={
            "csrf": token,
            "occurred_on": "1999-02-03",
            "text": "My recorded memory.",
            "use_ai": "true",
        },
        files={"media": ("voice.wav", audio_data.getvalue(), "audio/wav")},
    )
    assert added.status_code == 200, added.text
    assert added.text.index("03 Feb 1999") < added.text.index("02 May 2020")
    assert archivist.submitted == [audio_data.getvalue()]
    entry_match = re.search(r'data-entry-id="([^"]+)"', added.text)
    assert entry_match is not None
    media_url = f"/capsules/{capsule_id}/entries/{entry_match.group(1)}/media"
    assert owner.get(media_url).content == audio_data.getvalue()
    assert visitor.get(media_url).status_code == 404
    archivist.completed = ArchivistResult(text="A machine transcript.")
    whole = owner.get(f"/capsules/{capsule_id}/entries?view=whole")
    assert "capsule-gallery" in whole.text
    assert "AI transcription" in whole.text and "A machine transcript." in whole.text
    assert (
        "These are my original words." in whole.text
        and "My recorded memory." in whole.text
    )
    restored = browser_client(settings, RecordingGenerator())
    restored.cookies.update(owner.cookies)
    assert "A machine transcript." in restored.get(f"/capsules/{capsule_id}").text
    assert restored.get(media_url).content == audio_data.getvalue()
    visit = owner.post(
        "/jobs",
        data={
            "csrf": token,
            "location": "Hester Street",
            "year": "1905",
            "description": "A remembered storefront.",
            "use_web": "false",
        },
    )
    assert visit.status_code == 200, visit.text
    job_id = worker.submitted[0].job_id
    worker.completed = GenerationResult(
        video_path="/outputs/test.mp4", width=768, height=448, frames=124
    )
    assert owner.post(f"/jobs/{job_id}/world", data={"csrf": token}).status_code == 200
    world.completed = WorldResult(
        job_id=job_id,
        ply_path="/worlds/test/gaussians.ply",
        camera_path="/worlds/test/camera_params.json",
    )
    correction = owner.post(
        f"/capsules/{capsule_id}/entries",
        data={
            "csrf": token,
            "occurred_on": "2006-05-01",
            "text": "The shop sign was green.",
            "correction": "true",
            "linked_job_id": job_id,
        },
    )
    assert correction.status_code == 200, correction.text
    linked = owner.get(f"/visits/{job_id}")
    assert (
        "Step into this space" in linked.text
        and "The shop sign was green." in linked.text
    )
    assert "The shop sign was green." in owner.get(f"/jobs/{job_id}/world").text


def test_queue_admission_is_atomic(tmp_path: Path) -> None:
    store = JobStore(tmp_path, no_operation)
    scene = SceneRequest(
        location="Hester Street", year=1905, description="A quiet street."
    )
    jobs = [
        JobRecord(
            job_id=str(uuid4()),
            created_at=datetime.now(UTC),
            status="queued",
            scene=scene,
            render_prompt="A street.",
        )
        for job_number in range(8)
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        reserved = list(executor.map(lambda job: store.reserve(job, []), jobs))
    assert sum(reserved) == 4
    assert store.active_count() == 4


def test_folder_images_are_ordered_retained_bounded_and_private(tmp_path: Path) -> None:
    settings = client_settings(tmp_path)
    worker = RecordingGenerator()
    owner = browser_client(settings, worker)
    token = csrf_from(owner.get("/").text)
    uploads: list[tuple[str, tuple[str, bytes, str]]] = []
    for name, color in (("03.jfif", "blue"), ("01.png", "red"), ("02", "green")):
        content = BytesIO()
        Image.new("RGB", (1600, 800), color).save(content, format="PNG")
        uploads.append(("reference_images", (name, content.getvalue(), "image/png")))
    submitted = owner.post(
        "/jobs",
        data={
            "csrf": token,
            "location": "Hester Street",
            "year": "1905",
            "description": "A street viewed in several photographs.",
            "use_web": "false",
        },
        files=uploads,
    )
    assert submitted.status_code == 200, submitted.text
    job_id = worker.submitted[0].job_id
    record = owner.get(f"/api/jobs/{job_id}", headers=api_headers(settings)).json()
    assert record["reference_names"] == ["01.png", "02", "03.jfif"]
    assert record["reference_count"] == 3
    assert len(worker.references[0]) == 3
    with Image.open(BytesIO(worker.references[0][0])) as opening:
        assert opening.size == (768, 384) and opening.getpixel((0, 0)) == (255, 0, 0)
    with Image.open(BytesIO(worker.references[0][-1])) as ending:
        assert ending.getpixel((0, 0)) == (0, 0, 255)
    stranger = browser_client(settings, RecordingGenerator())
    for index in range(3):
        assert owner.get(f"/jobs/{job_id}/references/{index}").status_code == 200
        assert stranger.get(f"/jobs/{job_id}/references/{index}").status_code == 404
    oversized = owner.post(
        "/api/jobs",
        headers=api_headers(settings),
        data={
            "location": "Hester Street",
            "year": "1905",
            "description": "Too many references.",
            "use_web": "false",
        },
        files=uploads * 3,
    )
    assert oversized.status_code == 413
    assert len(worker.submitted) == 1
