import json
import re
import wave
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr

from nonslop.capsules.schemas import CapsuleEntry, CapsuleRecord
from nonslop.capsules.store import CapsuleStore
from nonslop.data.jobs import JobStore
from nonslop.generation.prompt_writer import compose_scene_prompt, write_scene_prompt
from nonslop.generation.schemas import (
    GenerationResult,
    GenerationTask,
    JobRecord,
    SceneRequest,
)
from nonslop.models.inference_schemas import ArchivistResult, MediaType
from nonslop.models.world_schemas import WorldResult
from nonslop.reconstruction.schemas import ReconstructionRecord
from nonslop.web.app import create_app
from nonslop.web.schemas import WebSettings
from nonslop.web.uploads import normalize_image


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
    references: list[list[bytes]] = field(default_factory=list)
    completed: WorldResult | None = None

    def submit(self, job_id: str, reference_images: list[bytes]) -> str:
        self.submitted.append(job_id)
        self.references.append(reference_images)
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


def template_context(response: object) -> dict:
    # Starlette's TestClient adds template context to the underlying httpx response.
    context = getattr(response, "context", None)
    assert isinstance(context, dict)
    return context


def capsule_id_from(html: str) -> str:
    match = re.search(r'data-capsule-id="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def api_headers(settings: WebSettings) -> dict[str, str]:
    return {
        "X-API-Key": settings.api_key.get_secret_value(),
        "X-Password": settings.password.get_secret_value(),
    }


def test_field_guide_is_static_and_leads_to_the_dashboard(tmp_path: Path) -> None:
    client = browser_client(client_settings(tmp_path), RecordingGenerator())
    home = client.get("/")
    assert home.status_code == 200
    assert "Memory Worlds" in home.text
    for part in ("photograph", "keepsake"):
        assert f'id="{part}"' in home.text
    assert 'href="/dashboard"' in home.text
    assert "<form" not in home.text
    assert "set-cookie" not in home.headers
    assert {"jobs", "capsules", "csrf"}.isdisjoint(template_context(home))
    assert client.get("/static/kitchen.svg").status_code == 200
    assert "Memory Worlds" in client.get("/capsules").text


def test_dashboard_pages_the_two_legacy_collections_independently(
    tmp_path: Path,
) -> None:
    settings = client_settings(tmp_path)
    jobs = JobStore(settings.data_root, no_operation)
    capsules = CapsuleStore(jobs)
    start = datetime.now(UTC)
    for number in range(31):
        timestamp = start + timedelta(seconds=number)
        owner = f"old-workspace-{number}"
        capsules.create(
            CapsuleRecord(
                capsule_id=uuid4(),
                owner_id=owner,
                created_at=timestamp,
                title=f"Notebook {number:02}",
            )
        )
        if number < 14:
            jobs.save(
                JobRecord(
                    job_id=str(uuid4()),
                    owner_id="api" if number == 13 else owner,
                    created_at=timestamp,
                    status="failed",
                    render_prompt="A street.",
                    scene=SceneRequest(
                        location=f"Street {number:02}",
                        year=1990,
                        description="The old market.",
                    ),
                )
            )
    client = browser_client(settings, RecordingGenerator())
    for box_page, capsule_page in ((1, 1), (2, 1), (1, 2), (2, 2)):
        response = client.get(
            "/dashboard", params={"box_page": box_page, "capsule_page": capsule_page}
        )
        assert response.status_code == 200
        context = template_context(response)
        assert (
            context["box_page"] == box_page and context["capsule_page"] == capsule_page
        )
        assert context["more_jobs"] is (box_page == 1)
        assert context["more_capsules"] is (capsule_page == 1)
        assert context["csrf"]
        assert len(context["jobs"]) == (12 if box_page == 1 else 1)
        assert len(context["capsules"]) == (30 if capsule_page == 1 else 1)
        assert ("Street 00" in response.text) == (box_page == 2)
        assert ("Notebook 00" in response.text) == (capsule_page == 2)
        assert (
            "Street 13" not in response.text and "old-workspace-" not in response.text
        )
    assert client.get("/dashboard?box_page=0").status_code == 422
    assert client.get("/dashboard?capsule_page=-1").status_code == 422


def test_unrelated_legacy_workspaces_can_link_a_capsule_to_a_visit(
    tmp_path: Path,
) -> None:
    settings = client_settings(tmp_path)
    jobs = JobStore(settings.data_root, no_operation)
    capsules = CapsuleStore(jobs)
    now = datetime.now(UTC)
    job = JobRecord(
        job_id=str(uuid4()),
        owner_id="legacy-photo-workspace",
        created_at=now,
        status="complete",
        scene=SceneRequest(
            location="Market square", year=1990, description="The clock tower."
        ),
        render_prompt="The square.",
        world=ReconstructionRecord(status="complete"),
    )
    jobs.save(job)
    capsule = CapsuleRecord(
        capsule_id=uuid4(),
        owner_id="legacy-notebook-workspace",
        created_at=now,
        title="A pocket watch",
    )
    capsules.create(capsule)
    capsules.append(
        CapsuleEntry(
            entry_id=uuid4(),
            capsule_id=capsule.capsule_id,
            owner_id=capsule.owner_id,
            created_at=now,
            occurred_on=now.date(),
            kind="text",
            text="An older note.",
        )
    )

    client = browser_client(settings, RecordingGenerator())
    url = f"/capsules/{capsule.capsule_id}"
    page = client.get(url)
    assert "An older note." in page.text
    assert f'value="{job.job_id}"' in page.text
    linked = client.post(
        f"{url}/entries",
        data={
            "csrf": csrf_from(page.text),
            "linked_job_id": job.job_id,
            "occurred_on": "1990-05-06",
            "text": "The clock was blue.",
            "correction": "true",
        },
    )
    assert linked.status_code == 200
    for target in (f"/visits/{job.job_id}", f"/jobs/{job.job_id}/world"):
        html = client.get(target).text
        assert "The clock was blue." in html and f'href="{url}"' in html
        for private_value in (
            job.owner_id,
            capsule.owner_id,
            *api_headers(settings).values(),
            settings.session_secret.get_secret_value(),
        ):
            assert private_value not in html and private_value not in page.text


def test_photo_intake_opens_the_saved_visit(tmp_path: Path) -> None:
    settings = client_settings(tmp_path)
    worker = RecordingGenerator()
    client = browser_client(settings, worker)
    assert 'href="/visits/new"' in client.get("/").text
    intake = client.get("/visits/new")
    assert intake.status_code == 200
    assert 'name="reference_image"' in intake.text
    assert client.post("/visits/new").status_code == 403
    fields = {
        "csrf": csrf_from(intake.text),
        "location": "Mill Street",
        "year": "1994",
        "description": "The old mill by the river.",
    }
    for use_htmx in (False, True):
        response = client.post(
            "/visits/new",
            data=fields,
            headers={"HX-Request": "true"} if use_htmx else {},
            follow_redirects=False,
        )
        assert response.status_code == (200 if use_htmx else 303)
        target = response.headers["HX-Redirect" if use_htmx else "Location"]
        assert target == f"/visits/{worker.submitted[-1].job_id}"
        assert "Mill Street" in client.get(target).text
        assert "Mill Street" in browser_client(settings, worker).get(target).text


def test_demo_intake_uses_the_bundled_photo_and_skips_web_lookup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("NONSLOP_PROMPT_TOKEN", "test-token")
    settings = client_settings(tmp_path)
    worker = RecordingGenerator()
    client = browser_client(settings, worker)
    intake = client.get("/visits/new")
    csrf = csrf_from(intake.text)
    visits = (
        (
            "hakodate",
            "Hakodate Red Brick Warehouse",
            2023,
            "hakodate-red-brick-warehouse.png",
            "Kanemori warehouse façades · Hakotamu",
            (
                "on the water in front of the Red Brick Warehouses",
                "Falling snow is unconfirmed.",
            ),
        ),
        (
            "flatbush",
            "25 Flatbush Ave",
            1990,
            "block-2106-lot-0026.jpg",
            "NYC tax photograph · Lot 26",
            (
                "Certificate 231396",
                "six years after the target year",
                "View this as a hypothesis.",
            ),
        ),
    )
    for preset, location, year, filename, source_title, key_findings in visits:
        assert f"{location}, {year}" in intake.text
        photo = client.get(f"/static/demo-visits/{filename}")
        assert photo.status_code == 200

        submitted = client.post(
            "/visits/new",
            data={
                "csrf": csrf,
                "visit_preset": preset,
                "use_web": "true",
                "description": "A remembered view. " * 180,
                "source_text": "Additional historical context. " * 200,
            },
            follow_redirects=False,
        )
        assert submitted.status_code == 303
        job_id = worker.submitted[-1].job_id
        record = client.get(f"/api/jobs/{job_id}", headers=api_headers(settings)).json()
        assert record["scene"]["location"] == location
        assert record["scene"]["year"] == year
        assert record["scene"]["use_web"] is False
        assert record["reference_names"] == [filename]
        assert worker.references[-1] == [normalize_image(photo.content)]
        assert record["sources"] == []
        opening_source = record["research_sources"][0]
        assert opening_source["title"] == source_title
        assert opening_source["excerpt"] in worker.submitted[-1].prompt

        def capture_writer_request(
            request: httpx.Request, findings: tuple[str, ...] = key_findings
        ) -> httpx.Response:
            packet = json.loads(request.content)["messages"][1]["content"]
            for finding in findings:
                assert finding in packet
            return httpx.Response(
                200,
                json=writer_payload(
                    "integrated_multimodal_description: [Shot 1] A continuous view.\n\n"
                    "overall_soundscape: Quiet surroundings.\n\n"
                    "non_diegetic_music: N/A"
                ),
            )

        with httpx.Client(
            transport=httpx.MockTransport(capture_writer_request),
            base_url="https://writer.test/v1",
        ) as writer:
            assert write_scene_prompt(worker.submitted[-1].prompt, client=writer)

        for path in (submitted.headers["Location"], f"/jobs/{job_id}"):
            page = client.get(path)
            assert "Creating your visit" in page.text
            assert "Research evidence" in page.text
            assert opening_source["url"] in page.text
            assert opening_source["excerpt"] in page.text

        older = JobRecord.model_validate(record)
        older.research_sources = []
        older.render_prompt = "A continuous visit using the opening photograph."
        JobStore(settings.data_root, no_operation).save(older)
        page = client.get(submitted.headers["Location"])
        assert opening_source["excerpt"] in page.text
        assert "This visit started before the research notes were added" in page.text


def test_first_intake_submissions_from_two_tabs_share_the_box(tmp_path: Path) -> None:
    first_tab = browser_client(client_settings(tmp_path), RecordingGenerator())
    intake = first_tab.get("/visits/new")
    assert intake.status_code == 200
    csrf = csrf_from(intake.text)

    # Both tabs submit with the intake cookie, before either sees the other's response.
    second_tab = TestClient(first_tab.app, base_url=str(first_tab.base_url))
    second_tab.cookies.update(intake.cookies)
    visits = []
    for tab, location in ((first_tab, "Mill Street"), (second_tab, "River Road")):
        response = tab.post(
            "/visits/new",
            data={
                "csrf": csrf,
                "location": location,
                "year": "1994",
                "description": location,
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        visits.append(response.headers["Location"])

    assert visits[0] != visits[1]
    for tab in (first_tab, second_tab):
        assert [tab.get(visit).status_code for visit in visits] == [200, 200]


def test_public_site_keeps_api_credentials_and_browser_csrf(tmp_path: Path) -> None:
    settings = client_settings(tmp_path)
    client = browser_client(settings, RecordingGenerator())
    home = client.get("/", follow_redirects=False)
    assert home.status_code == 200
    assert 'name="api_key"' not in home.text and 'name="password"' not in home.text
    intake = client.get("/visits/new")
    for value in api_headers(settings).values():
        assert value not in home.text and value not in intake.text
    cookie = intake.headers["set-cookie"].lower()
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
    assert client.post("/capsules", data={"title": "No token"}).status_code == 403
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
    for suffix in (
        "",
        "/video",
        "/reference",
        "/references/0",
        "/world",
        "/world/cameras",
        "/world/gaussians.ply",
    ):
        assert restarted.get(f"/jobs/{job_id}{suffix}").status_code == 404
    assert restarted.get(f"/visits/{job_id}").status_code == 404
    assert job_id not in restarted.get("/dashboard").text
    token = csrf_from(restarted.get("/capsules").text)
    assert (
        restarted.post(f"/jobs/{job_id}/world", data={"csrf": token}).status_code == 404
    )
    capsule = restarted.post("/capsules", data={"title": "Family album", "csrf": token})
    capsule_id = capsule_id_from(capsule.text)
    assert job_id not in restarted.get(f"/capsules/{capsule_id}").text
    assert (
        restarted.post(
            f"/capsules/{capsule_id}/entries",
            data={"csrf": token, "occurred_on": "1990-02-01", "linked_job_id": job_id},
        ).status_code
        == 404
    )
    for suffix in ("", "/video", "/world/cameras", "/world/gaussians.ply"):
        assert restarted.get(f"/api/jobs/{job_id}{suffix}").status_code == 401
    assert restarted.post(f"/api/jobs/{job_id}/world").status_code == 401
    assert (
        restarted.get(
            f"/api/jobs/{job_id}/video", headers=api_headers(settings)
        ).content
        == b"private-video"
    )


def test_browser_visits_and_worlds_are_public_after_restart(tmp_path: Path) -> None:
    settings = client_settings(tmp_path)
    worker = RecordingGenerator()
    world = RecordingReconstructor()
    owner = browser_client(settings, worker, world)
    visitor = TestClient(owner.app, base_url=str(owner.base_url))
    token = csrf_from(owner.get("/visits/new").text)
    image_data = BytesIO()
    Image.new("RGB", (16, 16), "red").save(image_data, format="PNG")
    submitted = owner.post(
        "/jobs",
        data={
            "csrf": token,
            "location": "Mill Street",
            "year": "1905",
            "description": "A street remembered by its owner.",
            "use_web": "false",
        },
        files={"reference_image": ("street.png", image_data.getvalue(), "image/png")},
    )
    assert submitted.status_code == 200, submitted.text
    job_id = worker.submitted[0].job_id
    assert owner.post(f"/jobs/{job_id}/world", data={"csrf": token}).status_code == 409
    assert job_id not in owner.get("/").text
    assert job_id in visitor.get("/dashboard").text
    assert visitor.get(f"/jobs/{job_id}").status_code == 200
    assert visitor.get(f"/jobs/{job_id}/video").status_code == 409
    assert visitor.get(f"/jobs/{job_id}/reference").content == worker.references[0][0]
    settings.output_root.mkdir()
    video = settings.output_root / f"{job_id}.mp4"
    video.write_bytes(b"private-video")
    worker.completed = GenerationResult(
        video_path=str(video), width=768, height=448, frames=124
    )
    assert (
        visitor.get(f"/jobs/{job_id}/video", headers={"Range": "bytes=0-3"}).status_code
        == 206
    )
    assert owner.post(f"/jobs/{job_id}/world").status_code == 403
    visitor_csrf = csrf_from(visitor.get("/visits/new").text)
    assert (
        visitor.post(f"/jobs/{job_id}/world", data={"csrf": token}).status_code == 403
    )
    assert (
        visitor.post(f"/jobs/{job_id}/world", data={"csrf": visitor_csrf}).status_code
        == 200
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
    assert f'href="/jobs/{job_id}/world"' in visitor.get(f"/jobs/{job_id}").text
    assert visitor.get(f"/jobs/{job_id}/world").status_code == 200
    assert visitor.get(f"/jobs/{job_id}/world/gaussians.ply").content.startswith(
        b"ply\n"
    )
    assert visitor.get(f"/jobs/{job_id}/world/cameras").json() == {"num_cameras": 1}
    restored = browser_client(settings, RecordingGenerator())
    assert job_id in restored.get("/dashboard").text
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


def test_capsule_timeline_and_original_media_are_shared_across_browsers(
    tmp_path: Path,
) -> None:
    settings = client_settings(tmp_path)
    archivist = RecordingArchivist()
    worker = RecordingGenerator()
    world = RecordingReconstructor()
    owner = browser_client(settings, worker, world, archivist)
    visitor = TestClient(owner.app, base_url=str(owner.base_url))
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
    capsule_id = capsule_id_from(response.text)
    assert owner.get(f"/capsules/{capsule_id}/anchor").content == image_data.getvalue()
    assert "My old bear" in visitor.get("/capsules").text
    assert capsule_id in visitor.get("/dashboard").text
    visitor_token = csrf_from(visitor.get(f"/capsules/{capsule_id}").text)
    assert (
        visitor.get(f"/capsules/{capsule_id}/anchor").content == image_data.getvalue()
    )
    assert (
        owner.post(
            f"/capsules/{capsule_id}/entries", data={"text": "Missing CSRF"}
        ).status_code
        == 403
    )
    added = visitor.post(
        f"/capsules/{capsule_id}/entries",
        data={
            "csrf": visitor_token,
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
    assert visitor.get(media_url).content == audio_data.getvalue()
    other = visitor.post(
        "/capsules", data={"csrf": visitor_token, "title": "Other album"}
    )
    other_id = capsule_id_from(other.text)
    assert visitor.get(media_url.replace(capsule_id, other_id)).status_code == 404
    archivist.completed = ArchivistResult(text="A machine transcript.")
    whole = owner.get(f"/capsules/{capsule_id}/entries?view=whole")
    assert "capsule-gallery" in whole.text
    assert "AI transcription" in whole.text and "A machine transcript." in whole.text
    assert (
        "These are my original words." in whole.text
        and "My recorded memory." in whole.text
    )
    restored = browser_client(settings, RecordingGenerator())

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
    correction = visitor.post(
        f"/capsules/{capsule_id}/entries",
        data={
            "csrf": visitor_token,
            "occurred_on": "2006-05-01",
            "text": "The shop sign was green.",
            "correction": "true",
            "linked_job_id": job_id,
        },
    )
    assert correction.status_code == 200, correction.text
    linked = owner.get(f"/visits/{job_id}")
    assert (
        f'href="/jobs/{job_id}/world"' in linked.text
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


def test_folder_images_are_ordered_retained_bounded_and_public(tmp_path: Path) -> None:
    settings = client_settings(tmp_path)
    worker = RecordingGenerator()
    owner = browser_client(settings, worker)
    token = csrf_from(owner.get("/visits/new").text)
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
        assert (
            stranger.get(f"/jobs/{job_id}/references/{index}").content
            == worker.references[0][index]
        )
    for index in (-1, 3):
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


def research_packet_payload() -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "query": {"location": "Trafalgar Square, London", "year": 1960},
            "items": [
                {
                    "id": "pastvu:1",
                    "provider": "PastVu",
                    "kind": "photograph",
                    "title": "Trafalgar Square",
                    "record_url": "https://pastvu.com/p/1",
                    "capture_from": 1960,
                    "capture_to": 1960,
                    "date_status": "target-year",
                    "date_basis": "PastVu catalogued image date",
                    "excerpt": "Pigeons on the north terrace.",
                },
                {
                    "kind": "photograph",
                    "title": "No usable link",
                    "excerpt": "Skipped without a record URL.",
                },
            ],
        }
    ).encode()


def test_research_packet_upload_joins_the_visit(tmp_path: Path) -> None:
    settings = client_settings(tmp_path)
    worker = RecordingGenerator()
    client = browser_client(settings, worker)
    response = client.post(
        "/api/jobs",
        headers=api_headers(settings),
        data={
            "location": "Trafalgar Square, London",
            "year": "1960",
            "description": "Standing on the north terrace.",
            "use_web": "false",
        },
        files={
            "research_file": (
                "manifest.json",
                research_packet_payload(),
                "application/json",
            )
        },
    )
    assert response.status_code == 200, response.text
    research = response.json()["research_sources"]
    assert [entry["title"] for entry in research] == ["Trafalgar Square"]
    assert research[0]["url"].startswith("https://pastvu.com/p/1")
    assert research[0]["date"] == "1960 (PastVu catalogued image date)"
    assert "PastVu" in research[0]["kind"]
    assert "Saved research context" in worker.submitted[0].prompt
    assert "Pigeons on the north terrace" in worker.submitted[0].prompt


def test_research_packet_rejects_unusable_json(tmp_path: Path) -> None:
    settings = client_settings(tmp_path)
    worker = RecordingGenerator()
    client = browser_client(settings, worker)
    response = client.post(
        "/api/jobs",
        headers=api_headers(settings),
        data={
            "location": "Trafalgar Square, London",
            "year": "1960",
            "description": "Standing on the north terrace.",
            "use_web": "false",
        },
        files={"research_file": ("manifest.json", b"not json", "application/json")},
    )
    assert response.status_code == 422
    assert "research packet" in response.json()["detail"]
    assert not worker.submitted


def writer_client(payload: object, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://writer.test/v1"
    )


def writer_payload(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def test_prompt_writer_composes_with_token(monkeypatch) -> None:
    monkeypatch.setenv("NONSLOP_PROMPT_TOKEN", "wk-id.ws-secret")
    prompt = (
        "integrated_multimodal_description: [Shot 1] A slow pan across the "
        "terrace.\n\noverall_soundscape: Pigeons and distant buses.\n\n"
        "non_diegetic_music: N/A"
    )
    written = write_scene_prompt(
        "integrated_multimodal_description: [Shot 1] evidence packet",
        client=writer_client(writer_payload(prompt)),
    )
    assert written == prompt


def test_prompt_writer_falls_back_without_token_or_bad_reply(
    monkeypatch,
) -> None:
    monkeypatch.delenv("NONSLOP_PROMPT_TOKEN", raising=False)
    packet = "integrated_multimodal_description: [Shot 1] evidence"
    assert write_scene_prompt(packet, client=writer_client(writer_payload("x"))) is None
    monkeypatch.setenv("NONSLOP_PROMPT_TOKEN", "wk-id.ws-secret")
    assert write_scene_prompt("   ", client=writer_client(writer_payload("x"))) is None
    assert (
        write_scene_prompt(packet, client=writer_client(writer_payload("no markers")))
        is None
    )
    assert write_scene_prompt(packet, client=writer_client({}, status=500)) is None


def test_deerflow_evidence_reaches_kimi_h3_and_world(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("NONSLOP_ENV_FILE", str(tmp_path / "unused.env"))
    from nonslop.runtime import cpu_app
    from nonslop.web.demo_visits import DEMO_VISITS, bundled_observations

    monkeypatch.setenv("NONSLOP_PROMPT_TOKEN", "test-writer-token")
    monkeypatch.setenv("NONSLOP_PROMPT_MODEL", "moonshotai/Kimi-K3.1-Pro")
    settings = client_settings(tmp_path)
    worker = RecordingGenerator()
    world = RecordingReconstructor()
    client = browser_client(settings, worker, world)
    headers = api_headers(settings)
    intake = client.get("/visits/new")
    notes = "Historical context. " * 980 + "End of supplied notes."
    refined_prompt = (
        "integrated_multimodal_description: [Shot 1] From the opening photo, the "
        "camera glides sideways. The illuminated tree shifts against the distant "
        "shore and stays in frame.\n\noverall_soundscape: Quiet harbor ambience."
        "\n\nnon_diegetic_music: N/A"
    )
    readings = [
        {
            "photo_number": 1,
            "visible_details": ["Cool lights surround the tree at center."],
            "uncertainty": "The opening photograph does not show the warehouse fronts.",
        }
    ]
    archivist = MagicMock()
    archivist.analyze_images.remote.return_value = ArchivistResult(
        text="The opening image shows an illuminated tree."
    ).model_dump_json()
    h3 = MagicMock()
    h3.generate.remote.return_value = GenerationResult(
        video_path="/outputs/test.mp4", frames=124, width=768, height=448
    ).model_dump_json()

    submitted = client.post(
        "/visits/new",
        data={
            "csrf": csrf_from(intake.text),
            "visit_preset": "hakodate",
            "source_text": notes,
        },
        follow_redirects=False,
    )
    assert submitted.status_code == 303
    task, photos = worker.submitted[0], worker.references[0]
    assert "Begin on the supplied first-frame image" in task.prompt
    assert "Move the camera smoothly sideways or diagonally forward" in task.prompt
    assert "nearby features shift visibly against the background" in task.prompt
    world_url = f"/api/jobs/{task.job_id}/world"
    assert client.post(world_url, headers=headers).status_code == 409
    assert world.submitted == []

    def capture_writer_request(request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content)["messages"]
        camera_guidance = messages[0]["content"]
        assert "sideways or diagonally forward from the opening view" in camera_guidance
        assert "distinctive landmarks remain visible across frames" in camera_guidance
        parts = messages[1]["content"]
        photo_observations, evidence = parts[0]["text"].split("\n\n", 1)
        assert "The opening image shows an illuminated tree." in photo_observations
        packet = json.loads(evidence)
        assert packet["scene"]["source_text"] == notes
        assert packet["reference_metadata"]
        assert packet["saved_research"]
        journal = bundled_observations(DEMO_VISITS["hakodate"].observation_file)
        assert len(journal) == 42
        assert packet["deerflow_observations"] == [
            observation.model_dump(mode="json") for observation in journal
        ]
        assert len(parts) == 3
        assert parts[1]["text"] == "Photo 1 (opening view)"
        assert b64decode(
            parts[2]["image_url"]["url"].split(",", 1)[1]
        ) == normalize_image(photos[0], 768)
        return httpx.Response(
            200,
            json=writer_payload(
                json.dumps({"prompt": refined_prompt, "photo_readings": readings})
            ),
        )

    models = {"MemoryArchivist": archivist, "StreetGenerator": h3}
    monkeypatch.setattr(
        cpu_app.modal.Cls, "from_name", lambda app, name: lambda: models[name]
    )
    with httpx.Client(
        transport=httpx.MockTransport(capture_writer_request),
        base_url="https://writer.test",
    ) as writer:
        monkeypatch.setattr(
            cpu_app,
            "compose_scene_prompt",
            lambda packet, settings, **kwargs: compose_scene_prompt(
                packet, settings, writer, **kwargs
            ),
        )
        worker.completed = GenerationResult.model_validate_json(
            cpu_app.prepare_visit.local(task.model_dump_json(), photos)
        )
    h3_task, h3_photos = h3.generate.remote.call_args.args
    assert json.loads(h3_task)["prompt"] == refined_prompt
    assert "evidence" not in json.loads(h3_task)
    assert h3_photos == photos
    assert worker.completed.used_prompt == refined_prompt
    assert worker.completed.prompt_model == "moonshotai/Kimi-K3.1-Pro"

    response = client.post(world_url, headers=headers)
    assert response.status_code == 200
    assert response.json()["world"]["input_kind"] == "generated_video"
    assert response.json()["result"]["used_prompt"] == refined_prompt
    assert response.json()["result"]["photo_readings"] == readings
    assert world.submitted == [task.job_id]
    assert world.references == [[]]
